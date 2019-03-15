import random
import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import interp1d

from .game_physics import CellTypes


def gen_regions(shape, alpha=1.5, max_regions=10, use_diag=False):
    ring = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int16)
    if use_diag:
        adjacent = np.array([
            [-1,0,1,-1,1,-1,0,1],
            [-1,-1,-1,0,0,1,1,1]], dtype=np.int16).T
    else:
        adjacent = np.array([
            [-1,0,0,1],
            [0,-1,1,0]], dtype=np.int16).T
    nearby = np.meshgrid([-2,-1,0,1,2], [-2,-1,0,1,2])

    board = np.zeros(shape, dtype=np.int16)
    perimeters = [{
        (i, j) for i, j in zip(*np.nonzero(board == 0))
    }]
    exclusions = [set()]
    while sum(len(p) for p in perimeters) > 0:
        weights = [len(p) for p in perimeters]
        weights[0] = min(alpha, weights[0]) if len(weights) < max_regions else 1e-10
        weights /= np.sum(weights)
        k = np.random.choice(len(perimeters), p=weights)
        i, j = random.sample(perimeters[k], 1)[0]
        perimeters[0].discard((i, j))
        perimeters[k].discard((i, j))
        if (i, j) in exclusions[k]:
            continue
        exclusions[0].add((i,j))
        exclusions[k].add((i,j))
        b = board[(i+nearby[0]) % shape[0], (j+nearby[1]) % shape[1]]
        b[2,2] = k or -1
        num_neighbors = signal.convolve2d(b != 0, ring, mode='valid')
        num_foreign = signal.convolve2d((b > 0) & (b != k), ring, mode='valid')
        if ((num_foreign > 0) & (num_neighbors > 2)).any() or num_foreign[1,1] > 0:
            continue
        # Add to the board
        if k == 0:
            k = len(perimeters)
            perimeters.append(set())
            exclusions.append(set())
        board[i, j] = k
        for i2, j2 in (adjacent + (i, j)) % shape:
            if board[i2, j2] == 0:
                perimeters[k].add((i2, j2))
    return board


def check_violations(alive, neighbors, preserved, inhibited):
    """
    Number of violations on each cell.

    Dead cells can have only one violation. Living cells can have up to 5,
    one for each neighbor that's in excess of the survival limit.
    """
    dead = 1 - alive
    will_be_born = neighbors == 3
    survival_risk = np.abs(2*neighbors - 5) // 2
    return (
        dead * (inhibited == 0) * will_be_born +
        alive * (preserved == 0) * survival_risk
    )


class GenLifeParams(object):
    """
    Set of parameters to modify still life generation.

    Parameters
    ----------
    temperature : float
        Determines the complexity of the resulting pattern.
        A very low temperature results in simple patterns (mostly 2x2 blocks),
        whereas a high temperature tends to create large branching structures.
        This should generally be in the range of (0, 2) (must be greater than
        zero).
    max_iter : float
        Maximum number of iterations allowed per cell.
    min_fill : float
        Minimum proportion of the interior to be filled. The cutoff will
        decrease linearly with the number of iterations, so the effective
        minimum fill will be zero at the maximum iteration.
    penalty_params : dict
        Each item in `penalty_params` is a list of x-y tuples to be used in
        interpolation of that cell type's penalty. The x-value is the
        proportion of filled cells that have that cell type, and the y-value
        is the logarithmic penalty. That is, increasing a penalty by 1
        decreases that cell's likelihood by a factor of e.
    """
    temperature = 0.1
    max_iter = 100
    min_fill = 0.1

    penalty_params = {
        CellTypes.empty: [(0, -2,), (0.5, 0), (.9, 0), (1, 5)],
        CellTypes.life: [(0, 0), (1, 0)],
        CellTypes.wall: [(0, 1), (0.05, 4)],
        CellTypes.plant: [(0, 1), (0.05, 3)],
        # CellTypes.weed: [(0, 3), (0.05, 10)],
        # CellTypes.predator: [(0, 3), (0.05, 10)],
        # CellTypes.fountain: [(0, 3), (0.01, 10)],
        # CellTypes.ice_cube: [(0, 3), (0.05, 10)],
    }

    def __init__(
            self, temperature=temperature, penalty_params=penalty_params,
            max_iter=max_iter, min_fill=min_fill):
        self.pfuncs = {
            key: interp1d(*np.array(val).T, fill_value="extrapolate")
            for key, val in self.penalty_params.items()
        }

    def can_stop(self, num_iter, totals):
        """
        Determine whether or not the totals are sufficient to stop.
        """
        if totals['area'] < 4 or totals['interior'] < 2:
            return True
        effective_area = 0.25*totals['area'] + 0.75*totals['interior']
        t = num_iter / (self.max_iter * totals['area'])
        num_filled = totals['area'] - totals[CellTypes.empty]
        return num_filled > effective_area * self.min_fill * (1 - t)

    def pick_cell(self, violations, mask, cell_list, num_iter, totals):
        """
        Pick a cell to switch by weighting violations.

        Parameters
        ----------
        violations : ndarray of shape (num_cell_types, num_cells)
            The number of violations associated with setting each cell
            to each of the available cell types.
        mask : ndarray of shape (num_cells,)
        cell_list : ndarray of shape (num_cell_types,)
            Each entry is a bit flag representation for a particular cell type.
        num_iter : int
        totals : dict
            Total number of cells of each type.
            Has one entry for each of the different cell types, plus an
            'area' entry denoting the total area of the field, and an
            'interior' entry denoting the interior area of the field.

        Returns
        -------
        type_idx : int
            The index of the type of cell to switch to.
        cell_idx : int
            The index of the cell to switch.
        """
        area = totals['area']
        filled = max(1, area - totals[CellTypes.empty])
        penalties = np.array([
            self.pfuncs[cell_type](totals[cell_type] / filled)
            if cell_type in self.pfuncs else 100.0
            for cell_type in cell_list
        ])

        # Was thinking about factoring num_iter into this, but doesn't seem
        # necessary.
        # t = num_iter / (self.max_iter * totals['area'])

        x = -violations / self.temperature - penalties[:, None]
        x += (1-mask) * -1e10  # If mask is zero, drop the logit to -inf
        x -= np.max(x)
        P = np.exp(x)
        P /= np.sum(P)
        k = np.random.choice(P.size, p=P.ravel())
        return k // violations.shape[1], k % violations.shape[1]


def gen_still_life(board, mask=None, params=GenLifeParams()):
    # First, set up some useful constants
    CT = CellTypes
    shape = board.shape
    cell_list = np.array([
        CT.empty,
        CT.life,
        CT.wall,
        CT.plant,
        CT.fountain,
        CT.weed,
        CT.predator,
        CT.ice_cube,
    ], dtype=np.int16)

    # The following our used to index the Moore neighborhood
    _i = np.array([-1,-1,-1,0,0,0,1,1,1], dtype=np.int16)
    _j = np.array([-1,0,1,-1,0,1,-1,0,1], dtype=np.int16)
    _i_ring = np.array([-1,-1,-1,0,0,1,1,1], dtype=np.int16)
    _j_ring = np.array([-1,0,1,-1,1,-1,0,1], dtype=np.int16)
    _ring = np.array([1,1,1,1,0,1,1,1,1], dtype=np.int16)
    _ring3 = _ring.reshape(3,3)
    _dot = 1 - _ring
    _ones = np.ones((9,9), dtype=np.int16)
    _I = np.add.outer(_i, _i)
    _J = np.add.outer(_j, _j)

    # Initialize the board with default values
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    alive = (board & CT.alive) >> CT.alive_bit
    preserving = (board & CT.preserving) >> CT.preserving_bit
    inhibiting = (board & CT.inhibiting) >> CT.inhibiting_bit
    frozen = (board & CT.frozen) >> CT.frozen_bit
    neighbors = ndimage.convolve(alive, _ring3, mode="wrap")
    preserved = ndimage.convolve(preserving, _ring3, mode="wrap") + frozen
    inhibited = ndimage.convolve(inhibiting, _ring3, mode="wrap") + frozen

    totals = {
        cell_type: np.sum(mask * (board & cell_type == cell_type))
        for cell_type in cell_list
    }
    totals['area'] = np.sum(mask)
    totals['interior'] = np.sum(ndimage.convolve(
        mask.astype(int), [[0,1,0],[1,1,1],[0,1,0]], mode='wrap') == 5)

    all_idx = {(i, j) for i, j in zip(*np.nonzero(mask))}
    all_violations = check_violations(alive, neighbors, preserved, inhibited)
    bad_idx = {(i, j) for i, j in zip(*np.nonzero(mask * all_violations))}

    for num_iter in range(int(params.max_iter * totals['area'])):
        if not bad_idx and params.can_stop(num_iter, totals):
            break
        i0, j0 = random.sample(bad_idx or all_idx, 1)[0]

        # Look at all of the neighborhoods that overlap that cell.
        i = (_i + i0) % board.shape[0]
        j = (_j + j0) % board.shape[1]
        I = (_I + i0) % board.shape[0]
        J = (_J + j0) % board.shape[1]
        # The sub-board is 9x9. Each row represents the neighborhood around
        # one cell which is itself in the neighborhood of i0, j0.
        # The center column is therefore the central cell in each of the nine
        # overlapping neighborhoods.
        b = board[I, J]
        a0 = alive[I, J]
        n0 = neighbors[I, J]
        p0 = preserved[I, J]
        h0 = inhibited[I, J]

        # Zero out the center column, plus neighbors, preserved, inhibited.
        center = b[:, 4]
        n0 -= np.multiply.outer((center & CT.alive) > 0, _ring)
        p0 -= np.multiply.outer((center & CT.preserving) > 0, _ring)
        h0 -= np.multiply.outer((center & CT.inhibiting) > 0, _ring)
        p0[:, 4] -= (center & CT.frozen) >> CT.frozen_bit
        h0[:, 4] -= (center & CT.frozen) >> CT.frozen_bit
        b[:, 4] = 0
        a0[:, 4] = 0

        # Now for each set of cell types, change the center column to that cell
        # type and check the number of violations that occurs.
        a1 = a0 + _dot
        n1 = n0 + _ring
        p1 = p0 + _dot
        h1 = h0 + _dot
        p2 = h2 = _ones
        # The penalties for different cell types depend on how many cells
        # have already been produced of that type.

        violations = check_violations(
            np.array([a0, a1, a0, a1, a0, a1, a1, a0]),
            np.array([n0, n1, n0, n1, n0, n1, n1, n0]),
            np.array([p0, p0, p1, p1, p2, p2, p1, p2]),
            np.array([h0, h0, h1, h1, h1, h1, h2, h2]),
        )  # This is an 8x9x9 array.
        type_idx, cell_idx = params.pick_cell(
            np.sum(violations, axis=2), mask[i, j], cell_list, num_iter, totals)

        i_new = i[cell_idx]
        j_new = j[cell_idx]
        old_cell = board[i_new, j_new]
        new_cell = cell_list[type_idx]

        # Update the board with the new cell.
        # Adjust neighbors, preserved, inhibited accordingly.
        i_ring = (_i_ring + i_new) % shape[0]
        j_ring = (_j_ring + j_new) % shape[1]
        board[i_new, j_new] = new_cell
        delta_alive = (new_cell & CT.alive) - (old_cell & CT.alive)
        delta_alive >>= CT.alive_bit
        delta_frozen = (new_cell & CT.frozen) - (old_cell & CT.frozen)
        delta_frozen >>= CT.frozen_bit
        delta_preserving = (new_cell & CT.preserving) - (old_cell & CT.preserving)
        delta_preserving >>= CT.preserving_bit
        delta_inhibiting = (new_cell & CT.inhibiting) - (old_cell & CT.inhibiting)
        delta_inhibiting >>= CT.inhibiting_bit
        alive[i_new, j_new] += delta_alive
        preserved[i_new, j_new] += delta_frozen
        inhibited[i_new, j_new] += delta_frozen
        neighbors[i_ring, j_ring] += delta_alive
        preserved[i_ring, j_ring] += delta_preserving
        inhibited[i_ring, j_ring] += delta_inhibiting
        totals[new_cell] += 1
        totals[old_cell] -= 1

        # Update the violations
        new_violations = violations[type_idx, cell_idx]
        i_block = (i_new + _i) % shape[0]
        j_block = (j_new + _j) % shape[1]
        all_violations[i_block, j_block] = new_violations
        for i, j, val in zip(i_block, j_block, new_violations):
            if val:
                bad_idx.add((i, j))
            else:
                bad_idx.discard((i, j))
    else:
        print("Failed to converge!", bad_idx)
    return board


def _main():
    from .asci_renderer import render_board
    from .game_physics import GameOfLife
    # just for testing
    shape = (25, 25)
    regions = gen_regions(shape)
    mask = regions == 1
    board = np.zeros(shape, dtype=np.int16)
    board = gen_still_life(board, mask)
    state = GameOfLife()
    state.deserialize({'board': board, 'goals': (regions & 7) << CellTypes.color_bit})
    print(render_board(state))


if __name__ == "__main__":
    _main()
