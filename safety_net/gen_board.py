import random
import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import interp1d

from .game_physics import CellTypes, GameOfLife


def gen_regions(shape, alpha=1.0, max_regions=5, min_regions=2, use_diag=False):
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
        weights = np.array([len(p) for p in perimeters], dtype=float)
        weights[0] = min(alpha, weights[0]) if len(weights) < max_regions else 1e-10
        if len(weights) <= min_regions:
            weights[1:] = 1e-10
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


def build_fence(mask, shuffle=True):
    """
    Create a fence around unmasked regions such nothing inside the regions
    can escape.

    Note that this is a little bit more aggressive than it strictly needs
    to be.

    Parameters
    ----------
    mask : ndarray, dtype int
        Binary array denoting regions around which to build fences (1) and
        everything else.

    Returns
    -------
    fence : ndarray, dtype int
        Binary array indicating fence locations.
    """
    mask = mask.astype(np.int32)
    _i = np.array([-1,-1,-1,0,0,0,1,1,1], dtype=np.int32)
    _j = np.array([-1,0,1,-1,0,1,-1,0,1], dtype=np.int32)
    neighbors = ndimage.convolve(mask, np.ones((3,3)), mode='wrap')
    fence = np.zeros_like(mask)
    edge_i, edge_j = np.nonzero(mask * neighbors % 9)
    neighbors *= (1 - mask)
    if edge_i.size == 0:
        return fence

    # First pass. Add in fence where needed.
    if shuffle:
        k = np.random.permutation(len(edge_i))
        edge_i = edge_i[k]
        edge_j = edge_j[k]
    for i, j in zip(edge_i, edge_j):
        n_i = (i + _i) % mask.shape[0]
        n_j = (j + _j) % mask.shape[1]
        if (neighbors[n_i, n_j] >= 3).any():
            neighbors[n_i, n_j] -= 1
            fence[i, j] += 1

    # Second pass. Remove fence where unneeded.
    fence_i, fence_j = np.nonzero(fence)
    if shuffle:
        k = np.random.permutation(len(fence_i))
        fence_i = fence_i[k]
        fence_j = fence_j[k]
    for i, j in zip(fence_i, fence_j):
        n_i = (i + _i) % mask.shape[0]
        n_j = (j + _j) % mask.shape[1]
        if (neighbors[n_i, n_j] < 2).all():
            neighbors[n_i, n_j] += 1
            fence[i, j] -= 1

    return fence


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
        decreases that cell's likelihood by a factor of exp(-1/temperature).
    """
    temperature = 0.7
    max_iter = 40
    min_fill = 0.3

    penalty_params = {
        CellTypes.empty: [(0, -2,), (5, 0), (10, 10)],
        CellTypes.life: [(0, 0), (1, 0)],
        # CellTypes.wall: [(0, 1), (0.05, 4)],
        CellTypes.plant: [(0, 1), (0.05, 3)],
        # CellTypes.weed: [(0, 3), (0.05, 10)],
        # CellTypes.predator: [(0, 3), (0.05, 10)],
        # CellTypes.fountain: [(0, 3), (0.01, 10)],
        # CellTypes.ice_cube: [(0, 3), (0.05, 10)],
    }

    def __init__(
            self, temperature=temperature, penalty_params=penalty_params,
            max_iter=max_iter, min_fill=min_fill):
        self.temperature = temperature
        self.min_fill = min_fill
        self.pfuncs = {
            key: interp1d(*np.array(val).T, fill_value="extrapolate")
            for key, val in penalty_params.items()
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

        x = -(violations + penalties[:,None]) / self.temperature
        x += (1-mask) * -1e10  # If mask is zero, drop the logit to -inf
        x -= np.max(x)
        P = np.exp(x)
        P /= np.sum(P)
        k = np.random.choice(P.size, p=P.ravel())
        return k // violations.shape[1], k % violations.shape[1]


def gen_still_life(board, mask=None, params=GenLifeParams(), num_retries=3):
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

    # The following are used to index the Moore neighborhood
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
        board *= (1 - mask)
        if num_retries > 0:
            return gen_still_life(board, mask, params, num_retries-1)
        else:
            print("Failed to converge! Returning empty board.")

    return board


def gen_region(board, goals, mask, fences, difficulty, region_type=None):
    def dscale(x, y=None, low=None, high=None):
        """
        Do linear interpolation based on difficulty.

        If low and high are defined, pick randomly between them.
        """
        k = np.searchsorted(x, difficulty, side='right')
        k1 = max(0, k-1)
        k2 = min(k, len(x) - 1)
        r = 1 if k1 == k2 else (difficulty - x[k1]) / (x[k2] - x[k1])
        if y is not None:
            return (1-r) * y[k1] + r * y[k2]
        else:
            y_low = (1-r) * low[k1] + r * low[k2]
            y_high = (1-r) * high[k1] + r * high[k2]
            return y_low + np.random.random() * (y_high - y_low)

    region_type_weights = {
        # min level, weight at min, max level, weight at max
        "still": dscale([1,1,10], [0,1,3]),
        "build": 1,
        "append": dscale([2,2,10], [0,1,2]),
        # "destroy": dscale([3,3,10], [0,1,2]),
        # "prune": dscale([4,4,10], [0,1,2]),
        # "spawner": dscale([3,3,10], [0,2,4]),
        # "oscillator": dscale([3,3,10], [0,2,4]),
        "fountain": dscale([6,6,10], [0,1.5,3]),
        "grow": dscale([7,7,10], [0,2,3]),
    }
    if not fences.any() and 'spawner' in region_type_weights:
        del region_type_weights["spawner"]
    region_type = region_type or np.random.choice(
        list(region_type_weights.keys()),
        p=np.array(list(region_type_weights.values())) /
        sum(region_type_weights.values())
    )

    # temperature < 0.3 tends to not converge, or converge very slowly
    # temperature = 0.4, fill = 0.05 yields pretty simple patterns
    # temperature = 1.5, fill = 0.4 yields pretty complex patterns
    temperature = dscale([0, 5, 10], low=[0.4, 0.4, 0.6], high=[0.4, 0.8, 2.0])
    min_fill = dscale([0, 5, 10], low=[0.05, 0.1, 0.3], high=[0.1, 0.2, 0.4])
    fence_frac = dscale([0, 10], low=[1.1, -0.2], high=[2, 0.8])
    extra_walls = dscale([0, 10], low=[0,0], high=[0, 0.1])
    crate_frac = dscale([0, 5, 10], low=[0, 0.2, -0.1], high=[0, 1, 0.5])

    penalty_params = {
        CellTypes.empty: [
            # Empty spaces are penalized so as to make them no less likely
            # than living spaces when the board is sparse.
            (0, 2),
            (0.9 * (1/min_fill - 1), 2),
            (1/min_fill - 1, 0),
            (1/min_fill, 0),
        ],
        CellTypes.life: [(0, 0), (1, 0)],
        CellTypes.wall: [(0, 1), (0.1, 4)],
        CellTypes.plant: [(0, 1), (0.1, 3)],
        CellTypes.weed: [(0, 1), (0.1, 10)],
        CellTypes.predator: [(0, 1), (0.1, 10)],
        CellTypes.ice_cube: [(0, 1), (0.1, 10)],
    }
    penalty_params_prob = {
        CellTypes.empty: 1,
        CellTypes.life: 1,
        CellTypes.wall: dscale([0,2,2,10], [0, 0, 0.25, 0.5]),
        CellTypes.plant: dscale([0,3,3,10], [0, 0, 0.25, 0.5]),
        CellTypes.weed: dscale([0,5,5,10], [0, 0, 0.15, 0.25]),
        CellTypes.predator: dscale([0,8,8,10], [0, 0, 0.15, 0.15]),
        CellTypes.ice_cube: dscale([0,6,6,10], [0, 0, 0.05, 0.05]),
    }
    for cell_type, prob in penalty_params_prob.items():
        if prob < np.random.random():
            del penalty_params[cell_type]
    params = GenLifeParams(
        temperature=temperature,
        min_fill=min_fill,
        penalty_params=penalty_params,
    )

    fence_mask = mask & (fences == 0)

    if region_type == "still":
        board = gen_still_life(board, fence_mask, params)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
    elif region_type == "build":
        board = gen_still_life(board, fence_mask, params)
        alive_mask = ((board & CellTypes.alive) > 0) & mask
        life_mask = (board == CellTypes.life) & mask
        board *= (1 - life_mask)
        goals += alive_mask * CellTypes.color_b
    elif region_type == "destroy":
        board = gen_still_life(board, fence_mask, params)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_r
    elif region_type == "append":
        board = gen_still_life(board, fence_mask, params)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
        mask2 = fence_mask & (board == 0)
        board = gen_still_life(board, mask2, params)
        alive_mask = ((board & CellTypes.alive) > 0) & mask2
        board *= ~alive_mask
        goals += alive_mask * CellTypes.color_b
    elif region_type == "grow":
        board = gen_still_life(board, fence_mask, params)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
        mask2 = fence_mask & (board == 0)
        board = gen_still_life(board, mask2, params)
        alive_mask = ((board & CellTypes.alive) > 0) & mask2
        board *= ~alive_mask
        goals += alive_mask * CellTypes.color_g
    elif region_type == "prune":
        board = gen_still_life(board, fence_mask, params)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
        mask2 = fence_mask & (board == 0)
        board = gen_still_life(board, mask2, params)
        life_mask2 = (board == CellTypes.life) & mask2
        board += life_mask2 * CellTypes.color_r
    elif region_type == "spawner":
        fence_frac = 1.0
        color_weights = {
            CellTypes.empty: 1,
            CellTypes.color_g: 1,
            CellTypes.color_r: dscale([5,6], [0,0.5]),
            CellTypes.color_b: dscale([6,7], [0,1]),
        }
        color = np.random.choice(
            list(color_weights.keys()),
            p=np.array(list(color_weights.values())) /
            sum(color_weights.values())
        )
        interior_mask = ndimage.minimum_filter(mask, size=3, mode='wrap')
        tree_frac = dscale([0,3,10], [0, 0, 0.1])
        tree_mask = interior_mask & (board == 0)
        tree_mask &= np.random.random(board.shape) < tree_frac
        board += tree_mask * (CellTypes.tree + color)
        life_frac = 0.3
        life_mask = interior_mask & (board == 0)
        life_mask &= np.random.random(board.shape) < life_frac
        board += life_mask * (CellTypes.life + color)
        i, j = np.nonzero(interior_mask)
        if len(i) > 0:
            k = np.random.randint(len(i))
            board[i[k], j[k]] = CellTypes.spawner + color
    elif region_type == "fountain":
        fountain_mask = fence_mask * (np.random.random(board.shape) < 0.04)
        fountain_neighbor = ndimage.maximum_filter(fountain_mask, size=3, mode='wrap')
        color = np.random.choice([
            CellTypes.color_r, CellTypes.color_g, CellTypes.color_b])
        board += fountain_mask * (CellTypes.fountain + color)
        goals += (mask & fountain_neighbor & ~fountain_mask) * color
    elif region_type == "oscillator":
        raise NotImplemented
    else:
        raise ValueError("Unexpected region type: '{}'".format(region_type))

    # Make some of the life types hardened
    life_mask = (board & ~CellTypes.rainbow_color == CellTypes.life) * mask
    frac_hardlife = dscale([5, 5, 10], low=[0, -1, -0.5], high=[0, 0, 1])
    hardlife_mask = np.random.random(board.shape) < frac_hardlife
    board -= life_mask * hardlife_mask * CellTypes.destructible

    # Remove fences and add extra walls in the middle
    wall_mask = mask & (board == 0) & (goals == 0)
    wall_mask &= (np.random.random(board.shape) < extra_walls)
    board += wall_mask * CellTypes.wall
    neighbors = ndimage.convolve(
        board & CellTypes.alive, np.ones((3,3)), mode='wrap')
    no_fence_mask = mask & (fences > 0) & (neighbors != 3)
    no_fence_mask &= (np.random.random(board.shape) > fence_frac)
    board *= ~no_fence_mask
    crate_mask = mask & (board == CellTypes.wall)
    crate_mask &= (np.random.random(board.shape) < crate_frac)
    board += crate_mask * CellTypes.movable


def gen_game(board_shape=(35,35), difficulty=10, has_fences=True, max_regions=5):
    regions = gen_regions(board_shape, max_regions=max_regions)
    fences = has_fences * build_fence(regions > 0)
    goals = (regions == 0).astype(np.int16)
    goals *= CellTypes.rainbow_color
    board = fences.astype(np.int16) * CellTypes.wall
    for k in np.unique(regions)[1:]:
        mask = regions == k
        region_type = 'build' if k == 1 else None
        gen_region(board, goals, mask, fences, difficulty, region_type)
    i, j = np.nonzero(regions == 0)
    k1, k2 = np.random.choice(len(i), size=2, replace=False)
    board[i[k1], j[k1]] = CellTypes.player
    board[i[k2], j[k2]] = CellTypes.level_exit | CellTypes.color_r
    game = GameOfLife()
    game.deserialize({
        'board': board,
        'goals': goals,
        'agent_loc': (j[k1], i[k1]),
    })
    return game


def _main(play=True):
    """Just for testing."""
    from .asci_renderer import render_board
    from .game_loop import GameLoop
    if play:
        game_loop = GameLoop()
        game_loop.centered_view = True
        game_loop.play(gen_game())
    else:
        print(render_board(gen_game()))


if __name__ == "__main__":
    _main()
