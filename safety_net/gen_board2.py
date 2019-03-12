import numpy as np
from scipy import ndimage

from .game_physics import CellTypes


def partition_board(board, alpha=1, max_regions=10):
    """
    Partition the board into contiguous regions using a Dirichlet process.

    Parameters
    ----------
    regions : ndarray of dtype int or tuple
        Integer of the starting regions.
    alpha : float
        Concentration parameter. Larger value means more regions.
    max_regions : int

    Returns
    -------
    board : ndarray
        Each board cell is an integer associated with its region. All regions
    """
    # The first pass at this isn't meant to be speedy.
    # It could be made a lot faster by using smarter data structures, or by
    # implementing in C.
    # 15x15 board takes about 15ms
    shape = board.shape
    neighborhood = np.array([[0,1,0],[1,0,1],[0,1,0]])
    reg, inv, count = np.unique(board, return_inverse=True, return_counts=True)
    if reg[0] < 0:
        raise ValueError("Board regions must be non-negative integers.")
    if reg[0] > 0:
        # All regions are already filled. Our job is done.
        return board.copy()
    board = inv.reshape(board.shape)  # Makes sure that numbers don't skip.
    perimeters = [
        set(np.flatnonzero(board == 0).tolist())
    ]
    for n in range(1, len(reg)):
        idx = np.flatnonzero(
            (board == 0) &
            ndimage.convolve(board == n, neighborhood, mode='wrap'))
        perimeters.append(set(idx.tolist()))
    board = board.ravel()
    while perimeters[0]:
        weights = [len(p) for p in perimeters]
        weights[0] = alpha if len(weights) < max_regions else 0
        weights /= np.sum(weights)
        k = np.random.choice(len(perimeters), p=weights)
        idx = np.random.choice(list(perimeters[k]))
        if k == 0:
            k = len(perimeters)
            perimeters.append(set())
        board[idx] = k
        for perim in perimeters:
            perim.discard(idx)
        r, c = idx // shape[1], idx % shape[0]
        i1 = c + ((r + 1) % shape[0]) * shape[1]
        i2 = c + ((r - 1) % shape[0]) * shape[1]
        i3 = (c + 1) % shape[1] + r * shape[1]
        i4 = (c - 1) % shape[1] + r * shape[1]
        for i in [i1, i2, i3, i4]:
            if board[i] > 0:
                continue
            perimeters[k].add(i)
    return board.reshape(shape)


def discard_boundaries(board, full=True):
    """
    Zero out regions at their boundaries.

    This is designed such that any pattern that's constrained to the interior
    of its region will never be able to affect a pattern that's constrained
    to the interior of another region. If the interiors went all the way to the
    edges, then there could be communication that goes across the edge.

    The Moore neighborhood is a little more aggressive than it needs to be,
    while the Von Neumann neighborhood is a little bit less aggressive.
    Should probably just stick with the
    """
    von_neumann = [[0,1,0],[1,0,1],[0,1,0]]
    moore = [[1,1,1],[1,0,1],[1,1,1]]
    neighborhood = moore if full else von_neumann
    bmin = ndimage.minimum_filter(board, footprint=neighborhood, mode='wrap')
    bmax = ndimage.maximum_filter(board, footprint=neighborhood, mode='wrap')
    return np.where(bmax == bmin, bmax, 0)


# ------------------

_ring = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int16)
_dot = 1 - _ring


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


def gen_still_life(board, mask=None, temperature=0.5, maxiter=10000):
    # First, set up some useful constants
    CT = CellTypes
    shape = board.shape
    num_seeds = 3
    temperature = 0.5
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
    _i = np.array([-1,0,1,-1,0,1,-1,0,1])
    _j = np.array([-1,-1,-1,0,0,0,1,1,1])
    _i_ring = np.array([-1,0,1,-1,1,-1,0,1])
    _j_ring = np.array([-1,-1,-1,0,0,1,1,1])
    _ring = np.array([1,1,1,1,0,1,1,1,1])
    _ring3 = _ring.reshape(3,3)
    _dot = 1 - _ring
    _I = np.add.outer(_i, _i)
    _J = np.add.outer(_j, _j)

    # Initialize the board with default values
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    i1, j1 = np.nonzero(mask)
    num_seeds = min(num_seeds, len(i1))
    if num_seeds == 0:
        return board
    k = np.random.choice(len(i1), num_seeds, replace=False)
    board[i1[k], j1[k]] = CellTypes.life

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

    for _ in range(maxiter):
        # At each iteration, pick a cell that isn't still
        # (i.e., that has the wrong number of neighbors)
        i_bad, j_bad = np.nonzero(check_violations(
            alive, neighbors, preserved, inhibited))
        if len(i_bad) == 0:
            break
        k = np.random.randint(len(i_bad))
        i0 = i_bad[k]
        j0 = j_bad[k]

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
        p2 = h2 = 1
        # The penalties for different cell types depend on how many cells
        # have already been produced of that type.
        penalties = np.array([0, 0, 1.5, 1.5, 2, 2, 2, 3])
        if totals[CT.life] < 15:  ## Temporary!
            penalties = np.array([5, 0, 9,9,9,9,9,9]) * 5
        else:
            penalties = np.array([0, 0, 9,9,9,9,9,9]) * 5
        # This is going to be an 8x9 array.
        violations = np.array([
            np.sum(check_violations(a0, n0, p0, h0), axis=1),  # empty
            np.sum(check_violations(a1, n1, p0, h0), axis=1),  # life
            np.sum(check_violations(a0, n0, p1, h1), axis=1),  # wall
            np.sum(check_violations(a1, n1, p1, h1), axis=1),  # plant
            np.sum(check_violations(a0, n0, p2, h1), axis=1),  # fountain
            np.sum(check_violations(a1, n1, p2, h1), axis=1),  # weed
            np.sum(check_violations(a1, n1, p1, h2), axis=1),  # predator
            np.sum(check_violations(a0, n0, p2, h2), axis=1),  # ice_cube
        ]) + penalties[:, np.newaxis]

        # Change the penalties to probabilities, and select a new cell
        x = -violations / temperature
        x += (1-mask[i, j]) * -1e10  # If mask is zero, drop the logit to -inf
        x -= np.max(x)
        P = np.exp(x)
        P /= np.sum(P)
        k = np.random.choice(P.size, p=P.ravel())
        i_new = i[k % 9]
        j_new = j[k % 9]
        old_cell = board[i_new, j_new]
        new_cell = cell_list[k // 9]

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
    else:
        print("Failed to converge!")
    return board


def _main():
    from .asci_renderer import render_board
    from .game_physics import GameOfLife
    # just for testing
    shape = (25, 25)
    regions = partition_board(np.zeros(shape, dtype=np.int16), alpha=3)
    regions = discard_boundaries(regions, False)
    mask = regions == 1
    board = np.zeros(shape, dtype=np.int16)
    board = gen_still_life(board, mask)
    state = GameOfLife()
    state.deserialize({'board': board, 'goals': (regions & 7) << CellTypes.color_bit})
    print(render_board(state))


if __name__ == "__main__":
    _main()
