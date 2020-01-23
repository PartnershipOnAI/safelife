"""
Procedural generation of SafeLife levels.
"""

import numpy as np
from scipy import ndimage, signal

from .safelife_game import CellTypes, SafeLifeGame
from .random import coinflip, get_rng
from . import speedups

import logging
logger = logging.getLogger(__name__)

COLORS = {
    'black': np.uint16(0),
    'red': CellTypes.color_r,
    'green': CellTypes.color_g,
    'blue': CellTypes.color_b,
    'yellow': CellTypes.color_r | CellTypes.color_g,
    'magenta': CellTypes.color_r | CellTypes.color_b,
    'cyan': CellTypes.color_g | CellTypes.color_b,
    'white': CellTypes.rainbow_color
}


def make_partioned_regions(shape, alpha=1.0, max_regions=5, min_regions=2):
    """
    Create a board with distinct regions.

    Each region is continuous and separated from all other regions by at least
    two cells. The regions are generated using a Dirichlet process in which
    new cells are added to existed regions with a probability proportional
    to their boundary.

    Parameters
    ----------
    shape : tuple (int, int)
        Shape of the board to generate.
    alpha : float
        Hyperparameter for the Dirichlet process. Larger values will tend to
        create more distinct regions.
    min_regions : int
    max_regions : int

    Returns
    -------
    array
        The output array is filled with different integers for each of the
        different regions. Zero values indicate border areas between regions.
    """
    ring = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int16)
    adjacent = np.array([  # Diagonals don't count as adjacent
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
        weights[0] = min(alpha, weights[0]) if len(weights) <= max_regions else 1e-10
        if len(weights) <= min_regions:
            weights[1:] = 1e-10
        weights /= np.sum(weights)
        k = get_rng().choice(len(perimeters), p=weights)
        plist = list(perimeters[k])
        i, j = plist[get_rng().choice(len(plist))]
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
    Create a fence around unmasked regions such that nothing inside the regions
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
        k = get_rng().permutation(len(edge_i))
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
        k = get_rng().permutation(len(fence_i))
        fence_i = fence_i[k]
        fence_j = fence_j[k]
    for i, j in zip(fence_i, fence_j):
        n_i = (i + _i) % mask.shape[0]
        n_j = (j + _j) % mask.shape[1]
        if (neighbors[n_i, n_j] < 2).all():
            neighbors[n_i, n_j] += 1
            fence[i, j] -= 1

    return fence


def _fix_random_values(val):
    if not isinstance(val, dict):
        return val
    if 'choices' in val:
        choices = val['choices']
        if isinstance(choices, list):
            keys = choices
            vals = np.ones(len(choices))
        elif isinstance(choices, dict):
            keys = list(choices.keys())
            vals = np.array(list(choices.values()))
        else:
            raise ValueError(
                "The 'choices' object must either be a list of options or a"
                " dictionary in which each key is associate with a specific"
                " probability for picking that key.")
        if (vals < 0).any() or np.sum(vals) <= 0:
            raise ValueError(
                "The values for different choices must be non-negative and their "
                "sum must be positive.")
        return get_rng().choice(keys, p=vals / np.sum(vals))
    if 'uniform' in val:
        low, high = np.array(val['uniform'])
        return (low + (high - low) * get_rng().random()).tolist()
    else:
        return {key: _fix_random_values(x) for key, x in val.items()}


def _gen_pattern(board, mask, seeds=None, num_retries=10, **kwargs):
    # temperature < 0.3 tends to not converge, or converge very slowly
    # temperature = 0.4, fill = 0.05 yields pretty simple patterns
    # temperature = 1.5, fill = 0.4 yields pretty complex patterns
    try:
        min_fill = kwargs.setdefault('min_fill', 0.2)
        max_fill = kwargs.pop('max_fill', min_fill * 2)
        new_board = speedups.gen_pattern(board, mask, seeds=seeds, **kwargs)
        working_area = mask & speedups.NEW_CELL_MASK
        new_cells = new_board != 0
        fill_ratio = np.sum(new_cells * working_area) / np.sum(working_area)
        if fill_ratio > max_fill:
            if num_retries > 0:
                kwargs['max_fill'] = 1.07 * max_fill
                return _gen_pattern(board, mask, seeds, num_retries-1, **kwargs)
            else:
                logger.debug("gen_pattern produced an overfull pattern. "
                      "num_retries exceeded; no patterns added.")
                return board
        return new_board
    except speedups.InsufficientAreaException:
        return board
    except speedups.MaxIterException:
        if num_retries > 0:
            kwargs['min_fill'] *= 0.94
            kwargs['max_fill'] = max_fill
            return _gen_pattern(board, mask, seeds, num_retries-1, **kwargs)
        else:
            logger.debug("gen_pattern did not converge! "
                  "num_retries exceeded; no patterns added.")
            return board
    except speedups.BoardGenException:
        return board


def _make_lattice(h, w, col_skip, row_skip, stagger):
    rows = np.arange(h)[:, np.newaxis]
    cols = np.arange(w)[np.newaxis, :]
    return (rows % row_skip < 1) & (
        (cols + (rows//row_skip)*stagger) % col_skip < 1)


def populate_region(mask, layer_params):
    """
    Populates an isolated region of the board.

    For examples of different region types, see
    ``safelife/levels/random/defaults.yaml``.

    Parameters
    ----------
    mask : array like, boolean grid
        The region will consist of all areas that are 'True' in the mask.
    layer_params : list
        Each layer should contain a set of parameters for that layer's draw
        operations. See `Other Parameters` for what each layer can contain.

    Other Parameters
    ----------------
    color : string
        Determines the color of spawners and life-like cells.
        Should be one of *black*, *red*, *green*, *blue*, *yellow*, *magenta*,
        *cyan*, and *white*.
    fences : float
        Proportion of the layer's boundary which should be "fenced" with walls.
        Patterns in a fully fenced layer won't be able to escape the fenced
        region without outside help.
    spawners : float
        Proportion of the layer's available area that are populated with
        spawner cells. Note that spawners will almost always disrupt existing
        patterns.
    pattern : dict
        If present, should contain a set of parameters to be passed to
        :func:`speedups.gen_pattern` to create new life-like patterns, either
        oscillators or still lifes. Note that the pattern can have period zero
        to produce unstable patterns, and additional *max_fill* and
        *num_retries* values can be passed in to reject certain results.
    tree_lattice : bool
        If True, a lattice of tree objects is added to the region.
        Tree lattices make it so that disrupted cells tend to grown chaotically
        rather than collapse.
    movable_walls : float
        Proportion of wall objects in this layer that pushable (i.e., crates).
    movable_trees : float
        Proportion of tree objects in this layer that pushable.
    hardened_life : float
        Proportion of life objects in this layer that are hardened.
        Hardened life objects cannot be directly removed by the agent.
    buffer_zone : int
        If non-zero, adds a buffer around all life-like cells in this (and
        prior) layers. Subsequent layers will not add new cells to the buffer
        zone.
    target : {'board', 'goals', 'both'}
        Determines whether new cells should get added to the board, as goals,
        or both.
    fountains : float
        Proportion of the region that's populated with special fountain cells.
        Fountains preserve any life that touches them, and they are always
        surrounded by goal cells.
    """

    from .speedups import (
        NEW_CELL_MASK, CAN_OSCILLATE_MASK, INCLUDE_VIOLATIONS_MASK)

    border = ndimage.maximum_filter(mask, size=3, mode='wrap') ^ mask
    interior = ndimage.minimum_filter(mask, size=3, mode='wrap')
    gen_mask = mask * (
        NEW_CELL_MASK |
        CAN_OSCILLATE_MASK |
        INCLUDE_VIOLATIONS_MASK
    ) + border * (
        INCLUDE_VIOLATIONS_MASK
    )
    board = np.zeros(mask.shape, dtype=np.uint16)
    foreground = np.zeros(mask.shape, dtype=bool)
    background = np.zeros(mask.shape, dtype=bool)
    background_color = np.zeros(mask.shape, dtype=bool)
    seeds = None
    max_period = 1

    for layer in layer_params:
        if not isinstance(layer, dict):
            raise ValueError(
                "'layer_params' should be a list of parameter dictionaries.")
        layer = _fix_random_values(layer)
        old_board = board.copy()
        gen_mask0 = gen_mask.copy()
        interior = ndimage.minimum_filter(
            gen_mask & NEW_CELL_MASK > 0, size=3, mode='wrap')
        color = COLORS.get(layer.get('color'), 0)

        fence_frac = layer.get('fences', 0.0)
        if fence_frac > 0:
            fences = build_fence(gen_mask & speedups.NEW_CELL_MASK)
            fences *= coinflip(fence_frac, fences.shape)
            gen_mask &= ~(fences * (NEW_CELL_MASK | CAN_OSCILLATE_MASK))
            board += fences.astype(np.uint16) * CellTypes.wall

        spawners = layer.get('spawners', 0)
        if spawners > 0:
            _mask = (gen_mask0 & NEW_CELL_MASK > 0) & interior
            new_cells = _mask & coinflip(spawners, board.shape)
            if not new_cells.any() and _mask.any():
                i, j = np.nonzero(_mask)
                k = get_rng().choice(len(i))  # ensure at least one spawner
                new_cells[i[k], j[k]] = True
            gen_mask[new_cells] ^= NEW_CELL_MASK
            board[new_cells] = CellTypes.spawner + color

        tree_lattice = layer.get('tree_lattice')
        # Create a lattice of trees that are spread throughout the region
        # such that every empty cell touches one (and only one) tree
        # (modulo edge effects).
        # Such a lattice tends to make the resulting board very chaotic.
        # Note that this will disrupt any pre-existing patterns.
        if tree_lattice is not None:
            if not isinstance(tree_lattice, dict):
                tree_lattice = {}
            h, w = board.shape
            stagger = tree_lattice.get('stagger', True)
            spacing = float(tree_lattice.get('spacing', 5))
            if not stagger:
                new_cells = _make_lattice(h, w, spacing, spacing, 0)
            elif spacing <= 3:
                new_cells = _make_lattice(h, w, 3, 3, 1)
            elif spacing == 4:
                new_cells = _make_lattice(h, w, 10, 1, 3)
            elif spacing == 5:
                new_cells = _make_lattice(h, w, 13, 1, 5)
            else:
                # The following gets pretty sparse.
                new_cells = _make_lattice(h, w, 6, 3, 3)

            new_cells &= gen_mask & NEW_CELL_MASK > 0
            board[new_cells] = CellTypes.tree + color

        period = 1
        if 'pattern' in layer:
            pattern_args = layer['pattern'].copy()
            period = pattern_args.get('period', 1)
            if period == 1:
                gen_mask2 = gen_mask & ~CAN_OSCILLATE_MASK
                pattern_args.update(period=max_period, osc_bonus=0)
            elif period == 0:
                gen_mask2 = gen_mask & ~INCLUDE_VIOLATIONS_MASK
                pattern_args.update(period=max_period, osc_bonus=0)
            elif period < max_period:
                raise ValueError(
                    "Periods for sequential layers in a region must be either 0, 1,"
                    " or at least as large as the largest period in prior layers.")
            else:
                gen_mask2 = gen_mask
                max_period = period

            board = _gen_pattern(board, gen_mask2, seeds, **pattern_args)

            # We need to update the mask for subsequent layers so that they
            # do not destroy the pattern in this layer.
            # First get a list of board states throughout the oscillation cycle.
            boards = [board]
            for _ in range(1, max_period):
                boards.append(speedups.advance_board(boards[-1]))
            non_empty = np.array(boards) != 0
            still_cells = non_empty.all(axis=0)
            osc_cells = still_cells ^ non_empty.any(axis=0)
            # Both still life cells and oscillating cells should disallow
            # any later changes. We also want to disallow changes to the cells
            # that are neighboring the oscillating cells, because any changes
            # there would propogate to the oscillating cells at later time
            # steps.
            # Note that it doesn't really matter whether the oscillating mask
            # is set for the currently oscillating cells, because we're not
            # checking for violations in them anyways, and we don't allow any
            # changes that would affect them.
            osc_neighbors = ndimage.maximum_filter(osc_cells, size=3, mode='wrap')
            gen_mask[osc_cells] &= ~(NEW_CELL_MASK | INCLUDE_VIOLATIONS_MASK)
            gen_mask[still_cells | osc_neighbors] &= ~(NEW_CELL_MASK | CAN_OSCILLATE_MASK)

            new_mask = board != old_board
            life_mask = ((board & CellTypes.alive) > 0) & new_mask
            board += color * new_mask * life_mask
            # The seeds are starting points for the next layer of patterns.
            # This just makes the patterns more likely to end up close together.
            seeds = ((board & CellTypes.alive) > 0) & mask

        new_mask = board != old_board

        movable_walls = layer.get('movable_walls', 0)
        if movable_walls > 0:
            new_cells = coinflip(movable_walls, board.shape) * new_mask
            new_cells *= (board & ~CellTypes.rainbow_color) == CellTypes.wall
            board += new_cells * CellTypes.movable

        movable_trees = layer.get('movable_trees', 0)
        if movable_trees > 0:
            new_cells = coinflip(movable_trees, board.shape) * new_mask
            new_cells *= (board & ~CellTypes.rainbow_color) == CellTypes.tree
            board += new_cells * CellTypes.movable

        hardened_life = layer.get('hardened_life', 0)
        if hardened_life > 0:
            new_cells = coinflip(hardened_life, board.shape) * new_mask
            new_cells *= (board & ~CellTypes.rainbow_color) == CellTypes.life
            board -= new_cells * CellTypes.destructible

        buffer_size = layer.get('buffer_zone', 0) * 2 + 1
        life_cells = board & CellTypes.alive > 0
        buf = ndimage.maximum_filter(life_cells, size=buffer_size, mode='wrap')
        gen_mask[buf] &= ~NEW_CELL_MASK

        target = layer.get('target', 'board')
        if target == 'board':
            foreground[new_mask] = True
            if period > 0:
                background[new_mask] = True
        elif target == 'goals':
            background[new_mask] = True
            background_color[new_mask] = True
            # Make sure to add walls and such to the foreground
            foreground[new_mask & (board & CellTypes.alive == 0)] = True
        elif target == 'both':
            foreground[new_mask] = True
            if period > 0:
                background[new_mask] = True
                background_color[new_mask] = True
        else:
            raise ValueError("Unexpected value for 'target': %s" % (target,))

        fountains = layer.get('fountains', 0)
        if fountains > 0:
            new_cells = coinflip(fountains, board.shape)
            new_cells *= gen_mask & NEW_CELL_MASK > 0
            neighbors = ndimage.maximum_filter(new_cells, size=3, mode='wrap')
            neighbors *= gen_mask & NEW_CELL_MASK > 0
            gen_mask[neighbors] = INCLUDE_VIOLATIONS_MASK
            if buffer_size > 1:
                buf = ndimage.maximum_filter(neighbors, size=buffer_size, mode='wrap')
                gen_mask[buf] &= ~NEW_CELL_MASK
            board[neighbors] = CellTypes.wall + color
            board[new_cells] = CellTypes.fountain + color
            foreground[new_cells] = True
            background[neighbors] = True
            background_color[neighbors] = True

    goals = board.copy()
    board *= foreground
    goals *= background
    goals &= ~CellTypes.spawning
    goals &= ~(CellTypes.rainbow_color * ~background_color)

    return board, goals


def gen_game(
        board_shape=(25,25), min_performance=-1, partitioning={},
        starting_region=None, later_regions=None, buffer_region=None,
        named_regions={}, **etc):
    """
    Randomly generate a new SafeLife game board.

    Generation proceeds by creating several different random "regions",
    and then filling in each region with one of several types of patterns
    or tasks. Regions can be surrounded by fences / walls to make it harder
    for patterns to spread from one region to another.

    Each set of parameters can additionally be randomized by passing in
    a dictionary either with the 'choices' key or the 'uniform' key.
    For example::

        gen_game(board_shape={'choices':[(25,25), (15,15)]})

    will create a new game with board size of either 15x15 or 25x15 with equal
    probability. Likewise, ::

        gen_game(min_performance={'uniform':[0,1]})

    will randomly pick the ``min_performance`` key to be between 0 and 1.
    This is especially useful for the individual region parameters such that
    regions with different characteristics can coexist on the same board.

    Parameters
    ----------
    board_shape : (int, int)
    min_performance : float
        The minimum proportion of the level that needs to be completed before
        the exit will open.
    partitioning : dict
        Set of parameters to pass to :func:`make_partioned_regions`.
        Individual values can also be randomized.
    start_region : str or None
        Fix the first region type to be of type `start_region`. If None, the
        start region is treated just like the later regions.
    later_regions : str
        Name of the region parameters to use for subsequent regions.
        This can be randomized (see above) to get different region types on the
        same game board.
    buffer_region : str or None
        Name of the region parameters to apply to the white buffer region.
        Can be None to keep the buffer region clear.
    named_regions : dict
        A dictionary of region types to region parameters. Each set of region
        parameters should consist of a list of layer parameters. See
        :func:`populate_region` for more details.

    Returns
    -------
        SafeLifeGame instance
    """
    board_shape = _fix_random_values(board_shape)
    min_performance = _fix_random_values(min_performance)
    partitioning = _fix_random_values(partitioning)

    regions = make_partioned_regions(board_shape, **partitioning)
    board = np.zeros(board_shape, dtype=np.uint16)
    goals = np.zeros(board_shape, dtype=np.uint16)

    # Create locations for the player and the exit
    zero_reg = regions == 0
    i, j = np.nonzero(zero_reg)
    k1 = get_rng().choice(len(i))
    i1, j1 = i[k1], j[k1]
    board[i1, j1] = CellTypes.player
    # Make the exit as far away from the player as possible
    row_dist = np.abs(np.arange(board_shape[0])[:, np.newaxis] - i1)
    col_dist = np.abs(np.arange(board_shape[1])[np.newaxis, :] - j1)
    row_dist = np.minimum(row_dist, board_shape[0] - row_dist)
    col_dist = np.minimum(col_dist, board_shape[1] - col_dist)
    dist = (row_dist + col_dist) * zero_reg
    k2 = np.argmax(dist)
    i2 = k2 // board_shape[1]
    j2 = k2 % board_shape[1]
    board[i2, j2] = CellTypes.level_exit | CellTypes.color_r

    # Ensure that the player and exit aren't touching any other region
    n = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    regions[(i1+n) % board.shape[0], (j1+n.T) % board.shape[1]] = -1
    regions[(i2+n) % board.shape[0], (j2+n.T) % board.shape[1]] = -1

    # and fill in the regions...
    for k in np.unique(regions)[2:]:
        mask = regions == k
        if starting_region is not None:
            region_name = _fix_random_values(starting_region)
        else:
            region_name = _fix_random_values(later_regions)
        if region_name not in named_regions:
            logger.error("No region parameters for name '%s'", region_name)
            continue
        logger.debug("Making region: %s", region_name)
        rboard, rgoals = populate_region(mask, named_regions[region_name])
        board += rboard
        goals += rgoals
        starting_region = None
    buffer_region = _fix_random_values(buffer_region)
    if buffer_region in named_regions:
        mask = regions == 0
        rboard, rgoals = populate_region(mask, named_regions[buffer_region])
        board += rboard
        goals += rgoals

    # Give the buffer (0) region a rainbow / white color
    # This is mostly a visual hint for humans
    buffer_mask = (regions <= 0) & (goals & CellTypes.rainbow_color == 0)
    goals[buffer_mask] += CellTypes.rainbow_color

    game = SafeLifeGame()
    game.deserialize({
        'board': board,
        'goals': goals,
        'agent_loc': (j[k1], i[k1]),
        'min_performance': min_performance,
        'orientation': 1,
    })
    return game


def stability_mask(board, period=6, remove_agent=True):
    """
    Highlights separable regions that stable with the given period.

    A "separable" region is one which can be removed from the board
    without effecting any of the rest of the board.

    Parameters
    ----------
    board : array
    period : int
        The stability period to check for. A period of 1 means that it's a
        still life only. A period of 0 allows for any pattern, stable or not.
    remove_agent : bool
        If True, the agent is removed from the board before checking for
        stability. This means that the agent's freezing power doesn't
        affect the stability.
    """
    if remove_agent:
        board = board * ((board & CellTypes.agent) == 0)

    neighborhood = np.ones((3,3))
    alive = (board & CellTypes.alive) // CellTypes.alive
    neighbors = ndimage.convolve(alive, neighborhood, mode='wrap')
    max_neighbors = neighbors
    ever_alive = alive
    orig_board = board
    for _ in range(period):
        board = speedups.advance_board(board)
        alive = (board & CellTypes.alive) // CellTypes.alive
        neighbors = ndimage.convolve(alive, neighborhood, mode='wrap')
        ever_alive |= alive
        max_neighbors = np.maximum(max_neighbors, neighbors)
    is_boundary = (board & CellTypes.frozen > 0)
    is_boundary |= (ever_alive == 0) & (max_neighbors <= 2)
    labels, num_labels = speedups.wrapped_label(~is_boundary)
    mask = np.zeros(board.shape, dtype=bool)
    for idx in range(1, num_labels+1):
        region = labels == idx
        if (board[region] == orig_board[region]).all():
            mask |= region
    return mask
