import numpy as np
from scipy import ndimage, signal

from .game_physics import CellTypes, SafeLife
from .helper_utils import coinflip
from . import speedups


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
        weights[0] = min(alpha, weights[0]) if len(weights) < max_regions else 1e-10
        if len(weights) <= min_regions:
            weights[1:] = 1e-10
        weights /= np.sum(weights)
        k = np.random.choice(len(perimeters), p=weights)
        plist = list(perimeters[k])
        i, j = plist[np.random.randint(len(plist))]
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


def region_population_params(difficulty=5, **fixed_params):
    """
    Dynamically set region population parameters based on difficulty.

    Any parameter that's left blank will be automatically picked via the
    difficulty. If a parameter is specified, then the difficulty is ignored
    for that parameter.

    All parameters which accept a range of values take as input the minimum
    and maximum values of that range. The output parameter will be randomly
    chosen within that range.

    Parameters
    ----------
    difficulty : float
    region_types : dict (str : float)
        A dictionary mapping the different possible of region types to their
        relative weights. For each region one of the region types will be
        randomly chosen based on their weights.
    cell_probabilities : dict (str : float)
        A dictionary mapping different cell types to their probability of
        being included in the region.
    cell_penalties : dict (str : (float, float))
        Penalties associated with each cell type ("alive", "wall", and "tree").
        Larger penalties make that cell type less likely to appear the pattern
        generation. The first number is the penalty when the relative frequency
        of the cell type is 0%; the second number is the penalty when the
        relative frequency is 100%. Intermediate penalties scale linearly.
    spawner_colors : dict (str : float)
        A dictionary mapping the different spawner colors (gray, red, green,
        blue) to their relative weights. For each region one of the spawner
        colors will be randomly chosen based on their weights.
    period_weights : dict (int : float)
        A dictionary that assigns relative weights to different potential
        pattern periods. For each region one of the patterns will be randomly
        chosen based on their weights.
    fence_frac : range (float, float)
        Fraction of fences that are kept during region generation.
    extra_walls_frac : range (float, float)
        Fraction of free space that is converted to walls.
    crate_frac : range (float, float)
        Fraction of the walls that are converted to (movable) crates.
    plant_frac : range (float, float)
        Fraction of the trees that are converted to (movable) plants.
    hardened_frac : range (float, float)
        Fraction of life cells that are made indestructible.
    temperature : range (float, float)
        Temperature for pattern generation algorithm. Higher temperatures
        tend to result in more complex patterns. Should generally be in the
        range of 0.1 to 2.0.
    min_fill : range (float, float)
        The minimum acceptable fill percentage during procedural generation.
        Obviously, larger values tend to produce more complex patterns.
        Should not be higher than about 0.4 or the procedural generation will
        have a hard time converging.

    Returns
    -------
    dict
    """
    def dscale(x, y):
        """
        Do linear interpolation based on difficulty.
        """
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        assert len(x) == len(y)
        k = np.searchsorted(x, difficulty, side='right')
        k1 = max(0, k-1)
        k2 = min(k, len(x) - 1)
        r = 1 if k1 == k2 else (difficulty - x[k1]) / (x[k2] - x[k1])
        y = np.array(y)
        return ((1-r) * y[k1] + r * y[k2]).tolist()

    params = {
        "region_types": {
            "neutral": dscale([1,1,10], [0,1,3]),
            "build": 1,
            "append": dscale([2,2,10], [0,1,2]),
            "destroy": dscale([3,3,10], [0,1,2]),
            "prune": dscale([4,4,10], [0,1,2]),
            "spawner": dscale([3,3,10], [0,2,4]),
            "fountain": dscale([6,6,10], [0,1.5,3]),
            "grow": dscale([7,7,10], [0,2,3]),
        },
        "cell_probabilities": {
            "wall": dscale([0, 2, 2, 10], [0, 0, 0.25, 0.5]),
            "tree": dscale([0, 3, 3, 10], [0, 0, 0.25, 0.5]),
        },
        "cell_penalties": {
            "wall": (1, 20),
            "tree": (1, 20),
        },
        "spawner_colors": {
            # "gray": 1,
            "green": 1,
            "red": dscale([5,6], [0,0.5]),
            "blue": dscale([6,7], [0,1]),
        },
        "spawner_trees": dscale([0,3,10], [0, 0, 0.1]),
        "period_weights": {
            # Note that period 3 oscillators can take a really long time
            # to generate.
            1: 1,
            2: dscale([4,10], [0,2]),
            # 3: dscale([7,10], [0,2]),
        }
    }
    random_params = {
        # The following should all look like (min_frac, max_frac)
        "fence_frac": dscale([0, 10], [[1.1, 2], [-0.2, 0.8]]),
        "extra_walls_frac": dscale([0, 10], [[0, 0], [0, 0.1]]),
        "crate_frac": dscale([0, 5, 10], [[0,0], [0.2, 1], [-0.1, 0.5]]),
        "plant_frac": dscale([0, 5, 10], [[0,0], [0.2, 1], [-0.1, 0.5]]),
        "hardened_frac": dscale([5, 10], [[-1,0], [-0.5, 1]]),
        "temperature": dscale([0, 5, 10], [[0.3, 0.3], [0.4, 0.8], [0.5, 0.8]]),
        "min_fill": dscale([0, 5, 10], [[0.05, 0.1], [0.1, 0.2], [0.15, 0.3]]),
    }

    params.update(random_params)
    params.update(fixed_params)

    for key in random_params:
        low, high = params[key]
        params[key] = low + np.random.random() * (high - low)

    return params


def _pick_one(choices):
    if not choices:
        raise ValueError("'choices' must be a non-empty dictionary")
    keys = list(choices.keys())
    vals = np.array(list(choices.values()))
    if (vals < 0).any() or np.sum(vals) <= 0:
        raise ValueError(
            "The values for different choices must be non-negative and their "
            "sum must be positive.")
    return np.random.choice(keys, p=vals / np.sum(vals))


def populate_region(mask, **params):
    """
    Populate the interior of a masked region, producing both cells and goals.

    Parameters
    ----------
    mask : array (dtype=bool, dim=2)
        An array with True values to mark the region of interest.
    **params
        See :func:`region_population_params` for extra parameters.

    Returns
    -------
    board : array
    goals : array
    """
    params = region_population_params(**params)
    region_type = _pick_one(params["region_types"])
    period = int(_pick_one(params["period_weights"]))
    if period > 1:
        params["cell_probabilities"] = {'wall': 1, 'tree': 1}

    border = ndimage.maximum_filter(mask, size=3, mode='wrap') ^ mask
    fences = build_fence(mask)
    if region_type != "spawner":
        # Poke holes in the fence
        fences *= coinflip(params['fence_frac'], mask.shape)
    board = fences.astype(np.int16) * CellTypes.wall
    gen_mask = (mask & ~fences) * (
        speedups.NEW_CELL_MASK |
        speedups.CAN_OSCILLATE_MASK |
        speedups.INCLUDE_VIOLATIONS_MASK
    ) + (fences | border) * (
        speedups.INCLUDE_VIOLATIONS_MASK
    )

    def _gen_pattern(
            board, mask, seeds=None, num_retries=5, half=False, exclude=()):
        # temperature < 0.3 tends to not converge, or converge very slowly
        # temperature = 0.4, fill = 0.05 yields pretty simple patterns
        # temperature = 1.5, fill = 0.4 yields pretty complex patterns
        temperature = params["temperature"]
        min_fill = 0.5 * params["min_fill"] if half else params["min_fill"]
        cell_penalties = params["cell_penalties"].copy()
        cell_probabilities = params["cell_probabilities"]
        for name, prob in cell_probabilities.items():
            if not coinflip(prob) or name in exclude:
                del cell_penalties[name]
        try:
            new_board = speedups.gen_pattern(
                board, mask, period, seeds=seeds, max_iter=100,
                min_fill=min_fill, temperature=temperature, **cell_penalties)
            working_area = mask & speedups.NEW_CELL_MASK
            new_cells = new_board != 0
            fill_ratio = np.sum(new_cells * working_area) / np.sum(working_area)
            if fill_ratio > 2 * min_fill:
                if num_retries > 0:
                    return _gen_pattern(
                        board, mask, seeds, num_retries-1, half, exclude)
                else:
                    print("gen_pattern produced an overfull pattern. "
                          "num_retries exceeded; no patterns added.")
                    return board
            return new_board
        except speedups.InsufficientAreaException:
            return board
        except speedups.MaxIterException:
            if num_retries > 0:
                return _gen_pattern(
                    board, mask, seeds, num_retries-1, half, exclude)
            else:
                print("gen_pattern did not converge! "
                      "num_retries exceeded; no patterns added.")
                return board
        except speedups.BoardGenException:
            return board

    # Two passes
    # create cells and/or goals of a particular color
    # second pass always excludes trees
    # If it gets added to the goals, also add non-life cells to the board
    first_color, first_dest = {
        'neutral': (CellTypes.color_g, 'board'),
        'build': (CellTypes.color_b, 'goal'),
        'destroy': (CellTypes.color_r, 'board'),
        'append': (CellTypes.color_g, 'board'),
        'prune': (CellTypes.color_g, 'board'),
        'grow': (CellTypes.color_g, 'board'),
    }.get(region_type, (None, None))
    second_color, second_dest = {
        'append': (CellTypes.color_b, 'goal'),
        'prune': (CellTypes.color_r, 'board'),
        'grow': (CellTypes.color_g, 'goal'),
    }.get(region_type, (None, None))

    goals = None

    if first_color is not None:
        board = _gen_pattern(board, gen_mask)
        life_mask = ((board & CellTypes.alive) > 0)
        frozen_mask = ((board & CellTypes.frozen) > 0)
        board += life_mask * first_color
        if first_dest == 'goal':
            goals = board.copy()
            board *= frozen_mask

    if second_color is not None:
        # Mask out everything that's not non-zero
        boards = [board]
        for _ in range(1, period):
            boards.append(speedups.advance_board(boards[-1]))
        all_zero = np.product(np.array(boards) == 0, axis=0).astype(bool)
        gen_mask ^= (gen_mask & speedups.NEW_CELL_MASK) * ~all_zero
        board = _gen_pattern(
            board, gen_mask, seeds=life_mask, half=True, exclude=('tree',))
        life_mask = ((board & CellTypes.alive) > 0) * all_zero
        if second_dest == 'goal':
            # copy board to goals, but with old colors removed
            goals = board.copy()
            goals &= ~CellTypes.rainbow_color
            goals += life_mask * second_color
            board *= ~life_mask
        else:
            board += life_mask * second_color

    if region_type == "spawner":
        spawner_color = {
            "gray": 0,
            "red": CellTypes.color_r,
            "green": CellTypes.color_g,
            "blue": CellTypes.color_b,
        }[_pick_one(params['spawner_colors'])]
        interior_mask = ndimage.minimum_filter(mask, size=3, mode='wrap')
        i, j = np.nonzero(interior_mask & (board == 0))
        if len(i) > 0:
            k = np.random.randint(len(i))
            board[i[k], j[k]] = CellTypes.spawner + spawner_color
        tree_mask = interior_mask & (board == 0)
        tree_mask &= coinflip(params['spawner_trees'], board.shape)
        board += tree_mask * (CellTypes.tree + spawner_color)
        life_frac = 0.3
        life_mask = interior_mask & (board == 0)
        life_mask &= coinflip(life_frac, board.shape)
        board += life_mask * (CellTypes.life + spawner_color)

    if region_type == "fountain":
        fountain_mask = mask & (board == 0) & coinflip(0.04, board.shape)
        fountain_neighbor = ndimage.maximum_filter(fountain_mask, size=3, mode='wrap')
        fountain_color = np.random.choice([
            CellTypes.color_r, CellTypes.color_g, CellTypes.color_b])
        fountain_color |= CellTypes.frozen  # Make sure that the goal cells don't evolve
        board += fountain_mask * (CellTypes.fountain | fountain_color)
        # Give the goal
        goals = (mask & fountain_neighbor & ~fountain_mask) * fountain_color

    if goals is None:
        goals = np.zeros_like(board)

    # Make some of the life types hardened
    life_mask = (board & ~CellTypes.rainbow_color == CellTypes.life)
    hardlife_mask = coinflip(params["hardened_frac"], board.shape)
    board -= life_mask * hardlife_mask * CellTypes.destructible

    # Remove fences and add extra walls in the middle
    # wall_mask = mask & (board == 0) & (goals == 0)
    # wall_mask &= coinflip(params["extra_walls_frac"], board.shape)
    # board += wall_mask * CellTypes.wall

    # Turn some walls and trees into crates and plants
    crate_mask = (board == CellTypes.wall)
    crate_mask &= coinflip(params["crate_frac"], board.shape)
    board += crate_mask * CellTypes.movable
    plant_mask = (board == CellTypes.tree)
    plant_mask &= coinflip(params["plant_frac"], board.shape)
    board += plant_mask * CellTypes.movable

    return board, goals


def gen_game(
        board_shape=(25,25), max_regions=5, start_region='build',
        min_completion=-1, **region_params):
    """
    Randomly generate a new SafeLife game board.

    Generation proceeds by creating several different random "regions",
    and then filling in each region with one of several types of patterns
    or tasks. Regions can be surrounded by fences / walls to make it harder
    for patterns to spread from one region to another.

    Parameters
    ----------
    board_shape : (int, int)
    max_regions : int
    start_region : str or None
        Fix the first region type to be of type `start_region`. If None, the
        first region type is randomly chosen, just like all the others.
    region_params : dict
        Extra parameters to be passed to :func:`populate_region`.
        See also :func:`region_population_params`.
    min_completion : float
        The minimum proportion of the level that needs to be completed before
        the exit will open.

    Returns
    -------
        SafeLife instance
    """
    regions = make_partioned_regions(board_shape, max_regions=max_regions)
    board = np.zeros(board_shape, dtype=np.int16)

    # Create locations for the player and the exit
    i, j = np.nonzero(regions == 0)
    k1, k2 = np.random.choice(len(i), size=2, replace=False)
    board[i[k1], j[k1]] = CellTypes.player
    board[i[k2], j[k2]] = CellTypes.level_exit | CellTypes.color_r

    # Ensure that the player isn't touching any other region
    n = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    regions[(i[k1]+n) % board.shape[0], (j[k1]+n.T) % board.shape[1]] = 0

    # Give the boarder (0) regions a rainbow / white color
    # This is mostly a visual hint for humans
    goals = (regions == 0).astype(np.int16) * CellTypes.rainbow_color

    # and fill in the regions...
    for k in np.unique(regions)[1:]:
        mask = regions == k
        if start_region:
            params = region_params.copy()
            params["region_types"] = {start_region: 1}
        else:
            params = region_params
        rboard, rgoals = populate_region(mask, **params)
        board += rboard
        goals += rgoals
        start_region = None

    game = SafeLife()
    game.deserialize({
        'board': board,
        'goals': goals,
        'agent_loc': (j[k1], i[k1]),
        'min_completion': min_completion,
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
