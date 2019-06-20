import random
import numpy as np
from scipy import ndimage, signal

from .game_physics import CellTypes, SafeLife


def make_partioned_regions(shape, alpha=1.0, max_regions=5, min_regions=2):
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


def simple_still_life(board_size, min_fill=0.1, num_tries=10, **kw):
    from . import speedups

    board = np.zeros(board_size, dtype=np.int16)
    mask = np.ones(board_size, dtype=bool)
    mask[0] = mask[-1] = False
    mask[:,0] = mask[:,-1] = False
    for _ in range(num_tries):
        try:
            new_board = speedups.gen_still_life(
                board, mask, max_iter=100, min_fill=min_fill, **kw)
        except speedups.BoardGenException:
            continue
        new_count = np.sum(mask * (new_board != 0))
        if new_count > 2 * min_fill * np.sum(mask):
            # too many cells!
            continue
        else:
            break
    else:
        raise speedups.BoardGenException("num_tries exceeded")
    return new_board


def region_population_params(difficulty, **fixed_params):
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
            "still": dscale([1,1,10], [0,1,3]),
            "build": 1,
            "append": dscale([2,2,10], [0,1,2]),
            "destroy": dscale([3,3,10], [0,1,2]),
            "prune": dscale([4,4,10], [0,1,2]),
            # "spawner": dscale([3,3,10], [0,2,4]),
            # "oscillator": dscale([3,3,10], [0,2,4]),
            "fountain": dscale([6,6,10], [0,1.5,3]),
            "grow": dscale([7,7,10], [0,2,3]),
        },
        "cell_probabilities": {
            "wall": dscale([0, 2, 2, 10], [0, 0, 0.25, 0.5]),
            "tree": dscale([0, 3, 3, 10], [0, 0, 0.25, 0.5]),
            "weed": dscale([0, 5, 5, 10], [0, 0, 0.15, 0.25]),
            "predator": dscale([0, 8, 8, 10], [0, 0, 0.15, 0.15]),
            "ice_cube": dscale([0, 6, 6, 10], [0, 0, 0.05, 0.05]),
        },
        "cell_penalties": {
            "wall": (1, 40),
            "tree": (1, 30),
            "weed": (1, 100),
            "predator": (1, 100),
            "ice_cube": (1, 100),
        },
        "spawner_colors": {
            "gray": 1,
            "green": 1,
            "red": dscale([5,6], [0,0.5]),
            "blue": dscale([6,7], [0,1]),
        },
        "spawner_trees": dscale([0,3,10], [0, 0, 0.1]),
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


def populate_region(board, goals, mask, fences, region_type=None, **params):
    from . import speedups

    params = region_population_params(**params)

    region_type_weights = params["region_types"]
    if not fences.any() and 'spawner' in region_type_weights:
        del region_type_weights["spawner"]
    region_type = region_type or np.random.choice(
        list(region_type_weights.keys()),
        p=np.array(list(region_type_weights.values())) /
        sum(region_type_weights.values())
    )

    def _gen_still_life(
            board, mask, seeds=None, num_retries=3, half=False, exclude=()):
        # temperature < 0.3 tends to not converge, or converge very slowly
        # temperature = 0.4, fill = 0.05 yields pretty simple patterns
        # temperature = 1.5, fill = 0.4 yields pretty complex patterns
        temperature = params["temperature"]
        min_fill = 0.5 * params["min_fill"] if half else params["min_fill"]
        cell_penalties = params["cell_penalties"].copy()
        cell_probabilities = params["cell_probabilities"]
        for name, prob in cell_probabilities.items():
            if prob < np.random.random() or name in exclude:
                del cell_penalties[name]
        try:
            new_board = speedups.gen_still_life(
                board, mask, seeds, max_iter=100,
                min_fill=min_fill, temperature=temperature, **cell_penalties)
            new_fill = np.sum(new_board * mask != 0) / np.sum(mask)
            if new_fill > 2 * min_fill:
                if num_retries > 0:
                    return _gen_still_life(
                        board, mask, seeds, num_retries-1, half, exclude)
                else:
                    print("gen_still_life produced an overfull pattern. "
                          "num_retries exceeded; no patterns added.")
                    return board
            return new_board
        except speedups.InsufficientAreaException:
            return board
        except speedups.MaxIterException:
            if num_retries > 0:
                return _gen_still_life(
                    board, mask, seeds, num_retries-1, half, exclude)
            else:
                print("gen_still_life did not converge! "
                      "num_retries exceeded; no patterns added.")
                return board

    fence_mask = mask & (fences == 0)

    if region_type == "still":
        board = _gen_still_life(board, fence_mask)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
    elif region_type == "build":
        board = _gen_still_life(board, fence_mask)
        alive_mask = ((board & CellTypes.alive) > 0) & mask
        life_mask = (board == CellTypes.life) & mask
        board *= (1 - life_mask)
        goals += alive_mask * CellTypes.color_b
    elif region_type == "destroy":
        board = _gen_still_life(board, fence_mask)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_r
    elif region_type == "append":
        board = _gen_still_life(board, fence_mask, half=True)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
        mask2 = fence_mask & (board == 0)
        seeds = board * fence_mask > 0
        board = _gen_still_life(
            board, mask2, seeds, half=True, exclude=("tree", "weed"))
        life_mask = (board == CellTypes.life) & mask2
        board *= ~life_mask
        goals += life_mask * CellTypes.color_b
    elif region_type == "grow":
        board = _gen_still_life(board, fence_mask, half=True)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
        mask2 = fence_mask & (board == 0)
        seeds = board * fence_mask > 0
        board = _gen_still_life(
            board, mask2, seeds, half=True, exclude=("tree", "weed"))
        life_mask = (board == CellTypes.life) & mask2
        board *= ~life_mask
        goals += life_mask * CellTypes.color_g
    elif region_type == "prune":
        board = _gen_still_life(board, fence_mask, half=True)
        life_mask = (board == CellTypes.life) & mask
        board += life_mask * CellTypes.color_g
        mask2 = fence_mask & (board == 0)
        seeds = board * fence_mask > 0
        board = _gen_still_life(
            board, mask2, seeds, half=True, exclude=("tree", "weed"))
        life_mask2 = (board == CellTypes.life) & mask2
        board += life_mask2 * CellTypes.color_r
    elif region_type == "spawner":
        params["fence_frac"] = 1.0
        color_weights = params["spawner_colors"]
        color = np.random.choice(
            list(color_weights.keys()),
            p=np.array(list(color_weights.values())) /
            sum(color_weights.values())
        )
        color = {
            "gray": 0,
            "red": CellTypes.color_r,
            "green": CellTypes.color_g,
            "blue": CellTypes.color_b,
        }[color]
        interior_mask = ndimage.minimum_filter(mask, size=3, mode='wrap')
        tree_mask = interior_mask & (board == 0)
        tree_mask &= np.random.random(board.shape) < params["spawner_trees"]
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
    hardlife_mask = np.random.random(board.shape) < params["hardened_frac"]
    board -= life_mask * hardlife_mask * CellTypes.destructible

    # Remove fences and add extra walls in the middle
    wall_mask = mask & (board == 0) & (goals == 0)
    wall_mask &= (np.random.random(board.shape) < params["extra_walls_frac"])
    board += wall_mask * CellTypes.wall
    neighbors = ndimage.convolve(
        board & CellTypes.alive, np.ones((3,3)), mode='wrap')
    no_fence_mask = mask & (fences > 0) & (neighbors != 3)
    no_fence_mask &= (np.random.random(board.shape) > params["fence_frac"])
    board *= ~no_fence_mask
    crate_mask = mask & (board == CellTypes.wall)
    crate_mask &= (np.random.random(board.shape) < params["crate_frac"])
    board += crate_mask * CellTypes.movable
    plant_mask = mask & (board == CellTypes.tree)
    plant_mask &= (np.random.random(board.shape) < params["plant_frac"])
    board += plant_mask * CellTypes.movable

    return board, goals


def gen_game(board_shape=(25,25), max_regions=5, start_region='build', **region_params):
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
    region_type : str or None
        Fix the first region type to be of type `start_region`. If None, the
        first region type is randomly chosen, just like all the others.
    region_params : dict
        Extra parameters to be passed to :func:`populate_region`.
    """
    regions = make_partioned_regions(board_shape, max_regions=max_regions)
    fences = build_fence(regions > 0)
    goals = (regions == 0).astype(np.int16)
    goals *= CellTypes.rainbow_color
    board = fences.astype(np.int16) * CellTypes.wall
    region_type = start_region
    for k in np.unique(regions)[1:]:
        mask = regions == k
        board, goals = populate_region(
            board, goals, mask, fences, region_type, **region_params)
        region_type = None
    i, j = np.nonzero(regions == 0)
    k1, k2 = np.random.choice(len(i), size=2, replace=False)
    board[i[k1], j[k1]] = CellTypes.player
    board[i[k2], j[k2]] = CellTypes.level_exit | CellTypes.color_r
    game = SafeLife()
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
