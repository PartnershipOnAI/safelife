import os
import glob
import random
import queue
from multiprocessing import Pool
import yaml
import numpy as np

from .game_physics import SafeLifeGame
from .proc_gen import gen_game


LEVEL_DIRECTORY = os.path.abspath(os.path.join(__file__, '../levels'))
_default_params = yaml.load(
    open(os.path.join(LEVEL_DIRECTORY, 'random/_defaults.yaml')))


def find_files(*paths, file_types=None, use_glob=True):
    """
    Find all files that match the given paths.

    If the files cannot be found relative to the current working directory,
    this searches for them in the 'levels' folder as well.
    """
    for path in paths:
        try:
            yield from _find_files(path, file_types, use_glob, use_level_dir=False)
        except FileNotFoundError:
            yield from _find_files(path, file_types, use_glob, use_level_dir=True)


def _find_files(path, file_types, use_glob, use_level_dir=False):
    path_0 = path
    if use_level_dir:
        path = os.path.join(LEVEL_DIRECTORY, path)
    else:
        path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if os.path.isdir(path) and file_types:
        use_glob = True
        path = os.path.join(path, '*')
    elif use_level_dir and file_types and '.' not in os.path.split(path)[1]:
        # Don't need to include the file extension
        use_glob = True
        path += '.*'
    if use_glob:
        paths = sorted(glob.glob(path, recursive=True))
        if not paths:
            raise FileNotFoundError("No files found for '%s'" % path_0)
        if file_types:
            paths = filter(lambda p: p.split('.')[-1] in file_types, paths)
        yield from paths
    else:
        if not os.path.exists(path):
            raise FileNotFoundError("No files found for '%s'" % path_0)
        yield path


def _load_data(paths, repeat, shuffle):
    """
    Generate data to be fed into `_game_from_data`.
    """
    if paths:
        all_data = [[f] for f in find_files(
            *paths, file_types=('json', 'npz', 'yaml'))]
    else:
        all_data = [[None, 'procgen', {}]]

    while True:
        if shuffle:
            random.shuffle(all_data)
        for data in all_data:
            if len(data) == 1:
                file_name = data[0]
                if file_name.endswith('.json') or file_name.endswith('yaml'):
                    data += ['procgen', yaml.load(open(file_name))]
                else:
                    file_data = np.load(file_name)
                    # npz files aren't pickleable, which will mess up
                    # multiprocessing. Convert to a dict first.
                    file_data = {k: file_data[k] for k in file_data.keys()}
                    data += ['static', file_data]
            yield data

        if len(all_data) == 0 or not repeat or repeat == "auto" and not (
                len(all_data) == 1 and all_data[0][1] == "procgen"):
            break


def _game_from_data(file_name, data_type, data):
    if data_type == "procgen":
        named_regions = _default_params['named_regions'].copy()
        named_regions.update(data.get('named_regions', {}))
        data2 = _default_params.copy()
        data2.update(**data)
        data2['named_regions'] = named_regions
        game = gen_game(**data2)
    else:
        game = SafeLifeGame.loaddata(data)
    game.file_name = file_name
    return game


def safelife_loader(
        *paths, repeat="auto", shuffle=False, num_workers=1, max_queue=10):
    """
    Generator function to Load SafeLifeGame instances from the specified paths.

    Note that the paths can either point to json files (for procedurally
    generated levels) or to npz files (specific files saved to disk).

    Parameters
    ----------
    paths : list of strings
        The paths to the files to load. Note that this can use glob
        expressions, or it can point to a directory of files to load.
        Files will first be searched for in the current working directory.
        If not found, the 'levels' directory will be searched as well.
        If no paths are supplied, this will generate a random level using
        default level generation parameters.
    repeat : "auto" or bool
        If true, files will be loaded (yielded) repeatedly and forever.
        If "auto", it repeats if and only if 'paths' points to a single
        file of procedural generation parameters.
    shuffle : bool
        If true, the order of the files will be shuffled.
    num_workers : int
        Number of workers used to generate new instances. If this is nonzero,
        then new instances will be generated asynchronously using the
        multiprocessing module. This can significantly reduce the wait time
        needed to retrieve new levels, as there will tend to be a ready queue.
    max_queue : int
        Maximum number of levels to queue up at once. This should be at least
        as large as the number of workers.

    Returns
    -------
    SafeLifeGame generator
        Note that if repeat is true, infinite items will be returned.
        Only iterate over as many instances as you need!
    """
    if num_workers < 1 or max_queue < 1:
        for data in _load_data(paths, repeat, shuffle):
            yield _game_from_data(*data)
    else:
        pool = Pool(processes=num_workers)
        game_queue = queue.deque()
        for data in _load_data(paths, repeat, shuffle):
            if (len(game_queue) >= max_queue or
                    len(game_queue) > 0 and game_queue[0].ready()):
                next_game = game_queue.popleft().get()
            else:
                next_game = None
            game_queue.append(pool.apply_async(_game_from_data, data))
            if next_game is not None:
                yield next_game
        for result in game_queue:
            yield result.get()
