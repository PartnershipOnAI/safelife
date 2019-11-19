import os
import glob
import random
import queue
import itertools
from multiprocessing import Pool

import yaml
import numpy as np

from .safelife_game import SafeLifeGame
from .proc_gen import gen_game


LEVEL_DIRECTORY = os.path.join(os.path.dirname(__file__), 'levels')
LEVEL_DIRECTORY = os.path.abspath(LEVEL_DIRECTORY)
_default_params = yaml.safe_load(
    open(os.path.join(LEVEL_DIRECTORY, 'random', '_defaults.yaml')))


def find_files(*paths, file_types=(), use_glob=True):
    """
    Find all files that match the given paths.

    If the files cannot be found relative to the current working directory,
    this searches for them in the 'levels' folder as well.
    """
    for path in paths:
        path = os.path.normpath(path)
        try:
            yield from _find_files(path, file_types, use_glob, use_level_dir=False)
        except FileNotFoundError:
            yield from _find_files(path, file_types, use_glob, use_level_dir=True)


def _find_files(path, file_types, use_glob, use_level_dir=False):
    orig_path = path
    if use_level_dir:
        path = os.path.join(LEVEL_DIRECTORY, path)
    else:
        path = os.path.expanduser(path)
    path = os.path.abspath(path)

    def file_filter(path):
        return os.path.exists(path) and not os.path.isdir(path) and (
            path.split('.')[-1] in file_types if file_types is not None else True)

    # First look for any file that directly matches the path.
    paths1 = glob.glob(path, recursive=True) if use_glob else [path]
    files = list(filter(file_filter, paths1))
    if files:
        yield from sorted(files)
        return

    # Next try adding an extension
    paths2 = []
    for ext in file_types:
        path2 = path + '.' + ext
        paths2 += glob.glob(path2, recursive=True) if use_glob else [path2]
    files = list(filter(file_filter, paths2))
    if files:
        yield from sorted(files)
        return

    # Finally, try loading folders
    folders = filter(os.path.isdir, paths1)
    files = []
    for folder in folders:
        contents = [os.path.join(folder, file) for file in os.listdir(folder)]
        files += list(filter(file_filter, contents))
    if files:
        yield from sorted(files)
        return

    raise FileNotFoundError("No files found for '%s'" % orig_path)


def _load_files(paths):
    if not paths:
        return [[None, 'procgen', {}]]
    all_data = []
    for file_name in find_files(*paths, file_types=('json', 'npz', 'yaml')):
        if file_name.endswith('.json') or file_name.endswith('.yaml'):
            with open(file_name) as file_data:
                all_data.append([file_name, 'procgen', yaml.safe_load(file_data)])
        else:  # npz
            with np.load(file_name) as data:
                if 'levels' in data:
                    # Multiple levels in one archive
                    for idx, level in enumerate(data['levels']):
                        fname = os.path.join(file_name[:-4], level['name'])
                        all_data.append([fname, 'static', level])
                else:
                    # npz files aren't pickleable, which will mess up
                    # multiprocessing. Convert to a dict first.
                    data = {k: data[k] for k in data.keys()}
                    all_data.append([file_name, 'static', data])
    return all_data


def _load_data(paths, repeat, shuffle):
    """
    Generate data to be fed into `_game_from_data`.
    """
    file_data = _load_files(paths)
    if len(file_data) == 0:
        return
    if repeat == "auto":
        repeat = len(file_data) == 1 and file_data[0][1] == "procgen"
    if isinstance(repeat, bool):
        loop = itertools.count() if repeat else range(1)
    else:
        loop = range(repeat)

    for idx in loop:
        if shuffle:
            random.shuffle(file_data)
        for data in file_data:
            yield data


def _game_from_data(file_name, data_type, data, set_seed=False):
    if set_seed:
        from . import speedups
        seed = 0
        for i, s in enumerate(os.urandom(4)):
            seed += s << 8*i
        np.random.seed(seed)
        speedups.seed(seed)
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
    repeat : "auto" or bool or int
        If true, files will be loaded (yielded) repeatedly and forever.
        If "auto", it repeats if and only if 'paths' points to a single
        file of procedural generation parameters.
        If an int, it repeats exactly that many times.
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
        # Need to reseed the random number in each request if we're using more
        # than one worker.
        kwargs = {'set_seed': num_workers > 1}
        game_queue = queue.deque()
        for data in _load_data(paths, repeat, shuffle):
            if (len(game_queue) >= max_queue or
                    len(game_queue) > 0 and game_queue[0].ready()):
                next_game = game_queue.popleft().get()
            else:
                next_game = None
            game_queue.append(pool.apply_async(_game_from_data, data, kwargs))
            if next_game is not None:
                yield next_game
        for result in game_queue:
            yield result.get()


# ----------------------------

# The following functions are utilities for creating and combining levels
# into large archives. They're used to create the saved benchmark levels,
# but once those are created they generally won't need to be called again.


def gen_many(param_file, out_dir, num_gen, num_workers=8, max_queue=100):
    """
    Generate and save many levels using the above loader.
    """
    out_dir = os.path.abspath(out_dir)
    base_name = os.path.basepath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    num_digits = int(np.log10(num_gen))+1
    fmt = "{}-{{:0{}d}}.npz".format(base_name, num_digits)
    fmt = os.path.join(out_dir, fmt)
    game_gen = safelife_loader(
        param_file, num_workers=num_workers, max_queue=max_queue)
    for k in range(1, num_gen+1):
        fname = fmt.format(k)
        if os.path.exists(fname):
            continue
        next(game_gen).save(fname)


def combine_levels(directory):
    """
    Merge all files in a single directory.
    """
    files = sorted(glob.glob(os.path.join(directory, '*.npz')))
    all_data = []
    max_name_len = 0
    for file in files:
        with np.load(file) as data:
            name = os.path.split(file)[1]
            max_name_len = max(max_name_len, len(name))
            all_data.append(data.items() + [('name', name)])
    dtype = []
    for key, val in all_data[0][:-1]:
        dtype.append((key, val.dtype, val.shape))
    dtype.append(('name', str, max_name_len))
    combo_data = np.array([
        tuple([val for key, val in data]) for data in all_data
    ], dtype=dtype)
    np.savez_compressed(directory + '.npz', levels=combo_data)


def expand_levels(filename):
    """
    Opposite of combine_levels. Handy if we want to edit a single level.
    """
    with np.load(filename) as data:
        directory = filename[:-4]  # assume .npz
        os.makedirs(directory, exist_ok=True)
        for level in data['levels']:
            level_data = {k: level[k] for k in level.dtype.fields}
            np.savez_compressed(
                os.path.join(directory, level['name']), **level_data)


def gen_benchmarks():
    """
    Generate the benchmark levels! Should only be run once.
    """
    names = (
        'append-still append-dynamic append-spawn '
        'prune-dynamic prune-spawn prune-still prune-still-hard navigation'
    )
    for name in names.split():
        directory = os.path.join(LEVEL_DIRECTORY, 'benchmarks', 'v1.0', name)
        gen_many(os.path.join('random', name), directory, 100)
        with open(os.path.join(directory, ".gitignore"), 'w') as f:
            f.write('*\n')
        combine_levels(directory)
