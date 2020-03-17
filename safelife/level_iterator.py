import os
import glob
import queue
import warnings
from multiprocessing.pool import Pool, ApplyResult

import yaml
import numpy as np

from .safelife_game import SafeLifeGame
from .proc_gen import gen_game
from .random import set_rng


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


def _game_from_data(file_name, data_type, data, seed=None):
    if data_type == "procgen":
        with set_rng(np.random.default_rng(seed)):
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


class SafeLifeLevelIterator(object):
    """
    Iterator to load SafeLifeGame instances from the specified paths.

    Note that the paths can either point to json files (for procedurally
    generated levels) or to npz files (specific files saved to disk).

    Parameters
    ----------
    paths : list of strings
        The paths to the files to load. Files should either ".npz" archives
        of saved SafeLife levels, or ".yaml" files of procedural generation
        parameters. Examples of each can be found in the "safelife/levels"
        folder.

        Note that the path names can use glob expressions or can point to a
        directory of files to load. Files will first be searched for in the
        current working directory. If not found, the "levels" directory will be
        searched as well. If no paths are supplied, this will generate a
        random level using default level generation parameters.
    repeat_levels : bool
        If true, levels are repeated (or procedurally generated) in an endless
        loop. Otherwise, each level that was specified in the input is
        generated only once before iteration stops. The default is True if any
        of the input files are procedural generation files, and False otherwise.
    distinct_levels : int or None
        Number of distinct levels to produce. If levels are procedurally
        generated, then putting a cap on the number of distinct levels will
        cause the same procedurally generated file to be yielded multiple
        times. If None, there is no cap on distinct levels.
    num_workers : int
        Number of workers used to generate new instances. If this is nonzero,
        then new instances will be generated asynchronously using the
        multiprocessing module. This can significantly reduce the wait time
        needed to retrieve new levels, as there will tend to be a ready queue.
    max_queue : int
        Maximum number of levels to queue up at once. This should be at least
        as large as the number of workers. Not applicable for zero workers.
    seed : int or numpy.random.SeedSequence or None
        Seed for the random number generator(s). The same seed ought to produce
        the same set of sequence of SafeLife levels across different trials.
    """
    def __init__(
            self, *paths, repeat_levels=None, distinct_levels=None,
            num_workers=1, max_queue=10, seed=None
    ):
        self.file_data = _load_files(paths)
        self.level_cache = []

        if repeat_levels is None:
            for data in self.file_data:
                repeat_levels = repeat_levels or data[1] == "procgen"

        self.repeat_levels = repeat_levels
        self.distinct_levels = distinct_levels

        self.num_workers = num_workers
        self.max_queue = max_queue if num_workers > 0 else 1
        self.results = None
        self.pool = None
        self.idx = 0

        self.seed(seed)

    def seed(self, seed):
        if not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(seed)
        self._seed = seed

    def get_next_parameters(self):
        """
        Return the parameters used to create the next level.

        This can be modified to return different parameters that, for example,
        dynamically change with time or some other metric.
        """
        return self.file_data[self.idx % len(self.file_data)]

    def fill_queue(self):
        if self.results is None:
            self.results = queue.deque(maxlen=self.max_queue)
        if self.num_workers > 0:
            if self.pool is None:
                self.pool = Pool(processes=self.num_workers)

        while len(self.results) < self.max_queue:
            if self.distinct_levels is not None and self.idx >= self.distinct_levels:
                break
            elif not self.repeat_levels and self.idx >= len(self.file_data):
                break
            else:
                data = self.get_next_parameters()
                if data is None:
                    break
            self.idx += 1
            kwargs = {'seed': self._seed.spawn(1)[0]}
            if self.num_workers > 0:
                result = self.pool.apply_async(_game_from_data, data, kwargs)
            else:
                result = _game_from_data(*data, **kwargs)
            self.results.append((data, result))

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.num_workers > 0:
            # Don't pickle the multiprocessing pool, and wait on all queued results.
            state['pool'] = None
            state['results'] = queue.deque([
                r.get() if isinstance(r, ApplyResult) else r
                for r in self.results
            ], maxlen=self.max_queue)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __iter__(self):
        return self

    def __next__(self):
        self.fill_queue()

        if not self.results and self.distinct_levels is not None:
            if not self.repeat_levels and self.idx >= self.distinct_levels:
                raise StopIteration
            else:
                # Repeat levels that we've already seen.
                # Should only get here if we've maxed out distinct levels.
                data = self.level_cache[self.idx % self.distinct_levels]
                result = _game_from_data(*data)
                self.idx += 1
        elif not self.results:
            raise StopIteration
        else:
            data, result = self.results.popleft()
        if isinstance(result, ApplyResult):
            result = result.get()
        if (self.distinct_levels is not None
                and len(self.level_cache) < self.distinct_levels):
            if data[1] == "procgen":
                data = (data[0], "static", result.serialize())
            self.level_cache.append(data)
        return result


def safelife_loader(*paths, **kwargs):
    """
    Alias for SafeLifeLevelIterator. Deprecated.

    Note that the "shuffle" parameter is no longer used. Level parameters will
    be shuffled if repeat is True, and otherwise they will be used in order.
    """
    warnings.warn(
        "`safelife_loader` is deprecated. Use `SafeLifeLevelIterator` instead.",
        DeprecationWarning, stacklevel=2)
    kwargs.pop('shuffle', None)
    return SafeLifeLevelIterator(*paths, **kwargs)


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
    game_gen = SafeLifeLevelIterator(
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
