import os
import glob
import random
import json
import numpy as np

from .game_physics import SafeLifeGame
from .proc_gen import gen_game


LEVEL_DIRECTORY = os.path.abspath(os.path.join(__file__, '../levels'))


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


def safelife_loader(*paths, repeat="auto", shuffle=False, callback=None):
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
        If true, the order of the files will be shuffled (not needed?).
    callback : function
        Optional callback that can be used to update board generation
        parameters before a level is procedurally generated. Should accept
        an integer logging how many games have been generated and a dictionary
        of parameters which it can (optionally) update in place.

    Returns
    -------
    SafeLifeGame generator
        Note that if repeat is true, infinite items will be returned.
        Only iterate over as many instances as you need!
    """
    game_num = 0
    if paths:
        all_data = [[f] for f in find_files(*paths, file_types=('json', 'npz'))]
    else:
        all_data = [[None, 'procgen', {}]]
    while True:
        if shuffle:
            random.shuffle(all_data)
        for data in all_data:
            game_num += 1
            if len(data) == 1:
                file_name = data[0]
                if file_name.endswith('.json'):
                    data += ['procgen', json.load(open(file_name))]
                else:
                    data += ['static', np.load(file_name)]
            file_name, datatype, data = data
            if datatype == "procgen":
                data = data.copy()  # maybe should be a deep copy?
                if callback is not None:
                    callback(game_num, data)
                game = gen_game(**data)
            else:
                game = SafeLifeGame.loaddata(data)
            game.file_name = file_name
            yield game
        if len(all_data) == 0 or not repeat or repeat == "auto" and not (
                len(all_data) == 1 and all_data[0][1] == "procgen"):
            break
