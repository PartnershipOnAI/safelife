import os
import glob


LEVEL_DIRECTORY = os.path.abspath(os.path.join(__file__, '../../levels'))


def find_files(*paths, ext='npz', use_glob=True):
    """
    Find all files that match the given paths.

    If the files cannot be found relative to the current working directory,
    this searches for them in the 'levels' folder as well.
    """
    for path in paths:
        try:
            yield from _find_files(path, ext, use_glob, use_level_dir=False)
        except FileNotFoundError:
            yield from _find_files(path, ext, use_glob, use_level_dir=True)


def _find_files(path, ext, use_glob, use_level_dir=False):
    path_0 = path
    if use_level_dir:
        path = os.path.join(LEVEL_DIRECTORY, path)
    else:
        path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if os.path.isdir(path):
        use_glob = True
        path = os.path.join(path, '*.' + ext if ext else '*')
    if use_glob:
        paths = sorted(glob.glob(path, recursive=True))
        if not paths:
            raise FileNotFoundError("No files found for '%s'" % path_0)
        yield from paths
    else:
        if not os.path.exists(path):
            raise FileNotFoundError("No files found for '%s'" % path_0)
        yield path
