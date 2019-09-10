from __future__ import print_function
try:
    import sys
    if sys.version_info[0] < 3:
        print("SafeLife only runs in Python 3+")
        sys.exit(1)
    elif sys.version_info[0] == 3 and sys.version_info[1] < 6:
        print("WARNING, SafeLife is unsupported on Python < 3.6")

    from . import speedups  # noqa: F401
except ImportError:
    raise ImportError(
        "Cannot import module 'speedups'. "
        "Make sure that the package is correctly built and compiled using, "
        "e.g., `python3 setup.py build`."
    )
