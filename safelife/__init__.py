import sys

if sys.version_info < (3,5):
    version_msg = (
        "\n\nSafeLife requires Python 3.5+\n"
        "(You are using python version %i.%i.%i)\n" % sys.version_info[:3]
    )
    if sys.version_info < (3,):
        raise ImportError(version_msg)
    else:
        import warnings
        warnings.warn(version_msg)

try:
    from . import speedups  # noqa: F401
except ImportError:
    raise ImportError(
        "Cannot import module 'speedups'. "
        "Make sure that the package is correctly built and compiled using, "
        "e.g., `python3 setup.py build`."
    )
