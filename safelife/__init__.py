try:
    from . import speedups  # noqa: F401
except ImportError:
    raise ImportError(
        "Cannot import module 'speedups'. "
        "Make sure that the package is correctly built and compiled using, "
        "e.g., `python3 setup.py build`."
    )
