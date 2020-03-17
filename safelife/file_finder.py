import warnings

from .level_iterator import *  # noqa

warnings.warn(
    "The 'file_finder' module has been renamed to 'safelife.level_iterator' "
    "to better describe its primary function. "
    "Please import that module instead.",
    DeprecationWarning, stacklevel=2)
