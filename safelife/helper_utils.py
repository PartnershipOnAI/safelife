import inspect
import numpy as np
import scipy.signal


class wrapping_array(np.ndarray):
    """
    Same as a normal array, but slicing wraps around boundaries.

    Assumes that the array is 2d.
    """
    def __getitem__(self, items):
        # To make life easier, we only wrap when `items` is a tuple of slices.
        # It'd be nifty if they could be integers or work on arbitrary arrays,
        # but that's more work and it won't be used.
        if not (isinstance(items, tuple) and len(items) == 2 and
                isinstance(items[0], slice) and isinstance(items[1], slice)):
            return super().__getitem__(items)
        rows, cols = items
        rows = np.arange(
            rows.start or 0,
            rows.stop if rows.stop is not None else self.shape[0],
            rows.step or 1
        ) % self.shape[0]
        cols = np.arange(
            cols.start or 0,
            cols.stop if cols.stop is not None else self.shape[0],
            cols.step or 1
        ) % self.shape[1]
        nrows = len(rows)
        ncols = len(cols)
        rows = np.broadcast_to(rows[:,None], (nrows, ncols))
        cols = np.broadcast_to(cols[None,:], (nrows, ncols))
        return super().__getitem__((rows, cols))


def wrapped_convolution(*args, **kw):
    y = scipy.signal.convolve2d(*args, boundary='wrap', mode='same', **kw)
    return y.astype(np.uint16)


def recenter_view(board, view_size, center, move_to_perimeter=None):
    """
    Create a view of the input array centered on a specific location.

    Parameters
    ----------
    board : ndarray
        Two-dimensional array to be centered.
    view_size : tuple
    center : tuple
    move_to_perimeter : ([int], [int]), optional
        Lists of indices that should be moved to the view perimeter if they
        would otherwise be out of sight. This the general direction of the
        features can be provided even if they can't be seen directly.
    """
    h, w = view_size
    bh, bw = board.shape
    y0, x0 = center
    x1 = x0 - w // 2
    y1 = y0 - h // 2
    board2 = board.view(wrapping_array)[y1:y1+h, x1:x1+w]
    board2 = board2.view(np.ndarray)
    if move_to_perimeter is not None:
        iy, ix = move_to_perimeter
        # Calculate indices relative to the center point.
        # Use the modulo operation to wrap to [-bw/2, +bw/2]
        jy = (iy - y0 + bh // 2) % bh - bh // 2
        jx = (ix - x0 + bw // 2) % bw - bw // 2
        # Clip the indices to the view
        jy = np.clip(jy + h // 2, 0, h-1)
        jx = np.clip(jx + w // 2, 0, w-1)
        # and replace the board values.
        board2[jy, jx] = board[iy, ix]
    return board2


def load_kwargs(self, kwargs):
    """
    Simple function to load kwargs during class initialization.
    """
    for key, val in kwargs.items():
        if (not key.startswith('_') and hasattr(self, key) and
                not inspect.ismethod(getattr(self, key))):
            setattr(self, key, val)
        else:
            raise ValueError("Unrecognized parameter: '%s'" % (key,))
