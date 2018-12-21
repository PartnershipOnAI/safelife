import numpy as np
import scipy.signal
import pyemd


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
    return y.astype(np.int16)


def earth_mover_distance(
        a, b, metric="manhattan", wrap_x=False, wrap_y=False,
        tanh_scale=0, extra_mass_penalty=0.0):
    """
    Calculate the earth mover distance between two 2d distributions.

    Parameters
    ----------
    a, b: ndarray
        Must be same shape, both floats.
    metric: str
        Either "manhattan" or "euclidean". Coordinate points are assumed to
        be equal to the indices of the grid.
    wrap_x, wrap_y: bool
        If true, treat the grid as a cylinder or torus (if both true).
    tanh_scale: float
        If provided, take the tanh of the distance metric divided by this
        scale. This effectively puts a cap on how large the distance can be
        between any two points.
    extra_mass_penalty: float
        Penalty for extra mass that needs to be added to the distributions.
        If less than zero, defaults to the largest distance possible on the
        grid.
    """
    a = np.asanyarray(a, dtype=float)
    b = np.asanyarray(b, dtype=float)
    x, y = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
    # Only need to look at the points that are not common to both.
    delta = np.abs(a - b)
    changed = delta > 1e-3 * np.max(delta)
    if not changed.any():
        return 0.0
    dx = np.subtract.outer(x[changed], x[changed])
    dy = np.subtract.outer(y[changed], y[changed])
    if wrap_x:
        dx = np.minimum(dx, a.shape[1] - dx)
    if wrap_y:
        dy = np.minimum(dy, a.shape[0] - dy)
    if metric == "manhattan":
        dist = (np.abs(dx) + np.abs(dy)).astype(float)
    else:
        dist = np.sqrt(dx*dx + dy*dy)
    if tanh_scale > 0:
        dist = np.tanh(dist / tanh_scale)
    return pyemd.emd(a[changed], b[changed], dist, extra_mass_penalty)
