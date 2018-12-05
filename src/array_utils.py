import numpy as np


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
