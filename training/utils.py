from collections import namedtuple
from functools import wraps

import numpy as np


def named_output(names):
    """
    A simple decorator to transform a function's output to a named tuple.

    For example,

        @named_output(['foo', 'bar'])
        def my_func():
            return 1, 2

    would, when called, return a named tuple ``my_func_rval(foo=1, bar=2)``.
    This is handy when returning lots of values from one function.
    """
    def decorator(func):
        rtype = namedtuple(func.__name__ + '_rval', names)

        @wraps(func)
        def wrapped(*args, **kwargs):
            rval = func(*args, **kwargs)
            if isinstance(rval, tuple):
                rval = rtype(*rval)
            return rval
        return wrapped

    return decorator


def round_up(x, r):
    """
    Round x up to the nearest multiple of r.

    Always rounds up, even if x is already divisible by r.
    """
    return x + r - x % r


def shuffle_arrays_in_place(*data):
    """
    This runs np.random.shuffle on multiple inputs, shuffling each in place
    in the same order (assuming they're the same length).
    """
    rng_state = np.random.get_state()
    for x in data:
        np.random.set_state(rng_state)
        np.random.shuffle(x)


def shuffle_arrays(*data):
    # Despite the nested for loops, this is actually a little bit faster
    # than the above because it doesn't involve any copying of array elements.
    # When the array elements are large (like environment states),
    # that overhead can be large.
    idx = np.random.permutation(len(data[0]))
    return [[x[i] for i in idx] for x in data]
