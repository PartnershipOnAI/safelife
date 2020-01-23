"""
Module that stores global random state for SafeLife.

Note that this uses the new numpy 1.17 random generators.
"""

from contextlib import contextmanager
import numpy as np

from . import speedups


random_gen = np.random.default_rng()  # global random generator object


def get_rng():
    return random_gen


@contextmanager
def set_rng(new_rng):
    global random_gen
    old_rng = random_gen
    random_gen = new_rng
    speedups.set_bit_generator(new_rng.bit_generator)
    try:
        yield
    finally:
        random_gen = old_rng
        speedups.set_bit_generator(old_rng.bit_generator)


def coinflip(p, n=None):
    """
    Return True with probability `p`, False with probability `1-p`.

    Parameters
    ----------
    p : float
    n : None or int or tuple
        If not None, return an array of `n` coin flips.
        Tuples can be used to return a multi-dimensional array.
    """
    return random_gen.random(n) < p
