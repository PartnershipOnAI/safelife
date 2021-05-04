from collections import namedtuple
from functools import wraps

import numpy as np

_no_default = object()


def named_output(names):
    """
    A simple decorator to transform a function's output to a named tuple.

    For example,

        @named_output(['foo', 'bar'])
        def my_func():
            return 1, 2

    would, when called, return a named tuple ``my_func_rval(foo=1, bar=2)``.
    This is handy when returning lots of values from one function.

    Note that the input can either be a list of names or a space-separated
    string of names.
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


def check_shape(tensor, shape, name=""):
    observed = tensor.shape
    for axis, value in enumerate(shape):
        if type(value) is str or value is None:
            continue
        if value != observed[axis]:
            raise TypeError(
                f"Tensor passed to {name} is of shape {observed} but should be {shape}")


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


def nested_getattr(obj, key, default=_no_default):
    """
    Get a named attribute from an object with support for nested keys.

    This is equivalent to the built-in function ``getattr``, except that
    they keys can have dots in them to signify nested attributes.

    When a default argument is given, it is returned when the attribute
    doesn't exist; without it, an exception is raised in that case.

    Example
    -------
    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(a=SimpleNamespace(b='hello!'))
    >>> nested_getattr(x, 'a.b')
        'hello!'
    >>> x.a.b == nested_getattr(x, 'a.b')
        True
    """
    obj2 = obj
    for subkey in key.split('.'):
        obj2 = getattr(obj2, subkey, _no_default)
    if obj2 is _no_default:
        if default is _no_default:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (obj, key))
        else:
            obj2 = default
    return obj2


def nested_setattr(obj, key, val):
    """
    Sets a named attribute on an object with support for nested keys.

    This is equivalent to the built-in function ``setattr``, except that
    they keys can have dots in them to signify nested attributes.

    Note that if an intermediate key does not exist, ``AttributeError``
    will be raised.

    Example
    -------
    >>> from types import SimpleNamespace
    >>> x = SimpleNamespace(a=SimpleNamespace())
    >>> nested_setattr(x, 'a.b', 'hello!')
    >>> x.a.b
        'hello!'
    """
    obj_key, _, set_key = key.rpartition('.')
    if obj_key:
        obj = nested_getattr(obj, obj_key)
    setattr(obj, set_key, val)


def get_compute_device():
    """
    Try first to get TPU, then cuda, then CPU.
    """
    import torch
    try:
        # TPU
        import torch_xla.core.xla_model as xm
        # import os
        # os.environ["XLA_USE_BF16"] = "1"
        return xm.xla_device()
    except (ModuleNotFoundError, ImportError):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


def recursive_shape(obj, indent=0):
    """
    Recursively prettyprint the shape of a hierarchical torch object.
    """
    s = ""
    if isinstance(obj, list) or isinstance(obj, tuple):
        c1, c2 = ["[", "]"] if isinstance(obj, list) else ["(", ")"]
        s += indent * " " + c1 + "\n"
        indent += 2
        for item in obj:
            s += recursive_shape(item, indent) + ",\n"
        indent -= 2
        s += indent * " " + c2 + "\n"
    else:
        if hasattr(obj, "shape"):
            s += " " * indent + str(obj.shape)
        else:
            s += " " * indent + str(obj)

    return s
