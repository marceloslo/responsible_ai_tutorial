"""Utils for handling dictionaries.
"""
import inspect
import operator
from functools import reduce
from typing import Mapping, Callable


def join_dictionaries(*dicts) -> dict:
    """Join multiple dictionaries into one."""
    return reduce(operator.or_, dicts)


def fit_dict(dct: dict, func: Callable, accept_kwargs: bool = True) -> dict:
    """Returns a subset of (k, v) pairs from `dct` whose key
    matches an argument of the callable `func`.

    If accept_kwargs==True and the function does accept key-word arguments,
    then all arguments are accepted.
    """
    func_spec = inspect.getfullargspec(func)
    func_args = func_spec.args
    if func_spec.varkw and accept_kwargs:
        return dct

    return {k: v for k, v in dct.items() if k in func_args}


def apply_recursively(
    dct: dict,
    apply: Callable[[object], object],
    pred: Callable[[object], bool] = (lambda _k: True),
) -> dict:
    """Applies a function recursively to the provided dictionary, possibly
    filtering the fields to which it is applied.

    Parameters
    ----------
    dct : dict
        The dictionary to which the callable `apply` will be recursively applied.

    apply : Callable
        The function to apply to the dictionary's fields.

    pred : Callable
        Predicate to filter which fields to apply the function to.
        Receives the (key, value) pair as an input.
    """
    for k in dct.keys():
        if isinstance(dct[k], Mapping):
            dct[k] = apply_recursively(dct[k], apply, pred)
        elif pred(k):
            dct[k] = apply(dct[k])
    return dct
