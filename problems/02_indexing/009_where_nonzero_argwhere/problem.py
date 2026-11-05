"""009 · where, nonzero, argwhere."""

import numpy as np


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Element-wise a / b as float64, with `fill` wherever b == 0.

    `a` and `b` broadcast together. The result has the broadcast shape and dtype float64.
    No RuntimeWarning (divide by zero / invalid value) may be emitted.

    Example:
        safe_divide(np.array([1, 2, 3]), np.array([2, 0, 4]), fill=-1.0) -> array([ 0.5 , -1.  ,  0.75])
    """
    raise NotImplementedError


def bounding_box(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Smallest axis-aligned box containing every True element of `mask`.

    `mask` is a boolean array with any number of dimensions. Returns a tuple with one slice per
    axis, slice(start, stop) with start = first index that has a True along that axis and
    stop = last such index + 1, so that x[box] crops any array x of the same shape.
    Returns None if `mask` has no True element.

    Example:
        m = np.array([[0, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]], dtype=bool)
        bounding_box(m) -> (slice(1, 3), slice(1, 3))
    """
    raise NotImplementedError


def first_true(mask: np.ndarray) -> int:
    """Index of the first True element of a 1-D boolean array, or -1 if there is none.

    Returns a Python int.

    Example:
        first_true(np.array([False, False, True, True])) -> 2
        first_true(np.array([False, False]))             -> -1
    """
    raise NotImplementedError


def piecewise_curve(x: np.ndarray) -> np.ndarray:
    """Evaluate a piecewise function element-wise.

        f(x) = x**2   if x < 0
               x      if 0 <= x < 1
               1      if x >= 1

    `x` is a float array of any shape. Returns a float64 array of the same shape.

    Example:
        piecewise_curve(np.array([-2.0, 0.5, 3.0])) -> array([4. , 0.5, 1. ])
    """
    raise NotImplementedError
