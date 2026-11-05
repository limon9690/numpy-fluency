"""019 · ufuncs: out=, where=, reduce, accumulate, outer."""

import numpy as np


def safe_log(x: np.ndarray, fill: float = -np.inf) -> np.ndarray:
    """Natural log of `x` where x > 0, and `fill` where x <= 0 (or NaN).

    Returns float64 with the shape of `x`. Must not emit any RuntimeWarning.

    Example:
        safe_log(np.array([1.0, 0.0, -1.0, np.e]), fill=-99.0) -> array([  0., -99., -99.,   1.])
    """
    raise NotImplementedError


def running_max(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """The running (cumulative) maximum of `x` along `axis`.

    out[..., i] = max(x[..., 0], ..., x[..., i]) along that axis. Same shape and dtype as `x`.

    Example:
        running_max(np.array([[1, 3, 2], [5, 4, 6]]), axis=1) -> [[1, 3, 3],
                                                                 [5, 5, 6]]
    """
    raise NotImplementedError


def gcd_all(x: np.ndarray) -> int:
    """The greatest common divisor of all elements of the integer array `x` (any shape).

    The result is non-negative. gcd of all zeros is 0. Returns a Python int.

    Example:
        gcd_all(np.array([[12, 18], [30, -42]])) -> 6
    """
    raise NotImplementedError


def pairs_within(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """Boolean table of which pairs are close: out[i, j] = |a[i] - b[j]| <= tol.

    `a` has shape (n,), `b` has shape (m,). Returns a bool array of shape (n, m).

    Example:
        pairs_within(np.array([1.0, 5.0]), np.array([1.5, 4.0, 9.0]), 1.0)
        -> [[ True, False, False],
            [False,  True, False]]
    """
    raise NotImplementedError
