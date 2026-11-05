"""030 · NaN and inf: isfinite, nan* functions, errstate."""

import numpy as np


def nan_column_means(X: np.ndarray) -> np.ndarray:
    """Mean of every column of the 2-D float array X, ignoring NaN.

    Columns that are entirely NaN give NaN. No warnings of any kind may be emitted.
    Returns float64 of shape (X.shape[1],).

    Example:
        nan_column_means(np.array([[1., np.nan, 5.], [3., np.nan, np.nan]])) -> array([2., nan, 5.])
    """
    raise NotImplementedError


def fill_nan_with_column_mean(X: np.ndarray) -> np.ndarray:
    """A copy of the 2-D float array X with every NaN replaced by its column's mean.

    The column mean ignores NaN; for a column that is entirely NaN, use 0.0. Non-NaN values
    (including inf) are kept. X is not modified. No warnings.

    Example:
        fill_nan_with_column_mean(np.array([[1., np.nan], [np.nan, np.nan], [3., np.nan]]))
        -> [[1., 0.],
            [2., 0.],
            [3., 0.]]
    """
    raise NotImplementedError


def finite_stats(x: np.ndarray) -> dict:
    """Statistics over the finite values of `x` (any shape), skipping NaN, inf and -inf.

    Returns {"min": float, "max": float, "mean": float, "count": int} with Python types.
    Raises ValueError if `x` has no finite value.

    Example:
        finite_stats(np.array([1.0, np.nan, 3.0, np.inf, -np.inf]))
        -> {"min": 1.0, "max": 3.0, "mean": 2.0, "count": 2}
    """
    raise NotImplementedError


def relative_change(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Element-wise (new - old) / old, as float64.

    Where old == 0 the result is inf (new > 0), -inf (new < 0) or nan (new == 0), exactly what
    IEEE division gives, but no RuntimeWarning may be emitted. Inputs broadcast together.

    Example:
        relative_change(np.array([2., 0., 0., 0.]), np.array([3., 1., -1., 0.]))
        -> array([ 0.5,  inf, -inf,  nan])
    """
    raise NotImplementedError
