"""028 · Percentiles, clip and winsorize."""

import numpy as np


def iqr_outlier_mask(x: np.ndarray) -> np.ndarray:
    """Tukey outliers of the 1-D array `x`.

    Q1 and Q3 are the 25th and 75th percentiles (NumPy's default linear method), IQR = Q3 - Q1.
    Returns a bool array of the same shape, True where x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR.

    Example:
        iqr_outlier_mask(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]))
        -> [False, ..., False, True]     # only 100 is an outlier
    """
    raise NotImplementedError


def winsorize(x: np.ndarray, lower_pct: float, upper_pct: float, axis: int | None = None) -> np.ndarray:
    """Clip `x` at its own lower_pct and upper_pct percentiles (0..100).

    axis=None: one pair of bounds for the whole array. axis=k: separate bounds for every slice
    along axis k (e.g. axis=0 on a 2-D array: per column). Returns a new float array with the
    shape of `x`; `x` is not modified.

    Example:
        winsorize(np.arange(11.0), 10, 90) -> array([1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9.])
    """
    raise NotImplementedError


def robust_scale(X: np.ndarray) -> np.ndarray:
    """Scale every column of the 2-D array X to (x - median) / IQR.

    median and IQR (75th minus 25th percentile, linear method) are per column. Columns with
    IQR == 0 are divided by 1 instead (only centered). Returns float64 of the same shape.

    Example:
        robust_scale(np.array([[1., 5.], [2., 5.], [3., 5.], [4., 5.], [100., 5.]]))
        -> [[-1. ,  0. ],
            [-0.5,  0. ],
            [ 0. ,  0. ],
            [ 0.5,  0. ],
            [48.5,  0. ]]      # column 0: median 3, IQR 4 - 2 = 2; column 1 is constant
    """
    raise NotImplementedError


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign every value of the 1-D array `x` to one of n_bins quantile bins.

    edges = the quantiles of x at 0, 1/n_bins, 2/n_bins, ..., 1 (default linear method).
    Bin k holds values v with edges[k] <= v < edges[k+1]; the maximum goes in the last bin.
    Returns an int array of the same shape with values in 0..n_bins-1.

    Example:
        quantile_bins(np.array([5, 1, 4, 2, 3, 6, 8, 7]), 4) -> array([2, 0, 1, 0, 1, 2, 3, 3])
    """
    raise NotImplementedError
