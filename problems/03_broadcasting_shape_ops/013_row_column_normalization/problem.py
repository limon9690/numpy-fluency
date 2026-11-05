"""013 · Row and column scaling and normalization."""

import numpy as np


def row_normalize(a: np.ndarray) -> np.ndarray:
    """Divide every row of the non-negative 2-D array `a` by its sum, so each row sums to 1.

    Rows that are all zero stay all zero (no NaN, no warnings). Returns a new float array of the
    same shape: float64 for integer input, same dtype for float input. `a` is not modified.

    Example:
        row_normalize(np.array([[1, 3], [0, 0], [2, 2]])) -> [[0.25, 0.75],
                                                             [0.  , 0.  ],
                                                             [0.5 , 0.5 ]]
    """
    raise NotImplementedError


def standardize_columns(a: np.ndarray) -> np.ndarray:
    """Standardize every column of the 2-D array `a` to mean 0 and std 1 (population std, ddof=0).

    Constant columns (std == 0) become all zeros (no NaN, no warnings). Returns a new float array
    of the same shape: float64 for integer input, same dtype for float input.

    Example:
        standardize_columns(np.array([[1, 5], [3, 5]])) -> [[-1., 0.],
                                                           [ 1., 0.]]
    """
    raise NotImplementedError


def minmax_scale_columns(a: np.ndarray) -> np.ndarray:
    """Rescale every column of the 2-D array `a` to [0, 1] with (x - col_min) / (col_max - col_min).

    Constant columns become all zeros. Returns a new float array of the same shape: float64 for
    integer input, same dtype for float input.

    Example:
        minmax_scale_columns(np.array([[0, 10], [5, 10], [10, 10]])) -> [[0. , 0.],
                                                                        [0.5, 0.],
                                                                        [1. , 0.]]
    """
    raise NotImplementedError


def scale_rows(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Multiply row i of the 2-D array `a` (shape (n, m)) by w[i] (`w` has shape (n,)).

    Returns a new array of shape (n, m). Careful: when n == m, `a * w` runs without error but
    scales the columns instead.

    Example:
        scale_rows(np.array([[1, 1], [1, 1]]), np.array([2, 3])) -> [[2, 2],
                                                                    [3, 3]]
    """
    raise NotImplementedError
