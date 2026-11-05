"""008 · Boolean masking."""

import numpy as np


def count_in_range(x: np.ndarray, low: float, high: float) -> int:
    """Number of elements of `x` (any shape) with low <= value <= high (both ends included).

    Returns an int.

    Example:
        count_in_range(np.array([[1, 5, 9], [3, 4, 10]]), 3, 9) -> 4
    """
    raise NotImplementedError


def replace_outside(x: np.ndarray, low: float, high: float, fill: float) -> np.ndarray:
    """A copy of `x` where every value < low or > high is replaced by `fill`.

    Values inside [low, high] are kept. Same shape and dtype as `x`. `x` is not modified.

    Example:
        replace_outside(np.array([-5, 0, 3, 12]), 0, 10, -1) -> array([-1,  0,  3, -1])
    """
    raise NotImplementedError


def select_rows(data: np.ndarray, col: int, threshold: float) -> np.ndarray:
    """The rows of `data` whose value in column `col` is > threshold.

    `data` has shape (n, m). Returns shape (k, m), rows kept in their original order
    (k may be 0). Same dtype as `data`.

    Example:
        data = np.array([[1, 50], [2, 10], [3, 70]])
        select_rows(data, 1, 20) -> [[1, 50],
                                    [3, 70]]
    """
    raise NotImplementedError


def drop_nan_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop every position i where x[i] or y[i] is NaN.

    `x` and `y` are 1-D float arrays of the same length. Returns (x_clean, y_clean): equal-length
    1-D arrays with the remaining values in their original order. Inputs are not modified.

    Example:
        drop_nan_pairs(np.array([1., nan, 3., 4.]), np.array([10., 20., nan, 40.]))
        -> (array([1., 4.]), array([10., 40.]))
    """
    raise NotImplementedError
