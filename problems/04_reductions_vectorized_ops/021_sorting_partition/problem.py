"""021 · sort, stable argsort, lexsort, partition."""

import numpy as np


def sort_rows_by_column(data: np.ndarray, col: int) -> np.ndarray:
    """Rows of the 2-D array `data` sorted by the values in column `col`, ascending.

    The sort must be stable: rows with equal keys keep their original relative order.
    Returns a new array with the same shape and dtype.

    Example:
        data = np.array([[2, 0], [1, 1], [2, 2], [1, 3]])
        sort_rows_by_column(data, 0) -> [[1, 1],
                                        [1, 3],
                                        [2, 0],
                                        [2, 2]]
    """
    raise NotImplementedError


def order_by_two_keys(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Indices that sort the records by `primary`, then by `secondary` among equal primaries.

    Both are 1-D arrays of the same length n. If both keys are equal, the original order is
    kept. Returns an int array of shape (n,).

    Example:
        order_by_two_keys(np.array([2, 1, 2, 1]), np.array([30, 40, 20, 25])) -> array([3, 1, 2, 0])
    """
    raise NotImplementedError


def k_smallest(x: np.ndarray, k: int) -> np.ndarray:
    """The k smallest values of the 1-D array `x`, in any order.

    1 <= k <= len(x). Returns shape (k,), same dtype. Must run in O(n): np.sort and
    np.argsort are banned (use np.partition).

    Example:
        np.sort(k_smallest(np.array([7, 2, 9, 1, 5]), 3)) -> array([1, 2, 5])
    """
    raise NotImplementedError


def rank(x: np.ndarray) -> np.ndarray:
    """Position of every element in the stable ascending sort of `x` (0 = smallest).

    `x` is 1-D. Ties are ranked by original position (earlier element gets the lower rank).
    Returns an int array of the same shape; it is a permutation of 0..n-1.

    Example:
        rank(np.array([30, 10, 20, 10])) -> array([3, 0, 2, 1])
    """
    raise NotImplementedError
