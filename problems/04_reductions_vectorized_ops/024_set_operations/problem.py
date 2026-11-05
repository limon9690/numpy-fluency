"""024 · Set operations: isin, intersect1d, setdiff1d, union1d."""

import numpy as np


def new_ids(old: np.ndarray, current: np.ndarray) -> np.ndarray:
    """IDs that appear in `current` but not in `old`.

    Both are 1-D int arrays (duplicates allowed). Returns a sorted 1-D array without duplicates.

    Example:
        new_ids(np.array([1, 2, 3]), np.array([3, 7, 5, 7])) -> array([5, 7])
    """
    raise NotImplementedError


def keep_known(x: np.ndarray, vocab: np.ndarray, unknown) -> np.ndarray:
    """A copy of `x` where every value that is not in `vocab` is replaced by `unknown`.

    `x` has any shape; `vocab` is 1-D. Returns an array with the shape and dtype of `x`.
    `x` is not modified.

    Example:
        keep_known(np.array([[1, 2], [3, 4]]), np.array([2, 3]), -1) -> [[-1,  2],
                                                                       [ 3, -1]]
    """
    raise NotImplementedError


def common_with_positions(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values present in both 1-D arrays, and where each one first appears.

    Returns (common, ia, ib): `common` is sorted and unique; ia[k] is the index of the first
    occurrence of common[k] in `a`, ib[k] the same in `b`. So a[ia] == b[ib] == common.

    Example:
        common_with_positions(np.array([5, 1, 5, 3]), np.array([3, 5, 5, 7]))
        -> (array([3, 5]), array([3, 0]), array([0, 1]))
    """
    raise NotImplementedError


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity |A ∩ B| / |A ∪ B| of the sets of distinct values in `a` and `b`.

    Both are 1-D arrays (duplicates allowed, they count once). If both are empty, return 1.0.
    Returns a Python float.

    Example:
        jaccard(np.array([1, 2, 2, 3]), np.array([2, 3, 4])) -> 0.5     # {2, 3} / {1, 2, 3, 4}
    """
    raise NotImplementedError
