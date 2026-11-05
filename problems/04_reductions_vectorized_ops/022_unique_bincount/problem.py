"""022 · unique and bincount."""

import numpy as np


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label-encode an array: map every value to the index of its class.

    `labels` is an int or float array of any shape. Returns (classes, codes):
        classes - 1-D array of the sorted distinct values (same dtype as labels)
        codes   - int array with the SAME SHAPE as `labels`, classes[codes] == labels

    Example:
        encode_labels(np.array([30, 10, 30, 20])) -> (array([10, 20, 30]), array([2, 0, 2, 1]))
    """
    raise NotImplementedError


def unique_rows_counts(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct rows of the 2-D array `a` and how often each one occurs.

    Returns (rows, counts): rows has shape (k, a.shape[1]), sorted lexicographically (what
    np.unique gives), counts is an int array of shape (k,).

    Example:
        unique_rows_counts(np.array([[1, 2], [3, 4], [1, 2]]))
        -> (array([[1, 2], [3, 4]]), array([2, 1]))
    """
    raise NotImplementedError


def most_frequent(x: np.ndarray) -> int:
    """The most common value in the 1-D array `x` of non-negative ints.

    On ties, return the smallest such value. Returns a Python int. np.unique is banned.

    Example:
        most_frequent(np.array([3, 1, 3, 1, 2])) -> 1
    """
    raise NotImplementedError


def group_mean(groups: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean of `values` within each group.

    `groups` is a 1-D int array of group ids in [0, n_groups); `values` is a 1-D float array of
    the same length. Returns float64 of shape (n_groups,): out[g] = mean of values[groups == g],
    or NaN if group g is empty. No warnings, no Python loops, np.unique is banned.

    Example:
        group_mean(np.array([0, 1, 0, 3]), np.array([10., 5., 20., 7.]), 4) -> array([15., 5., nan, 7.])
    """
    raise NotImplementedError
