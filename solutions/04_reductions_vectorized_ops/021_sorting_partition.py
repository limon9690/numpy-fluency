"""021 · sort, stable argsort, lexsort, partition (reference solution)."""

import numpy as np


def sort_rows_by_column(data: np.ndarray, col: int) -> np.ndarray:
    # The default quicksort may reorder equal keys; "stable" guarantees it won't.
    return data[np.argsort(data[:, col], kind="stable")]


def order_by_two_keys(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    return np.lexsort((secondary, primary))  # last key is the primary one


def k_smallest(x: np.ndarray, k: int) -> np.ndarray:
    # After partitioning, the first k slots hold the k smallest values (unordered).
    return np.partition(x, k - 1)[:k]


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(x))  # inverse permutation of the sort order
    return ranks
