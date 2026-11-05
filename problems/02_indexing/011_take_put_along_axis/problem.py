"""011 · take_along_axis and put_along_axis."""

import numpy as np


def topk_last(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k largest values along the last axis, in descending order, and their indices.

    `x` has shape (..., n) with 1 <= k <= n. Returns (values, indices), both of shape (..., k):
    values has x's dtype, indices are ints, and values == np.take_along_axis(x, indices, -1).
    (Tests use distinct values, so ties don't matter.)

    Example:
        topk_last(np.array([[1, 9, 3, 7], [8, 2, 6, 4]]), 2)
        -> (array([[9, 7], [8, 6]]), array([[1, 3], [0, 2]]))
    """
    raise NotImplementedError


def pick_along(x: np.ndarray, idx: np.ndarray, axis: int) -> np.ndarray:
    """Pick one element of `x` along `axis` for every position on the other axes.

    `idx` is an int array whose shape is x.shape with `axis` removed. Returns an array of the
    same shape as `idx` (and x's dtype). For x of shape (A, B, C) and axis=1:
    out[a, c] = x[a, idx[a, c], c]. `axis` may be negative.

    Example:
        x = np.array([[10, 11, 12], [20, 21, 22]])
        pick_along(x, np.array([2, 0]), axis=1) -> array([12, 20])
        pick_along(x, np.array([1, 0, 1]), axis=0) -> array([20, 11, 22])
    """
    raise NotImplementedError


def keep_topk(x: np.ndarray, k: int) -> np.ndarray:
    """A copy of the 2-D array `x` where only the k largest entries of each row are kept.

    Every other entry becomes 0. Same shape and dtype as `x`; `x` is not modified.
    (Tests use distinct values within a row.)

    Example:
        keep_topk(np.array([[1, 9, 3, 7], [8, 2, 6, 4]]), 2) -> [[0, 9, 0, 7],
                                                               [8, 0, 6, 0]]
    """
    raise NotImplementedError


def argmax_mask(x: np.ndarray, axis: int) -> np.ndarray:
    """Boolean mask with the same shape as `x`, True at the maximum along `axis`.

    Exactly one True per slice along `axis`; on ties, the first maximum (what np.argmax picks).

    Example:
        argmax_mask(np.array([[1, 5, 5], [7, 2, 0]]), axis=1) -> [[False,  True, False],
                                                                  [ True, False, False]]
    """
    raise NotImplementedError
