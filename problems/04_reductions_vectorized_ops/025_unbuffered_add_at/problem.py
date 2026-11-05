"""025 · np.add.at and scatter-style updates."""

import numpy as np


def scatter_add(target: np.ndarray, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Add values[k] to position idx[k] of a copy of `target`, for every k.

    `target` is 1-D; `idx` and `values` have the same shape (m,). Repeated indices must add up
    (unlike `target[idx] += values`). Returns a new array with target's shape and dtype;
    `target` is not modified.

    Example:
        scatter_add(np.zeros(4), np.array([0, 2, 2, 2]), np.array([1., 1., 2., 3.]))
        -> array([1., 0., 6., 0.])
    """
    raise NotImplementedError


def pixel_counts(ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> np.ndarray:
    """Count how many points land on each pixel of an h x w grid.

    `ys` and `xs` are int arrays of shape (m,) with 0 <= ys < h and 0 <= xs < w.
    Returns an int64 array of shape (h, w) with out[y, x] = number of k with (ys[k], xs[k]) == (y, x).

    Example:
        pixel_counts(np.array([0, 0, 1]), np.array([2, 2, 0]), 2, 3) -> [[0, 0, 2],
                                                                        [1, 0, 0]]
    """
    raise NotImplementedError


def scatter_max(n: int, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    """For each of n slots, the largest value sent to it.

    `idx` (ints in [0, n)) and `values` (floats) have the same shape (m,). Returns float64 of
    shape (n,): out[j] = max of values[idx == j], or -inf if no value went to slot j.

    Example:
        scatter_max(3, np.array([0, 2, 0]), np.array([1.0, 5.0, 4.0])) -> array([  4., -inf,   5.])
    """
    raise NotImplementedError


def embedding_grad(indices: np.ndarray, grad_out: np.ndarray, vocab_size: int) -> np.ndarray:
    """Gradient of an embedding lookup `table[indices]` with respect to `table`.

    `indices` is an int array of shape (B, T) with values in [0, vocab_size); `grad_out` has
    shape (B, T, D). Returns an array of shape (vocab_size, D) with grad_out's dtype where
    row v is the sum of grad_out[b, t] over all (b, t) with indices[b, t] == v (zeros if none).

    Example:
        indices = np.array([[1, 1]]); grad_out = np.array([[[1., 2.], [3., 4.]]])
        embedding_grad(indices, grad_out, 3) -> [[0., 0.],
                                                [4., 6.],
                                                [0., 0.]]
    """
    raise NotImplementedError
