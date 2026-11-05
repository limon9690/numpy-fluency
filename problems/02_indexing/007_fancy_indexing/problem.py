"""007 · Fancy indexing."""

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer class labels.

    `labels` is a 1-D int array of shape (n,) with values in [0, num_classes).
    Returns a float32 array of shape (n, num_classes) where row i has a 1 at column labels[i]
    and 0 elsewhere.

    Example:
        one_hot(np.array([1, 3, 0]), 4) -> [[0, 1, 0, 0],
                                           [0, 0, 0, 1],
                                           [1, 0, 0, 0]]   (float32)
    """
    raise NotImplementedError


def pick_per_row(a: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pick one element from every row: out[i] = a[i, cols[i]].

    `a` has shape (n, m), `cols` is an int array of shape (n,). Returns shape (n,), same dtype as `a`.

    Example:
        a = np.array([[10, 11, 12], [20, 21, 22]])
        pick_per_row(a, np.array([2, 0])) -> array([12, 20])
    """
    raise NotImplementedError


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a lookup table: replace every value v in `img` by lut[v].

    `img` is a uint8 array of any shape; `lut` is a uint8 array of shape (256,).
    Returns a uint8 array with the same shape as `img`.

    Example:
        invert = (255 - np.arange(256)).astype(np.uint8)
        apply_lut(np.array([[0, 10], [255, 128]], np.uint8), invert) -> [[255, 245], [0, 127]]
    """
    raise NotImplementedError


def inverse_permutation(p: np.ndarray) -> np.ndarray:
    """Invert a permutation (np.argsort is banned).

    `p` is a 1-D int array containing each of 0, 1, ..., n-1 exactly once.
    Returns `inv` (same shape and dtype) with inv[p[i]] = i, so that p[inv] and inv[p] are both
    np.arange(n).

    Example:
        inverse_permutation(np.array([2, 0, 1])) -> array([1, 2, 0])
    """
    raise NotImplementedError
