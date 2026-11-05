"""014 · reshape, ravel vs flatten, transpose, moveaxis."""

import numpy as np


def to_patches(img: np.ndarray, p: int) -> np.ndarray:
    """Cut a 2-D image into non-overlapping p x p patches.

    `img` has shape (H, W) with H and W multiples of p. Returns shape (H//p * W//p, p, p).
    Patches are ordered row by row over the patch grid: patch k covers rows
    (k // (W//p)) * p ... + p and columns (k % (W//p)) * p ... + p. Same dtype as `img`.

    Example:
        to_patches(np.arange(16).reshape(4, 4), 2)[1] -> [[2, 3],
                                                         [6, 7]]
    """
    raise NotImplementedError


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    """Move the channel axis from last to before height and width.

    (H, W, C) -> (C, H, W) and (N, H, W, C) -> (N, C, H, W).
    Return a view (no copy): the result shares memory with `img`.

    Example:
        hwc_to_chw(np.zeros((32, 48, 3))).shape    -> (3, 32, 48)
        hwc_to_chw(np.zeros((8, 32, 48, 3))).shape -> (8, 3, 32, 48)
    """
    raise NotImplementedError


def flatten_batch(x: np.ndarray) -> np.ndarray:
    """Flatten every axis except the first: (B, d1, d2, ...) -> (B, d1 * d2 * ...).

    Must work when B == 0, and for 1-D input (B,) -> (B, 1). Each row holds the elements of
    x[b] in C order. Same dtype as `x`.

    Example:
        flatten_batch(np.zeros((5, 3, 4))).shape -> (5, 12)
        flatten_batch(np.zeros((0, 3, 4))).shape -> (0, 12)
    """
    raise NotImplementedError


def increment_flat(a: np.ndarray, i: int, value) -> None:
    """In place, add `value` to the element of `a` at flat index `i` (C order). Returns None.

    Flat index i means "the i-th element when `a` is read in C order", the same element as
    a.ravel()[i]. Must also work when `a` is not contiguous (e.g. a transposed view), and the
    change must be visible in `a` (and in any array `a` is a view of).

    Example:
        a = np.zeros((2, 3)).T        # shape (3, 2), not contiguous
        increment_flat(a, 1, 5.0)     # a[0, 1] is now 5.0
    """
    raise NotImplementedError
