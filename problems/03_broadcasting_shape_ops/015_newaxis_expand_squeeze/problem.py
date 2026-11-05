"""015 · newaxis, expand_dims, squeeze."""

import numpy as np


def ensure_batch_image(img: np.ndarray) -> np.ndarray:
    """Return `img` as a 4-D batch (N, H, W, C), as a view.

        (H, W)        -> (1, H, W, 1)
        (H, W, C)     -> (1, H, W, C)
        (N, H, W, C)  -> unchanged
        anything else -> ValueError

    The result shares memory with `img`.

    Example:
        ensure_batch_image(np.zeros((28, 28))).shape -> (1, 28, 28, 1)
    """
    raise NotImplementedError


def squeeze_except(x: np.ndarray, axis: int) -> np.ndarray:
    """Remove every axis of length 1 except `axis`, which is always kept.

    `axis` may be negative (counted from the end of x.shape). Returns a view.

    Example:
        squeeze_except(np.zeros((1, 3, 1, 5)), 0).shape  -> (1, 3, 5)
        squeeze_except(np.zeros((1, 3, 1, 5)), -2).shape -> (3, 1, 5)
    """
    raise NotImplementedError


def broadcast_along(v: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    """Reshape the 1-D array `v` so it broadcasts along `axis` of an ndim-dimensional array.

    The result has `ndim` dimensions: length len(v) at `axis` and 1 everywhere else.
    `axis` may be negative. Returns a view of `v`.

    Example:
        b = np.array([1., 2., 3.])
        broadcast_along(b, 4, 1).shape  -> (1, 3, 1, 1)    # add to (N, C, H, W)
        broadcast_along(b, 3, -1).shape -> (1, 1, 3)       # add to (H, W, C)
    """
    raise NotImplementedError
