"""006 · Slicing and basic indexing."""

import numpy as np


def crop_center(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """The central h x w region of an image, as a view.

    `img` has shape (H, W) or (H, W, C), with h <= H and w <= W. The crop's top-left corner is
    at row (H - h) // 2 and column (W - w) // 2. The result has shape (h, w) or (h, w, C) and
    shares memory with `img`.

    Example:
        img = np.arange(16).reshape(4, 4)
        crop_center(img, 2, 2) -> [[ 5,  6],
                                   [ 9, 10]]
    """
    raise NotImplementedError


def flip_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """`a` reversed along `axis`, as a view (np.flip is banned).

    `a` can have any number of dimensions; `axis` may be negative (-1 = last axis).
    The result has the same shape and dtype and shares memory with `a`.

    Example:
        flip_axis(np.array([[1, 2, 3], [4, 5, 6]]), 1) -> [[3, 2, 1],
                                                           [6, 5, 4]]
    """
    raise NotImplementedError


def checkerboard(n: int) -> np.ndarray:
    """An n x n int8 checkerboard of 0s and 1s with 0 in the top-left corner.

    Element (i, j) is (i + j) % 2.

    Example:
        checkerboard(3) -> [[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]]   (dtype int8)
    """
    raise NotImplementedError


def last_k(a: np.ndarray, k: int) -> np.ndarray:
    """The last k entries of `a` along axis 0, as a view.

    0 <= k <= len(a). For k = 0 the result is empty with shape (0, *a.shape[1:]).

    Example:
        last_k(np.arange(5), 2) -> array([3, 4])
        last_k(np.arange(5), 0) -> array([], dtype=int64)
    """
    raise NotImplementedError
