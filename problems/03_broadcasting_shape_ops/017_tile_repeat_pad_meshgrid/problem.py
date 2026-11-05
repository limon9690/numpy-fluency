"""017 · tile, repeat, pad, meshgrid."""

import numpy as np


def upsample_nearest(img: np.ndarray, k: int) -> np.ndarray:
    """Nearest-neighbour upsampling by an integer factor k.

    `img` has shape (H, W) or (H, W, C). Every pixel becomes a k x k block of the same value.
    Returns shape (k*H, k*W) or (k*H, k*W, C), same dtype. out[i, j] == img[i // k, j // k].

    Example:
        upsample_nearest(np.array([[1, 2], [3, 4]]), 2) -> [[1, 1, 2, 2],
                                                           [1, 1, 2, 2],
                                                           [3, 3, 4, 4],
                                                           [3, 3, 4, 4]]
    """
    raise NotImplementedError


def pad_to_multiple(img: np.ndarray, m: int) -> np.ndarray:
    """Zero-pad an image at the bottom and right so its height and width are multiples of m.

    `img` has shape (H, W) or (H, W, C). Adds (-H) % m rows at the bottom and (-W) % m columns on
    the right, filled with 0. The channel axis is never padded. Same dtype; the original pixels
    stay at the top-left.

    Example:
        pad_to_multiple(np.ones((3, 5)), 4).shape     -> (4, 8)
        pad_to_multiple(np.ones((8, 8, 3)), 4).shape  -> (8, 8, 3)
    """
    raise NotImplementedError


def coordinate_grid(h: int, w: int) -> np.ndarray:
    """An (h, w, 2) integer array holding each pixel's own (row, column) coordinates.

    out[i, j, 0] == i and out[i, j, 1] == j.

    Example:
        coordinate_grid(2, 3)[..., 0] -> [[0, 0, 0],
                                          [1, 1, 1]]
        coordinate_grid(2, 3)[..., 1] -> [[0, 1, 2],
                                          [0, 1, 2]]
    """
    raise NotImplementedError


def repeat_and_cycle(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Two ways of repeating a 1-D array k times.

    Returns (each, cycled), both of shape (k * len(labels),) and the dtype of `labels`:
        each   - every element repeated k times in place
        cycled - the whole array repeated k times

    Example:
        repeat_and_cycle(np.array([0, 1, 2]), 2) -> (array([0, 0, 1, 1, 2, 2]), array([0, 1, 2, 0, 1, 2]))
    """
    raise NotImplementedError
