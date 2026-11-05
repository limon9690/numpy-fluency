"""006 · Slicing and basic indexing (reference solution)."""

import numpy as np


def crop_center(img: np.ndarray, h: int, w: int) -> np.ndarray:
    top = (img.shape[0] - h) // 2
    left = (img.shape[1] - w) // 2
    return img[top:top + h, left:left + w]  # a trailing channel axis is kept as-is


def flip_axis(a: np.ndarray, axis: int) -> np.ndarray:
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(None, None, -1)  # list indexing handles negative axes for free
    return a[tuple(idx)]


def checkerboard(n: int) -> np.ndarray:
    board = np.zeros((n, n), dtype=np.int8)
    board[::2, 1::2] = 1
    board[1::2, ::2] = 1
    return board


def last_k(a: np.ndarray, k: int) -> np.ndarray:
    # a[-k:] would return everything for k == 0, so use a non-negative start.
    return a[len(a) - k:]
