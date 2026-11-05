"""037 · Sliding windows and framing (sliding_window_view) (reference solution)."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def frames(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    # Every window, then keep every hop-th one: still a view, nothing is copied.
    return sliding_window_view(signal, frame_len)[::hop]


def moving_max(x: np.ndarray, w: int) -> np.ndarray:
    return sliding_window_view(x, w).max(axis=-1)


def patches_2d(img: np.ndarray, k: int) -> np.ndarray:
    return sliding_window_view(img, (k, k))


def box_blur_valid(img: np.ndarray, k: int) -> np.ndarray:
    # dtype=float64 accumulates without overflow and without copying the windows first.
    return patches_2d(img, k).mean(axis=(-2, -1), dtype=np.float64)
