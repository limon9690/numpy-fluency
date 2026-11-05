"""036 · Images as arrays: HWC uint8 vs float, channels, grayscale."""

import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    """Convert an image to float32 in [0, 1].

    uint8 input (0..255) is divided by 255. Float input (any float dtype, assumed already in
    [0, 1]) is returned as float32 with the same values. Shape is unchanged. Returns a new array.

    Example:
        to_float(np.array([[0, 255]], np.uint8)) -> array([[0., 1.]], dtype=float32)
    """
    raise NotImplementedError


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image (nominally in [0, 1]) to uint8.

    Clip to [0, 1] first, multiply by 255, round to the nearest integer, cast to uint8.
    Shape is unchanged.

    Example:
        to_uint8(np.array([-0.2, 0.0, 0.5, 0.999, 1.3])) -> array([  0,   0, 128, 255, 255], dtype=uint8)
    """
    raise NotImplementedError


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale from RGB with the BT.601 weights: 0.299 R + 0.587 G + 0.114 B.

    `img` has shape (..., 3) (one image or a batch). Returns shape (...) without the channel axis.
    uint8 input gives uint8 output (rounded to nearest); float input keeps its float dtype.

    Example:
        rgb_to_gray(np.array([[[255, 0, 0], [255, 255, 255]]], np.uint8)) -> array([[ 76, 255]], dtype=uint8)
    """
    raise NotImplementedError


def alpha_blend(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend a foreground over a background: out = alpha * fg + (1 - alpha) * bg.

    `fg` and `bg` are uint8 images of shape (H, W, 3); `alpha` is a float array of shape (H, W)
    with values in [0, 1] (1 = only foreground). Returns uint8 of shape (H, W, 3), rounded to
    nearest. No overflow, no Python loops.

    Example:
        fg = np.full((1, 1, 3), 200, np.uint8); bg = np.full((1, 1, 3), 100, np.uint8)
        alpha_blend(fg, bg, np.array([[0.25]])) -> [[[125, 125, 125]]]
    """
    raise NotImplementedError
