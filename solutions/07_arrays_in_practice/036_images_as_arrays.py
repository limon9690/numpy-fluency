"""036 · Images as arrays: HWC uint8 vs float, channels, grayscale (reference solution)."""

import numpy as np

_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255
    return img.astype(np.float32)  # already 0..1: only the dtype changes


def to_uint8(img: np.ndarray) -> np.ndarray:
    # Clip first: casting out-of-range floats to uint8 wraps or is undefined.
    return np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        gray = img.astype(np.float64) @ _GRAY_WEIGHTS  # @ contracts the channel axis per pixel
        return np.round(gray).astype(np.uint8)
    return img @ _GRAY_WEIGHTS.astype(img.dtype)


def alpha_blend(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]  # (H, W, 1) broadcasts over the 3 channels
    out = a * fg.astype(np.float64) + (1 - a) * bg
    return np.round(out).astype(np.uint8)
