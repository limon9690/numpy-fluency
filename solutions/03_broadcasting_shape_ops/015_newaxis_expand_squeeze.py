"""015 · newaxis, expand_dims, squeeze (reference solution)."""

import numpy as np


def ensure_batch_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[None, :, :, None]
    if img.ndim == 3:
        return img[None]
    if img.ndim == 4:
        return img
    raise ValueError(f"expected an image with 2, 3 or 4 dimensions, got {img.ndim}")


def squeeze_except(x: np.ndarray, axis: int) -> np.ndarray:
    keep = axis % x.ndim
    drop = tuple(i for i, n in enumerate(x.shape) if n == 1 and i != keep)
    return np.squeeze(x, axis=drop)  # explicit axes, so the kept axis survives even with length 1


def broadcast_along(v: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = len(v)  # list indexing accepts negative axes
    return v.reshape(shape)
