"""018 · Reductions with axis and keepdims (reference solution)."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    # Shifting by the max makes the largest exponent exp(0) = 1, so nothing overflows.
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def argmax_2d(a: np.ndarray) -> tuple[int, int]:
    r, c = np.unravel_index(a.argmax(), a.shape)  # argmax without axis is a flat index
    return int(r), int(c)


def channel_mean_std(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # dtype=float64 accumulates in float64 without first copying the whole batch to float.
    mean = images.mean(axis=(0, 1, 2), dtype=np.float64)
    std = images.std(axis=(0, 1, 2), dtype=np.float64)
    return mean, std


def center_per_sample(x: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, x.ndim))  # every axis except the batch axis
    x = x.astype(np.float64)
    return x - x.mean(axis=axes, keepdims=True)
