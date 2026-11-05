"""011 · take_along_axis and put_along_axis (reference solution)."""

import numpy as np


def topk_last(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # Reverse an ascending argsort instead of sorting -x (which wraps for unsigned ints).
    indices = np.argsort(x, axis=-1)[..., ::-1][..., :k]
    return np.take_along_axis(x, indices, axis=-1), indices


def pick_along(x: np.ndarray, idx: np.ndarray, axis: int) -> np.ndarray:
    # take_along_axis wants idx with the same ndim as x: give it a length-1 axis, then drop it.
    picked = np.take_along_axis(x, np.expand_dims(idx, axis), axis=axis)
    return np.squeeze(picked, axis=axis)


def keep_topk(x: np.ndarray, k: int) -> np.ndarray:
    _, indices = topk_last(x, k)
    out = np.zeros_like(x)
    np.put_along_axis(out, indices, np.take_along_axis(x, indices, axis=-1), axis=-1)
    return out


def argmax_mask(x: np.ndarray, axis: int) -> np.ndarray:
    best = np.argmax(x, axis=axis, keepdims=True)
    mask = np.zeros(x.shape, dtype=bool)
    np.put_along_axis(mask, best, True, axis=axis)
    return mask
