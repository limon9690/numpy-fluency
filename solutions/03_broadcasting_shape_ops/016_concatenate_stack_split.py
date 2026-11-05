"""016 · concatenate, stack, split (reference solution)."""

import numpy as np


def add_bias_column(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)  # default float64 would promote float32 X
    return np.concatenate([ones, X], axis=1)


def split_batches(x: np.ndarray, batch_size: int) -> list[np.ndarray]:
    # Cut points batch_size, 2*batch_size, ... (< len(x), so no empty last piece).
    return np.split(x, np.arange(batch_size, len(x), batch_size))


def interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pairs = np.stack([a, b], axis=1)  # (n, 2, ...): a[i] and b[i] next to each other
    return pairs.reshape(-1, *a.shape[1:])


def join_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([r, g, b], axis=-1)
