"""040 · Performance: vectorized vs loops, fake vectorization, memory blow-ups (reference solution)."""

import numpy as np


def pairwise_sq_dist_fast(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # |x - y|^2 = |x|^2 + |y|^2 - 2 x.y: only (n, m)-sized arrays, the heavy part is one matmul.
    D = (X * X).sum(axis=1)[:, None] + (Y * Y).sum(axis=1)[None, :] - 2.0 * (X @ Y.T)
    np.maximum(D, 0.0, out=D)  # in place: no extra (n, m) temporary
    return D


def standardize_inplace_f32(x: np.ndarray) -> None:
    x -= x.mean()
    flat = x.ravel()  # a view, because x is C-contiguous
    # dot(flat, flat) is the sum of squares with no temporary (x.std() would allocate one).
    std = np.sqrt(np.dot(flat, flat) / x.size)
    x /= std


def lookup(keys: np.ndarray, table_keys: np.ndarray, table_values: np.ndarray, default) -> np.ndarray:
    order = np.argsort(table_keys)
    pos = np.searchsorted(table_keys, keys, sorter=order)
    pos = np.clip(pos, 0, len(table_keys) - 1)  # keys past the end would index out of range
    idx = order[pos]  # back from sorted positions to table rows
    found = table_keys[idx] == keys
    return np.where(found, table_values[idx], default).astype(table_values.dtype, copy=False)


def channel_means_uint8(imgs: np.ndarray) -> np.ndarray:
    # dtype= makes the reduction accumulate in float64 chunk by chunk, with no full float copy.
    return imgs.mean(axis=(0, 1, 2), dtype=np.float64)
