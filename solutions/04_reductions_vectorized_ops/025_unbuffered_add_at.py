"""025 · np.add.at and scatter-style updates (reference solution)."""

import numpy as np


def scatter_add(target: np.ndarray, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = target.copy()
    np.add.at(out, idx, values)  # unbuffered: repeated indices each get their update
    return out


def pixel_counts(ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> np.ndarray:
    counts = np.zeros((h, w), dtype=np.int64)
    np.add.at(counts, (ys, xs), 1)
    return counts


def scatter_max(n: int, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(n, -np.inf)  # -inf is the identity for max, and marks empty slots
    np.maximum.at(out, idx, values)
    return out


def embedding_grad(indices: np.ndarray, grad_out: np.ndarray, vocab_size: int) -> np.ndarray:
    d = grad_out.shape[-1]
    grad = np.zeros((vocab_size, d), dtype=grad_out.dtype)
    # One flat list of (word id, gradient row) pairs; add.at sums rows for repeated words.
    np.add.at(grad, indices.ravel(), grad_out.reshape(-1, d))
    return grad
