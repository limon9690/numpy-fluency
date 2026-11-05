"""013 · Row and column scaling and normalization (reference solution)."""

import numpy as np


def row_normalize(a: np.ndarray) -> np.ndarray:
    sums = a.sum(axis=1, keepdims=True)  # (n, 1) lines up with rows
    # All-zero rows have sum 0; dividing their zeros by 1 keeps them zero without a warning.
    return a / np.where(sums == 0, 1, sums)


def standardize_columns(a: np.ndarray) -> np.ndarray:
    mean = a.mean(axis=0, keepdims=True)
    std = a.std(axis=0, keepdims=True)
    return (a - mean) / np.where(std == 0, 1, std)


def minmax_scale_columns(a: np.ndarray) -> np.ndarray:
    lo = a.min(axis=0, keepdims=True)
    span = a.max(axis=0, keepdims=True) - lo
    return (a - lo) / np.where(span == 0, 1, span)


def scale_rows(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    return a * w[:, None]  # (n,) alone would line up with the columns
