"""022 · unique and bincount (reference solution)."""

import numpy as np


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, codes = np.unique(labels, return_inverse=True)
    # NumPy 2 already returns the input's shape; the reshape keeps NumPy 1.x behaving too.
    return classes, codes.reshape(labels.shape)


def unique_rows_counts(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(a, axis=0, return_counts=True)


def most_frequent(x: np.ndarray) -> int:
    return int(np.bincount(x).argmax())  # argmax picks the first (smallest) value on ties


def group_mean(groups: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups)
    out = np.full(n_groups, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)  # skip empty groups: no 0/0 warning
    return out
