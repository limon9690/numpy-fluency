"""004 · Views vs copies (reference solution)."""

import numpy as np


def memory_relation(a: np.ndarray, b: np.ndarray) -> str:
    if a is b:
        return "same"
    # shares_memory checks real element overlap; may_share_memory only compares bounds.
    if np.shares_memory(a, b):
        return "shared"
    return "independent"


def first_column_copy(a: np.ndarray) -> np.ndarray:
    return a[:, 0].copy()  # a[:, 0] alone is a view into a


def standardize_inplace(a: np.ndarray) -> None:
    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(f"in-place standardization needs a float array, got {a.dtype}")
    # Augmented assignment writes into a's memory, so the caller (and any base array) sees it.
    a -= a.mean()
    std = a.std()
    if std > 0:
        a /= std


def zero_small_in_rows(a: np.ndarray, rows: list[int], threshold: float) -> None:
    sub = a[rows]                  # fancy indexing: this is a copy
    sub[sub < threshold] = 0
    a[rows] = sub                  # one assignment writes the result back into a
