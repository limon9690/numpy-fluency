"""008 · Boolean masking (reference solution)."""

import numpy as np


def count_in_range(x: np.ndarray, low: float, high: float) -> int:
    # Parentheses matter: & binds tighter than the comparisons.
    return int(np.count_nonzero((x >= low) & (x <= high)))


def replace_outside(x: np.ndarray, low: float, high: float, fill: float) -> np.ndarray:
    out = x.copy()
    out[(x < low) | (x > high)] = fill  # masked assignment is in place, so do it on the copy
    return out


def select_rows(data: np.ndarray, col: int, threshold: float) -> np.ndarray:
    return data[data[:, col] > threshold]  # a 1-D mask on axis 0 keeps whole rows


def drop_nan_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = ~(np.isnan(x) | np.isnan(y))
    return x[keep], y[keep]
