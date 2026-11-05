"""023 · searchsorted, digitize, histogram, interp (reference solution)."""

import numpy as np


def bucketize(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # side="right": a value equal to edges[i] lands after it, i.e. in bin i.
    return np.searchsorted(edges, x, side="right") - 1


def count_in_windows(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    # "left" on both ends: start is included, end is excluded.
    return np.searchsorted(times, ends, side="left") - np.searchsorted(times, starts, side="left")


def histogram_counts(x: np.ndarray, n_bins: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(lo, hi, n_bins + 1)
    x = np.ravel(x)
    x = x[(x >= lo) & (x <= hi)]
    idx = bucketize(x, edges)
    idx[x == hi] = n_bins - 1  # the last bin is closed on the right
    return np.bincount(idx, minlength=n_bins), edges


def resample_linear(t: np.ndarray, y: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    # np.interp doesn't check its input; unsorted t would give silent garbage.
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    return np.interp(t_new, t, y)
