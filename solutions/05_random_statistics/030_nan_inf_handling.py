"""030 · NaN and inf: isfinite, nan* functions, errstate (reference solution)."""

import numpy as np


def nan_column_means(X: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(X)
    sums = np.where(valid, X, 0.0).sum(axis=0)
    counts = valid.sum(axis=0)
    out = np.full(X.shape[1], np.nan)
    # Skip all-NaN columns instead of computing 0/0 (np.nanmean would warn here).
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def fill_nan_with_column_mean(X: np.ndarray) -> np.ndarray:
    means = nan_column_means(X)
    means = np.where(np.isnan(means), 0.0, means)
    return np.where(np.isnan(X), means, X)  # (d,) means broadcast down the rows


def finite_stats(x: np.ndarray) -> dict:
    vals = x[np.isfinite(x)]  # drops nan, inf and -inf in one mask
    if vals.size == 0:
        raise ValueError("no finite values")
    return {"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean()), "count": int(vals.size)}


def relative_change(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    old = np.asarray(old, dtype=np.float64)
    # inf/nan are the intended results for old == 0; only silence the warnings, only here.
    with np.errstate(divide="ignore", invalid="ignore"):
        return (new - old) / old
