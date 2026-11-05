"""028 · Percentiles, clip and winsorize (reference solution)."""

import numpy as np


def iqr_outlier_mask(x: np.ndarray) -> np.ndarray:
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)


def winsorize(x: np.ndarray, lower_pct: float, upper_pct: float, axis: int | None = None) -> np.ndarray:
    # keepdims makes both bounds broadcast against x, whatever the axis.
    lo, hi = np.percentile(x, [lower_pct, upper_pct], axis=axis, keepdims=True)
    return np.clip(x, lo, hi)


def robust_scale(X: np.ndarray) -> np.ndarray:
    q1, med, q3 = np.percentile(X, [25, 50, 75], axis=0, keepdims=True)
    iqr = q3 - q1
    return (X - med) / np.where(iqr == 0, 1.0, iqr)


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # Only the inner edges decide the bin; side="right" puts a value equal to an edge in the upper bin,
    # and the maximum (== last edge) lands in bin n_bins - 1 automatically.
    return np.searchsorted(edges[1:-1], x, side="right")
