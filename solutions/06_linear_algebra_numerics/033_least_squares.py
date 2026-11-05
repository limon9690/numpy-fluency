"""033 · Least squares with lstsq (reference solution)."""

import numpy as np


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.column_stack([x, np.ones_like(x, dtype=np.float64)])  # ones column = intercept
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def fit_plane(points: np.ndarray) -> np.ndarray:
    x, y, z = points.T
    A = np.column_stack([x, y, np.ones_like(x, dtype=np.float64)])
    return np.linalg.lstsq(A, z, rcond=None)[0]


def lstsq_multi(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.linalg.lstsq(A, B, rcond=None)[0]
    # lstsq's residuals are empty for rank-deficient A, so compute them directly.
    rss = ((A @ X - B) ** 2).sum(axis=0)
    return X, rss


def min_norm_solution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(A, b, rcond=None)[0]  # SVD-based: picks the minimum-norm solution
