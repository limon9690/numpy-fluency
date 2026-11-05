"""029 · var and std with ddof, cov, corrcoef (reference solution)."""

import numpy as np


def sample_std(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.std(x, axis=axis, ddof=1)


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)  # center each column (variable)
    return Xc.T @ Xc / (X.shape[0] - 1)


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    C = covariance_matrix(X)
    s = np.sqrt(np.diag(C))
    # The (n - 1) in C cancels against the one hidden inside each s.
    return C / np.outer(s, s)


def columnwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    zx = (x - x.mean(axis=0)) / x.std(axis=0)
    zy = (y - y.mean(axis=0)) / y.std(axis=0)
    return (zx * zy).mean(axis=0)  # ddof=0 everywhere, so the divisors match
