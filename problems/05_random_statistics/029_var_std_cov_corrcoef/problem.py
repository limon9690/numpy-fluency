"""029 · var and std with ddof, cov, corrcoef."""

import numpy as np


def sample_std(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Sample standard deviation (divisor n - 1) of `x` along `axis` (None = all elements).

    Every reduced slice has at least 2 elements. Returns float64 (a NumPy scalar for axis=None).

    Example:
        sample_std(np.array([1.0, 3.0])) -> 1.4142...      (np.std gives 1.0)
    """
    raise NotImplementedError


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Sample covariance matrix of the columns of X.

    `X` has shape (n, d): n samples (rows), d variables (columns), n >= 2.
    Returns float64 of shape (d, d) with C[i, j] = sum_k (X[k,i] - mean_i)(X[k,j] - mean_j) / (n - 1).
    np.cov is banned.

    Example:
        covariance_matrix(np.array([[0., 0.], [2., 4.]])) -> [[2., 4.],
                                                             [4., 8.]]
    """
    raise NotImplementedError


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of the columns of X (shape (n, d), no constant columns).

    Returns float64 of shape (d, d): R[i, j] = C[i, j] / (std_i * std_j), ones on the diagonal.
    np.cov and np.corrcoef are banned.

    Example:
        correlation_matrix(np.array([[1., 2., 3.], [2., 4., 1.], [3., 6., 2.]]))[0, 1] -> 1.0
    """
    raise NotImplementedError


def columnwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlation between matching columns of two arrays.

    `x` and `y` both have shape (n, d) with no constant columns. Returns float64 of shape (d,)
    with out[j] = Pearson correlation of x[:, j] and y[:, j]. Don't build a full (2d, 2d) matrix.

    Example:
        columnwise_corr(np.array([[1., 1.], [2., 2.], [3., 3.]]),
                        np.array([[2., 3.], [4., 2.], [6., 1.]])) -> array([ 1., -1.])
    """
    raise NotImplementedError
