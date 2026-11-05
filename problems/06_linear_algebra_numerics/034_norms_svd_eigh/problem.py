"""034 · Norms, SVD and eigh (batched)."""

import numpy as np


def row_norms(X: np.ndarray, ord=2) -> np.ndarray:
    """The vector norm of every row of the 2-D array X.

    `ord` is 1 (sum of |x|), 2 (Euclidean) or np.inf (max |x|). Returns float64 of shape
    (X.shape[0],).

    Example:
        row_norms(np.array([[3., 4.], [1., -1.]]), 2)      -> array([5.    , 1.4142])
        row_norms(np.array([[3., 4.], [1., -1.]]), np.inf) -> array([4., 1.])
    """
    raise NotImplementedError


def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation of A (in the Frobenius norm), via the SVD.

    `A` has shape (..., m, n): one matrix, or a batch of matrices. 1 <= k <= min(m, n).
    Keep the k largest singular values of each matrix. Returns the same shape as A.

    Example:
        low_rank_approx(np.diag([3., 2., 1.]), 2) -> np.diag([3., 2., 0.])
    """
    raise NotImplementedError


def top_eigenpair(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Largest eigenvalue and a matching unit eigenvector of every symmetric matrix in a batch.

    `S` has shape (B, n, n), each S[b] symmetric. Returns (values, vectors) with values of shape
    (B,) and vectors of shape (B, n), where S[b] @ vectors[b] ≈ values[b] * vectors[b] and
    ||vectors[b]|| = 1. (The sign of each vector is not fixed.)

    Example:
        top_eigenpair(np.array([[[2., 0.], [0., 5.]]])) -> (array([5.]), array([[0., 1.]]))   # or [[0., -1.]]
    """
    raise NotImplementedError


def psd_sqrt(S: np.ndarray) -> np.ndarray:
    """Symmetric square root of positive semi-definite matrices.

    `S` has shape (..., n, n), symmetric PSD. Returns R of the same shape, symmetric, with
    R @ R ≈ S and no NaN: eigenvalues that come out slightly negative from rounding are
    clipped to 0.

    Example:
        psd_sqrt(np.array([[4., 0.], [0., 9.]])) -> [[2., 0.],
                                                    [0., 3.]]
    """
    raise NotImplementedError
