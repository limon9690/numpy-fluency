"""034 · Norms, SVD and eigh (batched) (reference solution)."""

import numpy as np


def row_norms(X: np.ndarray, ord=2) -> np.ndarray:
    return np.linalg.norm(X, ord=ord, axis=1)  # axis=1: vector norm per row, not a matrix norm


def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    # Scaling U's kept columns by s avoids building diag(s); [..., None, :] keeps it batched.
    return (U[..., :, :k] * s[..., None, :k]) @ Vh[..., :k, :]


def top_eigenpair(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(S)
    # Ascending order: the largest eigenvalue is last, its eigenvector is the last COLUMN.
    return w[:, -1], V[:, :, -1]


def psd_sqrt(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    root = np.sqrt(np.clip(w, 0, None))  # rounding can push zero eigenvalues just below 0
    return (V * root[..., None, :]) @ np.swapaxes(V, -1, -2)
