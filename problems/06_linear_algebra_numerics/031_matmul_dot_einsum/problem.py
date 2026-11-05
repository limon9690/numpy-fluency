"""031 · matmul, dot, @ and einsum."""

import numpy as np


def batched_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One matrix-vector product per batch item: out[b] = A[b] @ x[b].

    `A` has shape (B, n, m), `x` has shape (B, m). Returns shape (B, n).

    Example:
        A = np.stack([np.eye(2), 2 * np.eye(2)]); x = np.array([[1., 2.], [3., 4.]])
        batched_matvec(A, x) -> [[1., 2.],
                                 [6., 8.]]
    """
    raise NotImplementedError


def bilinear(x: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear form for every batch item: out[b] = x[b] @ W @ y[b].

    `x` has shape (B, n), `W` has shape (n, m), `y` has shape (B, m). Returns shape (B,).

    Example:
        bilinear(np.array([[1., 0.]]), np.array([[2., 3.], [4., 5.]]), np.array([[0., 1.]])) -> array([3.])
    """
    raise NotImplementedError


def batch_trace(A: np.ndarray) -> np.ndarray:
    """Trace (sum of the diagonal) of every matrix in a batch.

    `A` has shape (B, n, n). Returns shape (B,), same dtype. np.trace is banned.

    Example:
        batch_trace(np.stack([np.eye(3), np.arange(9.).reshape(3, 3)])) -> array([ 3., 12.])
    """
    raise NotImplementedError


def attention_scores(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention scores: Q K^T / sqrt(d) for every batch item.

    `Q` has shape (B, T, d), `K` has shape (B, S, d). Returns shape (B, T, S) with
    out[b, t, s] = (Q[b, t] . K[b, s]) / sqrt(d).

    Example:
        Q = np.ones((1, 2, 4)); K = np.ones((1, 3, 4))
        attention_scores(Q, K) -> array of shape (1, 2, 3), every entry 4 / 2 = 2.0
    """
    raise NotImplementedError
