"""031 · matmul, dot, @ and einsum (reference solution)."""

import numpy as np


def batched_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    # x[..., None] makes one (m, 1) column per item, so @ batches over B.
    return (A @ x[..., None])[..., 0]


def bilinear(x: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", x, W, y)


def batch_trace(A: np.ndarray) -> np.ndarray:
    return np.einsum("bii->b", A)  # repeated i = walk the diagonal, missing from output = sum


def attention_scores(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    # swapaxes transposes only the matrix axes; K.T would also move the batch axis.
    return Q @ K.swapaxes(-1, -2) / np.sqrt(Q.shape[-1])
