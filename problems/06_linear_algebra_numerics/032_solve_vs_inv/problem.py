"""032 · solve vs inv, condition number."""

import numpy as np


def solve_many(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A X = B for X, for several right-hand sides at once.

    `A` has shape (n, n) and is non-singular; `B` has shape (n, k) (each column is one
    right-hand side). Returns X of shape (n, k). np.linalg.inv is banned.

    Example:
        solve_many(np.array([[3., 1.], [1., 2.]]), np.array([[9., 3.], [8., 1.]])) -> [[2., 1.],
                                                                                     [3., 0.]]
    """
    raise NotImplementedError


def batched_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve N independent systems A[i] x[i] = b[i].

    `A` has shape (N, n, n), `b` has shape (N, n) (one vector per system). Returns x of shape
    (N, n). Careful: in NumPy 2, np.linalg.solve reads a 2-D b as a matrix, not a stack of vectors.
    No Python loops; np.linalg.inv is banned.

    Example:
        A = np.stack([np.eye(2), 2 * np.eye(2)]); b = np.array([[1., 2.], [2., 4.]])
        batched_solve(A, b) -> [[1., 2.],
                                [1., 2.]]
    """
    raise NotImplementedError


def solve_checked(A: np.ndarray, b: np.ndarray, max_cond: float = 1e10) -> np.ndarray:
    """Solve A x = b, refusing badly conditioned systems.

    `A` is (n, n), `b` is (n,). If np.linalg.cond(A) > max_cond (this includes singular
    matrices, whose condition number is huge or inf), raise ValueError. Otherwise return x of
    shape (n,). np.linalg.inv is banned.

    Example:
        solve_checked(np.array([[1., 1.], [1., 1.]]), np.array([1., 1.])) -> ValueError
    """
    raise NotImplementedError


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Relative residual ||A x - b|| / ||b|| with the Euclidean (2-) norm, as a Python float.

    `A` is (n, n), `x` and `b` are (n,), b is not all zeros.

    Example:
        relative_residual(np.eye(2), np.array([1., 2.]), np.array([1., 2.])) -> 0.0
    """
    raise NotImplementedError
