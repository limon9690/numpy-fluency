"""032 · solve vs inv, condition number (reference solution)."""

import numpy as np


def solve_many(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, B)  # one LU factorization, reused for every column of B


def batched_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # NumPy 2 treats b as vectors only when it's 1-D, so pass explicit (n, 1) columns.
    return np.linalg.solve(A, b[..., None])[..., 0]


def solve_checked(A: np.ndarray, b: np.ndarray, max_cond: float = 1e10) -> np.ndarray:
    cond = np.linalg.cond(A)
    # `not cond <= max_cond` also catches a NaN condition number.
    if not cond <= max_cond:
        raise ValueError(f"matrix is too ill-conditioned (cond = {cond:.3g})")
    return np.linalg.solve(A, b)


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
