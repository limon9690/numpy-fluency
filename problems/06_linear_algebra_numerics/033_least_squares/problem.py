"""033 · Least squares with lstsq."""

import numpy as np


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares line y ≈ slope * x + intercept through 1-D points (x, y).

    Returns (slope, intercept) as Python floats. np.polyfit is banned.

    Example:
        fit_line(np.array([0., 1., 2.]), np.array([1., 3., 5.])) -> (2.0, 1.0)
    """
    raise NotImplementedError


def fit_plane(points: np.ndarray) -> np.ndarray:
    """Least-squares plane z ≈ a*x + b*y + c through 3-D points.

    `points` has shape (n, 3), columns x, y, z, n >= 3 and not all on one line.
    Returns float64 array([a, b, c]).

    Example:
        pts = np.array([[0., 0., 1.], [1., 0., 3.], [0., 1., 4.], [1., 1., 6.]])
        fit_plane(pts) -> array([2., 3., 1.])
    """
    raise NotImplementedError


def lstsq_multi(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of every column of B at once: minimize ||A X - B|| column by column.

    `A` has shape (n, d), `B` has shape (n, k). Returns (X, rss): X of shape (d, k) and the
    residual sum of squares of every column, rss[j] = sum_i (A @ X - B)[i, j]**2, shape (k,).
    rss must always be filled (np.linalg.lstsq leaves its own residuals empty in some cases).

    Example:
        A = np.array([[1., 0.], [0., 1.], [1., 1.]]); B = np.array([[1.], [1.], [2.]])
        lstsq_multi(A, B) -> (array([[1.], [1.]]), array([0.]))
    """
    raise NotImplementedError


def min_norm_solution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The minimum-norm exact solution of an underdetermined system A x = b.

    `A` has shape (m, n) with m < n and full row rank; `b` has shape (m,). Among all x with
    A x = b, return the one with the smallest ||x||. Shape (n,). np.linalg.inv is banned.

    Example:
        min_norm_solution(np.array([[1., 1.]]), np.array([2.])) -> array([1., 1.])
    """
    raise NotImplementedError
