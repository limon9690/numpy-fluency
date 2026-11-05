"""035 · Float precision: isclose, cancellation, logsumexp."""

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log(sum(exp(x))) along `axis`.

    Must work for very large and very small values (e.g. +-1000) without overflow, underflow to
    -inf, or warnings. A slice whose values are all -inf gives -inf. The reduced axis is removed.
    Returns float64.

    Example:
        logsumexp(np.array([[1000., 1000.], [-1000., -1000.]]), axis=1) -> array([1000.6931, -999.3069])
    """
    raise NotImplementedError


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus, log(1 + exp(x)), element-wise.

    Must be accurate for large positive x (softplus(1000) == 1000, no overflow) and for very
    negative x (softplus(-40) ≈ exp(-40) ≈ 4.25e-18, not 0). No warnings. Returns float64.

    Example:
        softplus(np.array([0.0])) -> array([0.6931])
    """
    raise NotImplementedError


def quadratic_roots(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both real roots of a*x**2 + b*x + c = 0, computed without catastrophic cancellation.

    a, b, c are arrays (or scalars) that broadcast together, with a != 0, b**2 - 4ac >= 0 and
    never b == c == 0. Returns (smaller_root, larger_root) as float64 arrays. Both roots must
    have small RELATIVE error, even when one is many orders of magnitude smaller than the other.

    Example:
        quadratic_roots(np.array([1.]), np.array([-3.]), np.array([2.])) -> (array([1.]), array([2.]))
    """
    raise NotImplementedError


def count_mismatches(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> int:
    """Number of positions where `a` is NOT close to the reference `b`.

    "Close" means |a - b| <= atol + rtol * |b| (np.isclose's rule). Two NaNs at the same position
    count as equal. Inputs broadcast. Returns a Python int.

    Example:
        count_mismatches(np.array([1.0, 2.0, np.nan, 0.0]), np.array([1.0 + 1e-9, 2.1, np.nan, 1e-9])) -> 1
    """
    raise NotImplementedError
