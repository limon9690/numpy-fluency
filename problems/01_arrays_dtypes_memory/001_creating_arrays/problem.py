"""001 · Creating arrays."""

import numpy as np


def safe_arange(start: float, stop: float, step: float) -> np.ndarray:
    """Float range from `start` toward `stop` (excluded) with a fixed, predictable count.

    The number of points is n = round((stop - start) / step); if n <= 0 the result is empty.
    The points are start, start + step, ..., start + (n - 1) * step.

    Returns a 1-D float64 array of length n.

    Example:
        safe_arange(1, 1.3, 0.1)  -> [1.0, 1.1, 1.2]       (np.arange gives 4 points here)
        safe_arange(1, 0, -0.25)  -> [1.0, 0.75, 0.5, 0.25]
    """
    raise NotImplementedError


def linspace_with_step(start: float, stop: float, n: int) -> tuple[np.ndarray, float]:
    """`n` (>= 2) evenly spaced float64 points from `start` to `stop`, both included.

    Returns (points, step) where points has shape (n,), dtype float64, points[0] == start and
    points[-1] == stop exactly, and step is the spacing as a Python float.

    Example:
        linspace_with_step(0, 1, 5) -> (array([0., 0.25, 0.5, 0.75, 1.]), 0.25)
    """
    raise NotImplementedError


def blank_like(a: np.ndarray, fill: float) -> np.ndarray:
    """A new float64 array with the same shape as `a`, every element equal to `fill`.

    The dtype is always float64, even if `a` is an integer array. `a` is not modified and the
    result does not share memory with it.

    Example:
        blank_like(np.array([[1, 2, 3]]), 0.5) -> array([[0.5, 0.5, 0.5]])
    """
    raise NotImplementedError


def as_float32_matrix(rows: list[list[float]]) -> np.ndarray:
    """Convert a list of equal-length lists into a 2-D float32 array.

    Raises ValueError if the rows have different lengths, or if the result is not 2-D
    (e.g. a flat list of numbers, or a list of lists of lists).

    Example:
        as_float32_matrix([[1, 2], [3, 4]]) -> float32 array of shape (2, 2)
        as_float32_matrix([[1, 2], [3]])    -> ValueError
    """
    raise NotImplementedError
