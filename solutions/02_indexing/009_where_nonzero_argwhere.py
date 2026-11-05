"""009 · where, nonzero, argwhere (reference solution)."""

import numpy as np


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    zero = b == 0
    # where() evaluates both branches, so the division itself must never see a zero.
    quotient = a / np.where(zero, 1.0, b)
    return np.where(zero, fill, quotient)


def bounding_box(mask: np.ndarray) -> tuple[slice, ...] | None:
    idx = np.argwhere(mask)  # (k, ndim): one row per True element
    if len(idx) == 0:
        return None
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1  # slice stops are exclusive
    return tuple(slice(int(s), int(e)) for s, e in zip(lo, hi))


def first_true(mask: np.ndarray) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else -1


def piecewise_curve(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < 0, x**2, np.where(x < 1, x, 1.0))
