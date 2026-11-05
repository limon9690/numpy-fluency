"""019 · ufuncs: out=, where=, reduce, accumulate, outer (reference solution)."""

import numpy as np


def safe_log(x: np.ndarray, fill: float = -np.inf) -> np.ndarray:
    # Prefill: positions where `where` is False are left untouched.
    out = np.full(np.shape(x), fill, dtype=np.float64)
    np.log(x, out=out, where=x > 0)  # NaN > 0 is False, so NaN gets the fill too
    return out


def running_max(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.maximum.accumulate(x, axis=axis)


def gcd_all(x: np.ndarray) -> int:
    return int(np.gcd.reduce(np.ravel(x)))


def pairs_within(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    return np.abs(np.subtract.outer(a, b)) <= tol
