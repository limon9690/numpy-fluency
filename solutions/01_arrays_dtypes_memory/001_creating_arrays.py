"""001 · Creating arrays (reference solution)."""

import numpy as np


def safe_arange(start: float, stop: float, step: float) -> np.ndarray:
    # Fix the count up front; arange's "keep adding step while < stop" is what goes wrong.
    n = max(int(round((stop - start) / step)), 0)
    return start + step * np.arange(n, dtype=np.float64)


def linspace_with_step(start: float, stop: float, n: int) -> tuple[np.ndarray, float]:
    points, step = np.linspace(start, stop, n, retstep=True)
    return points, float(step)


def blank_like(a: np.ndarray, fill: float) -> np.ndarray:
    # Without dtype=, full_like would copy a's dtype and truncate 0.5 to 0 for int arrays.
    return np.full_like(a, fill, dtype=np.float64)


def as_float32_matrix(rows: list[list[float]]) -> np.ndarray:
    out = np.array(rows, dtype=np.float32)  # NumPy 2 raises ValueError for ragged rows
    if out.ndim != 2:
        raise ValueError(f"expected a 2-D table of numbers, got {out.ndim}-D")
    return out
