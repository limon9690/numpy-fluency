"""005 · Strides and memory layout (reference solution)."""

import numpy as np


def strides_for(shape: tuple[int, ...], itemsize: int, order: str = "C") -> tuple[int, ...]:
    # The fastest-changing axis (last for C, first for F) has stride itemsize;
    # each next axis jumps over one full block of the previous ones.
    axes = range(len(shape) - 1, -1, -1) if order == "C" else range(len(shape))
    strides = [0] * len(shape)
    step = itemsize
    for k in axes:
        strides[k] = step
        step *= shape[k]
    return tuple(strides)


def element_offset(a: np.ndarray, index: tuple[int, ...]) -> int:
    return sum(i * s for i, s in zip(index, a.strides))


def ensure_c_contiguous(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a)  # returns `a` itself when no copy is needed


def repeat_rows_view(row: np.ndarray, n: int) -> np.ndarray:
    # broadcast_to fakes the repeat with a 0 stride and marks the result read-only.
    return np.broadcast_to(row, (n, row.shape[0]))
