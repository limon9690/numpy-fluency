"""005 · Strides and memory layout."""

import numpy as np


def strides_for(shape: tuple[int, ...], itemsize: int, order: str = "C") -> tuple[int, ...]:
    """Strides in bytes of a freshly allocated array with this shape, itemsize and order.

    `shape` is a tuple of positive ints (it may be empty for a 0-d array), `itemsize` is the size
    of one element in bytes, `order` is "C" (row-major) or "F" (column-major).
    Compute the answer with arithmetic; don't create an array to read its strides.

    Returns a tuple of Python ints with one entry per axis.

    Example:
        strides_for((2, 3), 4, "C") -> (12, 4)
        strides_for((2, 3), 4, "F") -> (4, 8)
    """
    raise NotImplementedError


def element_offset(a: np.ndarray, index: tuple[int, ...]) -> int:
    """Byte distance from the element a[0, 0, ..., 0] to the element a[index].

    `index` has one non-negative int per axis of `a`. Use `a.strides`; `a` can be any array or
    view (sliced, transposed, reversed), so the result may be negative.

    Returns a Python int.

    Example:
        a = np.zeros((2, 3), dtype=np.int32)   # strides (12, 4)
        element_offset(a, (1, 2))   -> 20
        element_offset(a.T, (2, 1)) -> 20      # the same element
    """
    raise NotImplementedError


def ensure_c_contiguous(a: np.ndarray) -> np.ndarray:
    """Return a C-contiguous array with the same values, shape and dtype as `a`.

    If `a` is already C-contiguous, don't copy: return `a` (or a view of it).
    Otherwise return a C-ordered copy. `a` itself is not modified.

    Example:
        x = np.arange(6).reshape(2, 3).T          # not C-contiguous
        ensure_c_contiguous(x).flags.c_contiguous -> True
    """
    raise NotImplementedError


def repeat_rows_view(row: np.ndarray, n: int) -> np.ndarray:
    """A read-only (n, len(row)) array whose every row equals the 1-D array `row`.

    No data may be copied: the result shares memory with `row` and has stride 0 along axis 0.
    Same dtype as `row`.

    Example:
        repeat_rows_view(np.array([1, 2, 3]), 2) -> [[1, 2, 3],
                                                     [1, 2, 3]]   (read-only, stride 0 on axis 0)
    """
    raise NotImplementedError
