"""004 · Views vs copies."""

import numpy as np


def memory_relation(a: np.ndarray, b: np.ndarray) -> str:
    """Describe how two arrays relate in memory.

    Returns:
        "same"         if `a` and `b` are the very same Python object,
        "shared"       otherwise, if at least one element of `a` is stored in the same memory as
                       an element of `b` (so writing to one can change the other),
        "independent"  otherwise.

    Example:
        a = np.arange(6)
        memory_relation(a, a)          -> "same"
        memory_relation(a, a[1:3])     -> "shared"
        memory_relation(a, a.copy())   -> "independent"
        memory_relation(a[::2], a[1::2]) -> "independent"   (same buffer, no common element)
    """
    raise NotImplementedError


def first_column_copy(a: np.ndarray) -> np.ndarray:
    """Return column 0 of the 2-D array `a` (shape (n, m)) as a 1-D array of shape (n,).

    The result has the same dtype as `a` and owns its memory: modifying it must not change `a`.

    Example:
        first_column_copy(np.array([[1, 2], [3, 4]])) -> array([1, 3])
    """
    raise NotImplementedError


def standardize_inplace(a: np.ndarray) -> None:
    """Standardize a floating-point array in place: a <- (a - mean) / std. Returns None.

    - mean and std are over all elements; std is the population std (ddof=0).
    - The caller's array must change. This must also work when `a` is a view into a bigger
      array (e.g. one column), in which case only that part of the bigger array changes.
    - If every element is equal (std == 0), set every element to 0.
    - If `a` is not a floating-point array, raise TypeError.

    Example:
        x = np.array([1.0, 2.0, 3.0])
        standardize_inplace(x)   # x is now [-1.2247, 0., 1.2247]
    """
    raise NotImplementedError


def zero_small_in_rows(a: np.ndarray, rows: list[int], threshold: float) -> None:
    """In place: inside the rows listed in `rows`, set every value < threshold to 0. Returns None.

    `a` is 2-D. Rows not listed are left alone. Values >= threshold are left alone.

    Example:
        a = np.array([[-1, 5], [-2, 6], [-3, 7]])
        zero_small_in_rows(a, [0, 2], 0)
        # a is now [[0, 5], [-2, 6], [0, 7]]
    """
    raise NotImplementedError
