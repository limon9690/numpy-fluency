"""002 · dtypes, casting and overflow."""

import numpy as np


def brighten(img: np.ndarray, amount: int) -> np.ndarray:
    """Add `amount` to every pixel of a uint8 array, saturating instead of wrapping.

    `img` is a uint8 array of any shape. `amount` is a Python int, possibly negative.
    Results above 255 become 255 and results below 0 become 0.

    Returns a new uint8 array with the same shape. `img` is not modified.

    Example:
        brighten(np.array([250, 100, 3], np.uint8), 10)  -> array([255, 110, 13], dtype=uint8)
        brighten(np.array([250, 100, 3], np.uint8), -5)  -> array([245,  95,  0], dtype=uint8)
    """
    raise NotImplementedError


def round_half_up(x: np.ndarray) -> np.ndarray:
    """Round to the nearest integer, with halves rounded away from zero ("school rounding").

    `x` is a float array of any shape. Returns a float64 array of the same shape.
    (np.round rounds halves to even, so it gives a different answer for 0.5, 2.5, -2.5, ...)

    Example:
        round_half_up(np.array([0.5, 1.5, 2.5, -2.5, 2.4])) -> array([ 1.,  2.,  3., -3.,  2.])
    """
    raise NotImplementedError


def to_int_safe(x: np.ndarray, dtype) -> np.ndarray:
    """Round a float array with np.round (half to even) and cast it to the integer `dtype`.

    - If any value is NaN or infinite, raise ValueError.
    - If any rounded value is outside the range of `dtype`, raise OverflowError.

    Returns an array of dtype `dtype` with the same shape as `x`.

    Example:
        to_int_safe(np.array([1.6, 2.5, -1.7]), np.int8) -> array([ 2,  2, -2], dtype=int8)
        to_int_safe(np.array([127.6]), np.int8)          -> OverflowError (rounds to 128)
    """
    raise NotImplementedError


def scale_keep_dtype(a: np.ndarray, factor: float) -> np.ndarray:
    """Return `a * factor` with the same dtype as `a`.

    `a` is a floating-point array (float16, float32 or float64). `factor` may be a Python float
    or a NumPy scalar such as np.float64(0.1); either way the result keeps `a.dtype`.
    `a` is not modified.

    Example:
        scale_keep_dtype(np.ones(3, np.float32), np.float64(0.5)).dtype -> float32
    """
    raise NotImplementedError
