"""002 · dtypes, casting and overflow (reference solution)."""

import numpy as np


def brighten(img: np.ndarray, amount: int) -> np.ndarray:
    # int32 has room for 255 + any reasonable amount, so nothing wraps before the clip.
    wide = img.astype(np.int32) + amount
    return np.clip(wide, 0, 255).astype(np.uint8)


def round_half_up(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Work on |x| so "half up" means "half away from zero" for negatives too.
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def to_int_safe(x: np.ndarray, dtype) -> np.ndarray:
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError("cannot convert NaN or inf to an integer")
    rounded = np.round(x)
    info = np.iinfo(dtype)
    # Check before casting: an out-of-range float -> int cast gives garbage, not an error.
    if rounded.size and (rounded.min() < info.min or rounded.max() > info.max):
        raise OverflowError(f"values don't fit in {np.dtype(dtype)} [{info.min}, {info.max}]")
    return rounded.astype(dtype)


def scale_keep_dtype(a: np.ndarray, factor: float) -> np.ndarray:
    # dtype= forces the computation and the output to a's dtype, even for a strong float64 scalar.
    return np.multiply(a, factor, dtype=a.dtype)
