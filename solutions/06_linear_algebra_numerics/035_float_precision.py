"""035 · Float precision: isclose, cancellation, logsumexp (reference solution)."""

import numpy as np


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    # An all -inf slice would give -inf - (-inf) = NaN; shifting by 0 there is harmless.
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):  # log(0) = -inf is the right answer for all -inf slices
        out = np.log(np.exp(x - m).sum(axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # exp(-|x|) <= 1 never overflows, and log1p keeps the digits when it's tiny.
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def quadratic_roots(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    sign = np.where(b >= 0, 1.0, -1.0)  # np.sign(0) would be 0
    # b and sign*sqrt(...) have the same sign, so this sum never cancels.
    q = -0.5 * (b + sign * np.sqrt(b * b - 4 * a * c))
    r1, r2 = q / a, c / q  # the second root from r1 * r2 = c / a
    return np.minimum(r1, r2), np.maximum(r1, r2)


def count_mismatches(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> int:
    return int(np.count_nonzero(~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)))
