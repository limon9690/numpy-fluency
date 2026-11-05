"""012 · Broadcasting rules (reference solution)."""

import numpy as np


def broadcast_shape(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    n = max(len(shape1), len(shape2))
    # Missing leading axes count as length 1.
    s1 = (1,) * (n - len(shape1)) + tuple(shape1)
    s2 = (1,) * (n - len(shape2)) + tuple(shape2)
    out = []
    for a, b in zip(s1, s2):
        if a == b or b == 1:
            out.append(a)
        elif a == 1:
            out.append(b)
        else:
            raise ValueError(f"shapes {shape1} and {shape2} can't be broadcast ({a} vs {b})")
    return tuple(out)


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, None] + b[None, :]  # (n, 1) + (1, m) -> (n, m)


def pairwise_diff(x: np.ndarray) -> np.ndarray:
    return x[:, None, :] - x[None, :, :]  # (n, 1, d) - (1, n, d) -> (n, n, d)


def pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :].astype(np.float64) - y[None, :, :]
    return np.sum(diff**2, axis=-1)
