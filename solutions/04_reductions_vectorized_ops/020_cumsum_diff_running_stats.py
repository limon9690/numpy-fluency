"""020 · cumsum, diff and running statistics (reference solution)."""

import numpy as np


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    # Leading 0 so every window sum is c[i + w] - c[i]; float64 avoids int overflow and drift.
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[w:] - c[:-w]) / w


def running_mean(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x, dtype=np.float64) / np.arange(1, len(x) + 1)


def lengths_to_offsets(lengths: np.ndarray) -> np.ndarray:
    out = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=out[1:])
    return out


def rising_edges(signal: np.ndarray) -> np.ndarray:
    # diff on bools would be XOR; on int8 the sign tells on (+1) from off (-1).
    d = np.diff(signal.astype(np.int8))
    return np.flatnonzero(d == 1) + 1
