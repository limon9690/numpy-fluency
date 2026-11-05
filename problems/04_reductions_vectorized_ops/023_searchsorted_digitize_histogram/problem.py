"""023 · searchsorted, digitize, histogram, interp."""

import numpy as np


def bucketize(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index of every value of `x` for sorted, strictly increasing bin `edges`.

    For each value v: the i with edges[i] <= v < edges[i+1]; -1 if v < edges[0];
    len(edges) - 1 if v >= edges[-1]. Returns an int array with the shape of `x`.

    Example:
        bucketize(np.array([-5, 0, 9.9, 10, 25]), np.array([0, 10, 20])) -> array([-1, 0, 0, 1, 2])
    """
    raise NotImplementedError


def count_in_windows(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """How many event times fall into each half-open window [starts[k], ends[k]).

    `times` is a sorted 1-D array. `starts` and `ends` have the same shape (k,) with
    starts <= ends. Returns an int array of shape (k,).

    Example:
        count_in_windows(np.array([1, 2, 2, 5, 9]), np.array([0, 2, 6]), np.array([2, 6, 9]))
        -> array([1, 3, 0])
    """
    raise NotImplementedError


def histogram_counts(x: np.ndarray, n_bins: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width histogram of `x` over [lo, hi], like np.histogram(x, n_bins, range=(lo, hi)).

    edges: float64 array of n_bins + 1 evenly spaced edges from lo to hi.
    counts: int array of shape (n_bins,). Bins are [edges[i], edges[i+1]) except the last,
    which also includes hi. Values outside [lo, hi] are ignored. np.histogram is banned.

    Returns (counts, edges).

    Example:
        histogram_counts(np.array([0.1, 0.4, 0.5, 1.0, 3.0]), 2, 0.0, 1.0)
        -> (array([2, 2]), array([0. , 0.5, 1. ]))
    """
    raise NotImplementedError


def resample_linear(t: np.ndarray, y: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Linearly interpolate samples y(t) at the new times `t_new`.

    `t` and `y` are 1-D of the same length. Outside [t[0], t[-1]] use the nearest end value.
    Raise ValueError if `t` is not strictly increasing. Returns float64 with the shape of `t_new`.

    Example:
        resample_linear(np.array([0., 1., 2.]), np.array([0., 10., 0.]), np.array([0.5, 1.5, 3.0]))
        -> array([5., 5., 0.])
    """
    raise NotImplementedError
