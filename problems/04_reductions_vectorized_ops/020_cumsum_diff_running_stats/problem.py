"""020 · cumsum, diff and running statistics."""

import numpy as np


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of every window of length w of the 1-D array `x` ("valid" windows only).

    1 <= w <= len(x). Returns float64 of length len(x) - w + 1 with
    out[i] = mean(x[i:i+w]). No Python loops, no np.convolve, no sliding_window_view.

    Example:
        moving_average(np.array([3, 1, 4, 1, 5]), 3) -> array([2.6667, 2.    , 3.3333])
    """
    raise NotImplementedError


def running_mean(x: np.ndarray) -> np.ndarray:
    """Cumulative mean: out[i] = mean(x[:i+1]) for a 1-D array `x`.

    Returns float64 of the same length.

    Example:
        running_mean(np.array([2, 4, 6, 0])) -> array([2., 3., 4., 3.])
    """
    raise NotImplementedError


def lengths_to_offsets(lengths: np.ndarray) -> np.ndarray:
    """Start offsets of consecutive segments with the given lengths, plus the total at the end.

    `lengths` is a 1-D array of non-negative ints. Returns int64 of length len(lengths) + 1:
    offsets[0] = 0 and offsets[k+1] = offsets[k] + lengths[k]. Segment k of a packed array is
    packed[offsets[k]:offsets[k+1]].

    Example:
        lengths_to_offsets(np.array([2, 3, 1])) -> array([0, 2, 5, 6])
    """
    raise NotImplementedError


def rising_edges(signal: np.ndarray) -> np.ndarray:
    """Indices where a boolean signal switches on.

    `signal` is a 1-D bool array. Returns a 1-D int array of every i >= 1 with
    signal[i-1] == False and signal[i] == True, in increasing order (may be empty).

    Example:
        rising_edges(np.array([False, False, True, True, False, True])) -> array([2, 5])
    """
    raise NotImplementedError
