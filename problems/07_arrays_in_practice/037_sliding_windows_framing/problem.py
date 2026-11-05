"""037 · Sliding windows and framing (sliding_window_view)."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view  # noqa: F401  (you'll want this)


def frames(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Cut a 1-D signal into (possibly overlapping) frames, without copying.

    Frame i is signal[i*hop : i*hop + frame_len]. Frames that would run past the end are
    dropped, so n_frames = 1 + (len(signal) - frame_len) // hop (assume len(signal) >= frame_len).
    Returns a read-only view of shape (n_frames, frame_len) that shares memory with `signal`.

    Example:
        frames(np.arange(7), 3, 2) -> [[0, 1, 2],
                                       [2, 3, 4],
                                       [4, 5, 6]]
    """
    raise NotImplementedError


def moving_max(x: np.ndarray, w: int) -> np.ndarray:
    """The maximum of every window of length w of the 1-D array x.

    Returns shape (len(x) - w + 1,), same dtype, out[i] = max(x[i:i+w]).

    Example:
        moving_max(np.array([1, 3, 2, 5, 4, 1]), 3) -> array([3, 5, 5, 5])
    """
    raise NotImplementedError


def patches_2d(img: np.ndarray, k: int) -> np.ndarray:
    """All k x k patches of a 2-D image, as a view.

    `img` has shape (H, W). Returns a read-only view of shape (H - k + 1, W - k + 1, k, k) with
    out[i, j] = img[i:i+k, j:j+k]. Shares memory with `img`.

    Example:
        patches_2d(np.arange(16).reshape(4, 4), 2)[0, 1] -> [[1, 2],
                                                            [5, 6]]
    """
    raise NotImplementedError


def box_blur_valid(img: np.ndarray, k: int) -> np.ndarray:
    """Box blur without padding: the mean of every k x k patch of a 2-D image.

    Returns float64 of shape (H - k + 1, W - k + 1) with out[i, j] = img[i:i+k, j:j+k].mean().

    Example:
        box_blur_valid(np.arange(16).reshape(4, 4), 2) -> [[ 2.5,  3.5,  4.5],
                                                          [ 6.5,  7.5,  8.5],
                                                          [10.5, 11.5, 12.5]]
    """
    raise NotImplementedError
