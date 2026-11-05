"""010 · Mixing basic and advanced indexing, np.ix_."""

import numpy as np


def submatrix(a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The block of `a` made of the given rows and the given columns (every combination).

    `a` is 2-D. `rows` and `cols` are each either a 1-D int array or a 1-D boolean mask
    (length = a.shape[0] for rows, a.shape[1] for cols). Returns shape (len(rows), len(cols))
    for int arrays (number of Trues for masks). Order follows the index arrays.

    Example:
        a = np.arange(12).reshape(3, 4)
        submatrix(a, np.array([0, 2]), np.array([1, 3])) -> [[ 1,  3],
                                                             [ 9, 11]]
    """
    raise NotImplementedError


def channel_pixels(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """All channel values at k points of an image.

    `img` has shape (H, W, C); `ys` and `xs` are int arrays of shape (k,).
    Returns shape (k, C) where out[i] = img[ys[i], xs[i], :].

    Example:
        img of shape (4, 5, 3), ys = [0, 3], xs = [1, 4] -> shape (2, 3): img[0, 1] and img[3, 4]
    """
    raise NotImplementedError


def select_frame_channels(video: np.ndarray, t: int, channels: list[int]) -> np.ndarray:
    """Frame `t` of a video, keeping only the listed channels.

    `video` has shape (T, H, W, C). `channels` is a list of channel indices.
    Returns shape (H, W, len(channels)) with out[..., j] = video[t, :, :, channels[j]].

    Example:
        video of shape (10, 4, 5, 3), t = 2, channels = [2, 0] -> shape (4, 5, 2)  (BGR-style order)
    """
    raise NotImplementedError


def batch_gather(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """For every batch element, gather some rows.

    `x` has shape (B, N, D); `idx` is an int array of shape (B, K) with values in [0, N).
    Returns shape (B, K, D) with out[b, k] = x[b, idx[b, k]].
    np.take_along_axis and np.take are banned here.

    Example:
        x of shape (2, 4, 3), idx = [[3, 0], [1, 1]] -> out[0] = x[0, [3, 0]], out[1] = x[1, [1, 1]]
    """
    raise NotImplementedError
