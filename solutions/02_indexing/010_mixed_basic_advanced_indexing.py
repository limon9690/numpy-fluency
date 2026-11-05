"""010 · Mixing basic and advanced indexing, np.ix_ (reference solution)."""

import numpy as np


def submatrix(a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    # ix_ turns the indices into (r, 1) and (1, c) arrays, which broadcast to a grid.
    return a[np.ix_(rows, cols)]


def channel_pixels(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return img[ys, xs]  # paired (k,) indices, then the untouched channel axis


def select_frame_channels(video: np.ndarray, t: int, channels: list[int]) -> np.ndarray:
    # video[t, :, :, channels] would put the channel axis first (advanced indices split by slices).
    return video[t][..., channels]


def batch_gather(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    batch = np.arange(x.shape[0])[:, None]  # (B, 1) broadcasts against idx (B, K)
    return x[batch, idx]
