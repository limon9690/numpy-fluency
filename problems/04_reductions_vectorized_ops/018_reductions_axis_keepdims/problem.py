"""018 · Reductions with axis and keepdims."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`.

    out = exp(x - max) / sum(exp(x - max)), with max and sum taken along `axis`.
    Must not overflow for large inputs (e.g. 1000). Same shape as `x`; values along `axis` sum
    to 1. Float input keeps its dtype (float32 stays float32); integer input gives float64.

    Example:
        softmax(np.array([[1., 2., 3.], [1000., 1000., 1000.]]), axis=1)
        -> [[0.0900, 0.2447, 0.6652],
            [0.3333, 0.3333, 0.3333]]
    """
    raise NotImplementedError


def argmax_2d(a: np.ndarray) -> tuple[int, int]:
    """Row and column of the largest element of the 2-D array `a`.

    On ties, the first one in C order (what np.argmax picks). Returns a tuple of two Python ints.

    Example:
        argmax_2d(np.array([[1, 5, 2], [7, 0, 3]])) -> (1, 0)
    """
    raise NotImplementedError


def channel_mean_std(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of a batch of images.

    `images` has shape (N, H, W, C) and any numeric dtype (often uint8). The statistics are over
    all images and all pixels. std is the population std (ddof=0).
    Returns (mean, std), both float64 of shape (C,).

    Example:
        images = np.zeros((2, 4, 4, 3), np.uint8); images[..., 1] = 10
        channel_mean_std(images) -> (array([ 0., 10.,  0.]), array([0., 0., 0.]))
    """
    raise NotImplementedError


def center_per_sample(x: np.ndarray) -> np.ndarray:
    """Subtract from every sample x[i] the mean of all its own elements.

    `x` has shape (N, ...) with any number of trailing axes (possibly none). Returns float64 of
    the same shape where every out[i] has mean 0. For 1-D input every sample is a single number,
    so the result is all zeros.

    Example:
        center_per_sample(np.array([[1, 3], [10, 20]])) -> [[-1.,  1.],
                                                           [-5.,  5.]]
    """
    raise NotImplementedError
