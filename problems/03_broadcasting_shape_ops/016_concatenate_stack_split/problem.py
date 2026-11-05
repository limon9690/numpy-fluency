"""016 · concatenate, stack, split."""

import numpy as np


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Put a column of ones in front of the 2-D matrix `X`.

    `X` has shape (n, d). Returns shape (n, d + 1) with column 0 all ones and the rest equal to
    X. The dtype is the same as X's (a float32 X gives a float32 result).

    Example:
        add_bias_column(np.array([[2., 3.], [4., 5.]], np.float32)) -> [[1., 2., 3.],
                                                                       [1., 4., 5.]]  (float32)
    """
    raise NotImplementedError


def split_batches(x: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Cut `x` into consecutive batches along axis 0.

    Every batch has `batch_size` rows except possibly the last, which holds what is left.
    Returns a list of arrays that are views of `x` (no copies). If len(x) is a multiple of
    batch_size there is no empty batch at the end. Assume len(x) >= 1.

    Example:
        [b.tolist() for b in split_batches(np.arange(7), 3)] -> [[0, 1, 2], [3, 4, 5], [6]]
    """
    raise NotImplementedError


def interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Alternate the rows of `a` and `b`: a[0], b[0], a[1], b[1], ...

    `a` and `b` have the same shape (n, ...) and dtype. Returns shape (2n, ...).

    Example:
        interleave(np.array([1, 2, 3]), np.array([10, 20, 30])) -> array([ 1, 10,  2, 20,  3, 30])
    """
    raise NotImplementedError


def join_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combine three (H, W) channel planes into one (H, W, 3) image.

    out[..., 0] == r, out[..., 1] == g, out[..., 2] == b. Same dtype as the inputs.

    Example:
        join_channels(np.zeros((4, 5)), np.ones((4, 5)), np.ones((4, 5))).shape -> (4, 5, 3)
    """
    raise NotImplementedError
