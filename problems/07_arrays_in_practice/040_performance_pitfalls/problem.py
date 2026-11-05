"""040 · Performance: vectorized vs loops, fake vectorization, memory blow-ups."""

import numpy as np


def pairwise_sq_dist_fast(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between all rows of X and all rows of Y, memory-efficiently.

    `X` is (n, d), `Y` is (m, d). Returns float64 (n, m) with out[i, j] = ||X[i] - Y[j]||**2,
    clipped at 0 (rounding can make tiny values negative). Must not create any (n, m, d) array:
    use ||x||^2 + ||y||^2 - 2 x.y. No Python loops.

    Example:
        pairwise_sq_dist_fast(np.array([[0., 0.]]), np.array([[3., 4.], [1., 0.]])) -> [[25., 1.]]
    """
    raise NotImplementedError


def standardize_inplace_f32(x: np.ndarray) -> None:
    """Standardize a C-contiguous float32 array in place: x <- (x - mean) / std. Returns None.

    mean and std are over all elements; std is the population std (ddof=0); assume std > 0.
    `x` must stay float32 and the extra memory used must stay far below x.nbytes (no full-size
    temporaries; note that x.std() makes one internally). No Python loops.

    Example:
        x = np.array([1., 2., 3.], np.float32); standardize_inplace_f32(x)
        x -> array([-1.2247,  0.    ,  1.2247], dtype=float32)
    """
    raise NotImplementedError


def lookup(keys: np.ndarray, table_keys: np.ndarray, table_values: np.ndarray, default) -> np.ndarray:
    """Map every key to its value in a lookup table, vectorized.

    `table_keys` is a 1-D int array of unique keys in any order, `table_values` the matching 1-D
    values. `keys` is an int array of any shape. Returns an array with the shape of `keys` and
    the dtype of `table_values`; keys not in the table get `default`.
    No Python loops; np.vectorize, np.frompyfunc and np.apply_along_axis are banned.

    Example:
        lookup(np.array([5, 1, 9]), np.array([1, 5, 3]), np.array([10., 50., 30.]), -1.0)
        -> array([50., 10., -1.])
    """
    raise NotImplementedError


def channel_means_uint8(imgs: np.ndarray) -> np.ndarray:
    """Per-channel mean of a uint8 image batch, without making a float copy of the batch.

    `imgs` has shape (N, H, W, C), dtype uint8. Returns float64 of shape (C,). Peak extra
    memory must stay well below imgs.nbytes (imgs.astype(np.float64) alone is 8x that).
    No Python loops.

    Example:
        imgs = np.zeros((2, 4, 4, 3), np.uint8); imgs[..., 2] = 255
        channel_means_uint8(imgs) -> array([  0.,   0., 255.])
    """
    raise NotImplementedError
