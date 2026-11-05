"""012 · Broadcasting rules."""

import numpy as np


def broadcast_shape(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    """The shape that results from broadcasting arrays of shapes `shape1` and `shape2`.

    Apply the rule yourself: align from the right, pad the shorter shape with 1s on the left;
    two lengths are compatible if equal or if one is 1. Raise ValueError if incompatible.
    np.broadcast_shapes and np.broadcast are banned. Returns a tuple of ints.

    Example:
        broadcast_shape((8, 1, 6, 1), (7, 1, 5)) -> (8, 7, 6, 5)
        broadcast_shape((3, 4), (3,))            -> ValueError
    """
    raise NotImplementedError


def outer_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All pairwise sums: out[i, j] = a[i] + b[j].

    `a` has shape (n,), `b` has shape (m,). Returns shape (n, m).
    No Python loops; np.add.outer and np.outer are banned.

    Example:
        outer_sum(np.array([0, 10, 20]), np.array([1, 2])) -> [[ 1,  2],
                                                              [11, 12],
                                                              [21, 22]]
    """
    raise NotImplementedError


def pairwise_diff(x: np.ndarray) -> np.ndarray:
    """All pairwise differences between rows: out[i, j] = x[i] - x[j].

    `x` has shape (n, d). Returns shape (n, n, d). No Python loops; np.subtract.outer is banned.

    Example:
        pairwise_diff(np.array([[0, 0], [1, 2]]))[1, 0] -> array([1, 2])
    """
    raise NotImplementedError


def pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every row of `x` and every row of `y`.

    `x` has shape (n, d), `y` has shape (m, d). Returns float64 of shape (n, m) with
    out[i, j] = sum_k (x[i, k] - y[j, k]) ** 2. No Python loops.

    Example:
        pairwise_sq_dist(np.array([[0., 0.]]), np.array([[3., 4.], [1., 0.]])) -> [[25., 1.]]
    """
    raise NotImplementedError
