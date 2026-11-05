"""027 · Sampling: choice, permutation, shuffle."""

import numpy as np


def split_indices(n: int, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random train/test split of the indices 0..n-1.

    Recipe: perm = rng.permutation(n); n_test = round(test_frac * n);
    test = perm[:n_test], train = perm[n_test:].
    Returns (train, test): disjoint int arrays that together contain every index once.

    Example:
        train, test = split_indices(10, 0.3, np.random.default_rng(0))
        len(train), len(test) -> (7, 3)
    """
    raise NotImplementedError


def shuffle_together(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle two aligned arrays along axis 0 with the same random order.

    `x` has shape (n, ...) and `y` has shape (n, ...). Recipe: perm = rng.permutation(n), then
    reorder both with it. Returns new arrays (x_shuffled, y_shuffled); inputs are not modified.

    Example:
        xs, ys = shuffle_together(np.arange(5), np.arange(5) * 10, np.random.default_rng(0))
        np.array_equal(ys, xs * 10) -> True
    """
    raise NotImplementedError


def sample_distinct(population: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k elements of the 1-D array `population` at k different positions (no replacement).

    Recipe: rng.choice(population, size=k, replace=False). Returns shape (k,), same dtype.
    If k > len(population), the ValueError from rng.choice is fine.

    Example:
        sample_distinct(np.arange(100), 5, np.random.default_rng(0)).shape -> (5,)
    """
    raise NotImplementedError


def weighted_sample(classes: np.ndarray, weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n draws (with replacement) from `classes`, with probability proportional to `weights`.

    `weights` are non-negative and need not sum to 1 (e.g. raw counts). Recipe:
    p = weights / weights.sum(); rng.choice(classes, size=n, p=p). Returns shape (n,).

    Example:
        weighted_sample(np.array([0, 1]), np.array([3, 1]), 8, rng)  # about 75% zeros
    """
    raise NotImplementedError
