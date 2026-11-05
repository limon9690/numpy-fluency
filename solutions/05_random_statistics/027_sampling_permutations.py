"""027 · Sampling: choice, permutation, shuffle (reference solution)."""

import numpy as np


def split_indices(n: int, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    n_test = round(test_frac * n)
    return perm[n_test:], perm[:n_test]


def shuffle_together(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # One permutation for both keeps pairs aligned; fancy indexing returns copies.
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def sample_distinct(population: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(population, size=k, replace=False)


def weighted_sample(classes: np.ndarray, weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    p = weights / weights.sum()  # choice needs probabilities that sum to 1
    return rng.choice(classes, size=n, p=p)
