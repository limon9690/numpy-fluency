"""026 · Generator API and seeding."""

import numpy as np


def noisy_signal(n: int, noise_std: float, seed: int) -> np.ndarray:
    """A sine wave with Gaussian noise, reproducible from `seed`.

    Recipe (follow it exactly):
        rng = np.random.default_rng(seed)
        t = n evenly spaced points from 0 to 2*pi, both included
        return sin(t) + rng.normal(0, noise_std, size=n)

    Returns float64 of shape (n,). Same seed -> identical output.

    Example:
        np.array_equal(noisy_signal(100, 0.1, 7), noisy_signal(100, 0.1, 7)) -> True
    """
    raise NotImplementedError


def dropout_mask(shape: tuple[int, ...], p: float, seed: int) -> np.ndarray:
    """Boolean keep-mask for dropout with drop probability p.

    Recipe: rng = np.random.default_rng(seed); return rng.random(shape) >= p.
    Must not use or change NumPy's legacy global random state (np.random.seed / np.random.rand).

    Example:
        dropout_mask((2, 3), 0.5, 0).dtype -> bool
    """
    raise NotImplementedError


def spawn_streams(seed: int, n: int) -> list[np.random.Generator]:
    """n independent random Generators derived from one seed.

    Recipe: spawn n children of np.random.SeedSequence(seed) and make one Generator from each
    (np.random.default_rng(seed).spawn(n) gives the same streams).

    Example:
        a, b = spawn_streams(0, 2)
        a.random() != b.random()  -> True
    """
    raise NotImplementedError


def bootstrap_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means of n_boot bootstrap resamples of the 1-D array `x`.

    Recipe: idx = rng.integers(0, len(x), size=(n_boot, len(x))) in ONE call, then the mean of
    x[idx] along axis 1. Use the Generator you're given (don't create a new one).
    Returns float64 of shape (n_boot,). No Python loops.

    Example:
        bootstrap_means(np.array([1., 2., 3.]), 4, np.random.default_rng(0)).shape -> (4,)
    """
    raise NotImplementedError
