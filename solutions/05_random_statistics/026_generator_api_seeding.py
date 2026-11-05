"""026 · Generator API and seeding (reference solution)."""

import numpy as np


def noisy_signal(n: int, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    return np.sin(t) + rng.normal(0, noise_std, size=n)


def dropout_mask(shape: tuple[int, ...], p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)  # its own state: the global np.random state is untouched
    return rng.random(shape) >= p


def spawn_streams(seed: int, n: int) -> list[np.random.Generator]:
    # Spawned children are statistically independent, unlike hand-picked seeds 0, 1, 2, ...
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n)]


def bootstrap_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))  # one resample per row
    return x[idx].mean(axis=1)
