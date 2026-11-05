"""017 · tile, repeat, pad, meshgrid (reference solution)."""

import numpy as np


def upsample_nearest(img: np.ndarray, k: int) -> np.ndarray:
    # Always pass axis=: without it, repeat flattens the image first.
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def pad_to_multiple(img: np.ndarray, m: int) -> np.ndarray:
    h, w = img.shape[:2]
    # One (before, after) pair per axis; extra axes (channels) get no padding.
    pad = [(0, (-h) % m), (0, (-w) % m)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad)


def coordinate_grid(h: int, w: int) -> np.ndarray:
    # indexing="ij" gives (h, w) arrays; the default "xy" would give (w, h).
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([rows, cols], axis=-1)


def repeat_and_cycle(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(labels, k), np.tile(labels, k)
