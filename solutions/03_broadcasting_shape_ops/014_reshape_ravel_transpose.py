"""014 · reshape, ravel vs flatten, transpose, moveaxis (reference solution)."""

import math

import numpy as np


def to_patches(img: np.ndarray, p: int) -> np.ndarray:
    h, w = img.shape
    # (patch row, row in patch, patch col, col in patch) -> bring the two patch indices together.
    grid = img.reshape(h // p, p, w // p, p).swapaxes(1, 2)
    return grid.reshape(-1, p, p)


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    return np.moveaxis(img, -1, -3)  # counting from the end works with or without a batch axis


def flatten_batch(x: np.ndarray) -> np.ndarray:
    # reshape(B, -1) is ambiguous when B == 0, so give the size explicitly.
    return x.reshape(x.shape[0], math.prod(x.shape[1:]))


def increment_flat(a: np.ndarray, i: int, value) -> None:
    # a.flat indexes in C order and writes into a's own memory, unlike a.ravel() (maybe a copy).
    a.flat[i] += value
