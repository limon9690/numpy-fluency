"""007 · Fancy indexing (reference solution)."""

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    # Paired index arrays: (row i, column labels[i]) for every i.
    out[np.arange(len(labels)), labels] = 1
    return out


def pick_per_row(a: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return a[np.arange(a.shape[0]), cols]


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[img]  # result takes the shape of the index array


def inverse_permutation(p: np.ndarray) -> np.ndarray:
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p), dtype=p.dtype)  # "position p[i] gets value i"
    return inv
