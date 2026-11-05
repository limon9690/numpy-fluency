"""024 · Set operations: isin, intersect1d, setdiff1d, union1d (reference solution)."""

import numpy as np


def new_ids(old: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.setdiff1d(current, old)


def keep_known(x: np.ndarray, vocab: np.ndarray, unknown) -> np.ndarray:
    # isin keeps x's shape, so the mask lines up element by element.
    return np.where(np.isin(x, vocab), x, unknown).astype(x.dtype, copy=False)


def common_with_positions(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.intersect1d(a, b, return_indices=True)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.union1d(a, b).size
    if union == 0:
        return 1.0
    return np.intersect1d(a, b).size / union
