"""038 · Saving and loading: .npy, .npz, memmap, loadtxt (reference solution)."""

import numpy as np


def save_arrays(path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)  # keyword names become the keys inside the .npz


def load_arrays(path) -> dict:
    # NpzFile is lazy and keeps the zip open; read everything inside the `with`.
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def memmap_column_mean(path, shape: tuple[int, int], dtype, col: int) -> float:
    m = np.memmap(path, dtype=dtype, mode="r", shape=shape)  # raw file: we supply dtype and shape
    return float(m[:, col].mean(dtype=np.float64))


def load_csv(path) -> tuple[list[str], np.ndarray]:
    with open(path) as f:
        names = [name.strip() for name in f.readline().split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)  # ndmin=2: one row stays 2-D
    return names, data
