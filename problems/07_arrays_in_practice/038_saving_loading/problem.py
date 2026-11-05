"""038 · Saving and loading: .npy, .npz, memmap, loadtxt."""

import numpy as np


def save_arrays(path, arrays: dict) -> None:
    """Save several named arrays into one compressed .npz file at `path`.

    `path` ends with ".npz"; `arrays` maps names (valid Python identifiers) to arrays.
    Loading the file back with np.load must give the same names, dtypes, shapes and values.

    Example:
        save_arrays(tmp / "d.npz", {"x": np.arange(3), "y": np.eye(2)})
    """
    raise NotImplementedError


def load_arrays(path) -> dict:
    """Load an .npz file into a plain dict {name: np.ndarray}.

    The values must be real ndarrays (not lazy references into the file) and the file must be
    closed when the function returns.

    Example:
        load_arrays(tmp / "d.npz") -> {"x": array([0, 1, 2]), "y": array([[1., 0.], [0., 1.]])}
    """
    raise NotImplementedError


def memmap_column_mean(path, shape: tuple[int, int], dtype, col: int) -> float:
    """Mean of one column of a big 2-D array stored as raw binary on disk.

    The file at `path` holds the array's bytes in C order with no header (as written by
    a.tofile(path)), with the given `shape` and `dtype`. Open it read-only with np.memmap
    (don't load the whole file with np.fromfile/np.load), and return the mean of column `col`,
    accumulated in float64, as a Python float. The file must not be modified.

    Example:
        np.arange(12, dtype=np.float32).reshape(4, 3).tofile(p)
        memmap_column_mean(p, (4, 3), np.float32, 1) -> 5.5
    """
    raise NotImplementedError


def load_csv(path) -> tuple[list[str], np.ndarray]:
    """Read a comma-separated file of numbers that has one header line.

    Returns (names, data): names is the list of column names from the header (whitespace
    stripped), data is a float64 array of shape (n_rows, n_columns), 2-D even when there is
    only one data row.

    Example (file contents):
        x,y,label
        1.5,2.0,0
    load_csv(path) -> (["x", "y", "label"], array([[1.5, 2. , 0. ]]))
    """
    raise NotImplementedError
