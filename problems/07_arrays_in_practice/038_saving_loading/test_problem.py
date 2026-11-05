import numpy as np
import pytest


def _arrays():
    rng = np.random.default_rng(0)
    return {
        "images": rng.integers(0, 256, size=(4, 5, 5, 3), dtype=np.uint8),
        "labels": np.array([0, 2, 1, 1], dtype=np.int64),
        "weights": rng.standard_normal((3, 2)).astype(np.float32),
        "scalar": np.array(3.25),
    }


# ---------------------------------------------------------------- save_arrays / load_arrays

def test_save_arrays_readable_by_np_load(sol, tmp_path):
    path = tmp_path / "data.npz"
    arrays = _arrays()
    sol.save_arrays(path, arrays)
    assert path.exists(), "save_arrays should create the .npz file at `path`"
    with np.load(path) as f:
        assert sorted(f.files) == sorted(arrays), f"saved names {sorted(f.files)} != {sorted(arrays)}"
        for k, v in arrays.items():
            assert f[k].dtype == v.dtype and f[k].shape == v.shape, f"{k}: dtype or shape changed"
            np.testing.assert_array_equal(f[k], v)


def test_save_arrays_is_compressed(sol, tmp_path):
    path = tmp_path / "zeros.npz"
    sol.save_arrays(path, {"z": np.zeros((500, 500))})  # 2 MB of zeros compresses to almost nothing
    assert path.stat().st_size < 100_000, "use np.savez_compressed"


def test_load_arrays_round_trip(sol, tmp_path):
    path = tmp_path / "data.npz"
    arrays = _arrays()
    np.savez(path, **arrays)
    out = sol.load_arrays(path)
    assert type(out) is dict, f"return a plain dict, got {type(out).__name__}"
    assert sorted(out) == sorted(arrays)
    for k, v in arrays.items():
        assert isinstance(out[k], np.ndarray), f"{k} should be an ndarray"
        assert out[k].dtype == v.dtype and out[k].shape == v.shape
        np.testing.assert_array_equal(out[k], v)


def test_load_arrays_values_survive_file_removal(sol, tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, a=np.arange(5))
    out = sol.load_arrays(path)
    path.unlink()
    np.testing.assert_array_equal(out["a"], np.arange(5), err_msg="values must be loaded, not lazy references")


# ---------------------------------------------------------------- memmap_column_mean

def test_memmap_column_mean_example(sol, tmp_path):
    p = tmp_path / "raw.bin"
    np.arange(12, dtype=np.float32).reshape(4, 3).tofile(p)
    out = sol.memmap_column_mean(p, (4, 3), np.float32, 1)
    assert type(out) is float, f"return a Python float, got {type(out).__name__}"
    assert out == pytest.approx(5.5)


def test_memmap_column_mean_random_untouched(sol, tmp_path, ban):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(2000, 16), dtype=np.uint8)
    p = tmp_path / "big.bin"
    a.tofile(p)
    before = p.read_bytes()
    ban(np, "fromfile")
    ban(np, "load")
    for col in (0, 7, 15):
        out = sol.memmap_column_mean(p, (2000, 16), np.uint8, col)
        assert out == pytest.approx(a[:, col].astype(np.float64).mean()), f"wrong mean for column {col}"
    assert p.read_bytes() == before, "the file must not be modified (open it with mode='r')"


# ---------------------------------------------------------------- load_csv

def test_load_csv(sol, tmp_path):
    p = tmp_path / "table.csv"
    p.write_text("x, y ,label\n1.5,2.0,0\n3.0,4.5,1\n-1,0,2\n")
    names, data = sol.load_csv(p)
    assert names == ["x", "y", "label"], f"header names should be stripped strings, got {names}"
    assert data.dtype == np.float64 and data.shape == (3, 3)
    np.testing.assert_allclose(data, [[1.5, 2.0, 0.0], [3.0, 4.5, 1.0], [-1.0, 0.0, 2.0]])


def test_load_csv_single_row(sol, tmp_path):
    p = tmp_path / "one.csv"
    p.write_text("a,b\n7,8\n")
    names, data = sol.load_csv(p)
    assert names == ["a", "b"]
    assert data.shape == (1, 2), f"a single data row should still give a 2-D array (1, 2), got {data.shape}"
