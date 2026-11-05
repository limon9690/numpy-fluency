import numpy as np
import pytest


# ---------------------------------------------------------------- count_in_range

def test_count_in_range_example(sol, no_loops):
    x = np.array([[1, 5, 9], [3, 4, 10]])
    out = sol.count_in_range(x, 3, 9)
    assert isinstance(out, (int, np.integer)), f"expected an int, got {type(out).__name__}"
    assert out == 4, f"expected 4 (3, 4, 5, 9 with both ends included), got {out}"
    no_loops(sol.count_in_range)


def test_count_in_range_random(sol):
    x = np.random.default_rng(0).standard_normal((40, 25))
    assert sol.count_in_range(x, -0.5, 1.0) == np.sum((x >= -0.5) & (x <= 1.0))
    assert sol.count_in_range(x, 5, 6) == 0


# ---------------------------------------------------------------- replace_outside

def test_replace_outside_example(sol, no_loops):
    x = np.array([-5, 0, 3, 12])
    out = sol.replace_outside(x, 0, 10, -1)
    np.testing.assert_array_equal(out, [-1, 0, 3, -1])
    no_loops(sol.replace_outside)


def test_replace_outside_2d_keeps_shape_dtype_and_input(sol):
    x = np.random.default_rng(0).standard_normal((5, 6)).astype(np.float32)
    before = x.copy()
    out = sol.replace_outside(x, -1.0, 1.0, 0.0)
    assert out.shape == (5, 6), f"shape should stay (5, 6), got {out.shape}"
    assert out.dtype == np.float32, f"dtype should stay float32, got {out.dtype}"
    expected = np.where((before < -1) | (before > 1), 0, before)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(x, before, err_msg="replace_outside must not modify its input")
    assert not np.shares_memory(out, x), "return a new array, not `x` itself"


# ---------------------------------------------------------------- select_rows

def test_select_rows_example(sol, no_loops):
    data = np.array([[1, 50], [2, 10], [3, 70]])
    out = sol.select_rows(data, 1, 20)
    np.testing.assert_array_equal(out, [[1, 50], [3, 70]])
    no_loops(sol.select_rows)


def test_select_rows_shapes(sol):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=(30, 4))
    out = sol.select_rows(data, 2, 60)
    assert out.ndim == 2 and out.shape[1] == 4, f"rows should keep all 4 columns, got shape {out.shape}"
    np.testing.assert_array_equal(out, np.array([r for r in data if r[2] > 60]).reshape(-1, 4))
    none = sol.select_rows(data, 0, 1000)
    assert none.shape == (0, 4), f"no matching rows should give shape (0, 4), got {none.shape}"


# ---------------------------------------------------------------- drop_nan_pairs

def test_drop_nan_pairs_example(sol, no_loops):
    nan = np.nan
    x = np.array([1.0, nan, 3.0, 4.0])
    y = np.array([10.0, 20.0, nan, 40.0])
    xc, yc = sol.drop_nan_pairs(x, y)
    np.testing.assert_array_equal(xc, [1.0, 4.0])
    np.testing.assert_array_equal(yc, [10.0, 40.0])
    assert np.isnan(x[1]) and np.isnan(y[2]), "inputs must not be modified"
    no_loops(sol.drop_nan_pairs)


def test_drop_nan_pairs_random(sol):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    x[rng.random(100) < 0.2] = np.nan
    y[rng.random(100) < 0.2] = np.nan
    xc, yc = sol.drop_nan_pairs(x, y)
    keep = [i for i in range(100) if not (np.isnan(x[i]) or np.isnan(y[i]))]
    assert xc.shape == yc.shape == (len(keep),), "both outputs need the same length"
    np.testing.assert_array_equal(xc, x[keep])
    np.testing.assert_array_equal(yc, y[keep])
    assert not np.isnan(xc).any() and not np.isnan(yc).any()
