import warnings

import numpy as np
import pytest


def _call_quietly(func, *args):
    """Run func and turn any NumPy RuntimeWarning (divide by zero, invalid value) into a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args)


# ---------------------------------------------------------------- row_normalize

def test_row_normalize_example(sol, no_loops):
    a = np.array([[1, 3], [0, 0], [2, 2]])
    out = _call_quietly(sol.row_normalize, a)
    assert out.dtype == np.float64, f"int input should give float64, got {out.dtype}"
    np.testing.assert_allclose(out, [[0.25, 0.75], [0.0, 0.0], [0.5, 0.5]],
                               err_msg="all-zero rows must stay zero, not NaN")
    no_loops(sol.row_normalize)


def test_row_normalize_random_float32(sol):
    rng = np.random.default_rng(0)
    a = rng.random((6, 4)).astype(np.float32)
    before = a.copy()
    out = sol.row_normalize(a)
    assert out.dtype == np.float32, f"float32 input should stay float32, got {out.dtype}"
    np.testing.assert_allclose(out.sum(axis=1), np.ones(6), rtol=1e-6, err_msg="each row should sum to 1")
    np.testing.assert_allclose(out, before / before.sum(axis=1, keepdims=True), rtol=1e-6)
    np.testing.assert_array_equal(a, before, err_msg="the input must not be modified")


def test_row_normalize_square(sol):
    a = np.array([[1.0, 1.0, 2.0], [3.0, 3.0, 3.0], [0.0, 0.0, 5.0]])
    out = sol.row_normalize(a)
    np.testing.assert_allclose(out.sum(axis=1), [1, 1, 1],
                               err_msg="rows must sum to 1; on a square matrix a / a.sum(axis=1) divides columns")


# ---------------------------------------------------------------- standardize_columns

def test_standardize_columns_example(sol, no_loops):
    out = _call_quietly(sol.standardize_columns, np.array([[1, 5], [3, 5]]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[-1.0, 0.0], [1.0, 0.0]], err_msg="constant columns must become 0, not NaN")
    no_loops(sol.standardize_columns)


def test_standardize_columns_random(sol):
    rng = np.random.default_rng(0)
    a = rng.normal([0, 50, -3], [1, 10, 0.1], size=(200, 3))
    before = a.copy()
    out = sol.standardize_columns(a)
    assert out.shape == a.shape
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12, err_msg="every column should have mean 0")
    np.testing.assert_allclose(out.std(axis=0), 1, rtol=1e-12, err_msg="every column should have std 1 (ddof=0)")
    np.testing.assert_array_equal(a, before, err_msg="the input must not be modified")


def test_standardize_columns_float32(sol):
    a = np.arange(12, dtype=np.float32).reshape(4, 3) ** 2
    out = sol.standardize_columns(a)
    assert out.dtype == np.float32, f"float32 input should stay float32, got {out.dtype}"
    np.testing.assert_allclose(out, (a - a.mean(0)) / a.std(0), rtol=1e-5)


# ---------------------------------------------------------------- minmax_scale_columns

def test_minmax_scale_columns_example(sol, no_loops):
    out = _call_quietly(sol.minmax_scale_columns, np.array([[0, 10], [5, 10], [10, 10]]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    no_loops(sol.minmax_scale_columns)


def test_minmax_scale_columns_random(sol):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(50, 4), dtype=np.uint8)
    out = sol.minmax_scale_columns(a)
    assert out.dtype == np.float64, f"uint8 input should give float64, got {out.dtype}"
    np.testing.assert_allclose(out.min(axis=0), 0)
    np.testing.assert_allclose(out.max(axis=0), 1)
    af = a.astype(np.float64)
    np.testing.assert_allclose(out, (af - af.min(0)) / (af.max(0) - af.min(0)))


# ---------------------------------------------------------------- scale_rows

def test_scale_rows_square_trap(sol, no_loops):
    out = sol.scale_rows(np.array([[1, 1], [1, 1]]), np.array([2, 3]))
    np.testing.assert_array_equal(out, [[2, 2], [3, 3]], err_msg="row i should be scaled by w[i] (got columns?)")
    no_loops(sol.scale_rows)


def test_scale_rows_rect(sol):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 6))
    w = rng.standard_normal(4)
    before = a.copy()
    out = sol.scale_rows(a, w)
    assert out.shape == (4, 6)
    np.testing.assert_allclose(out, np.diag(w) @ before)
    np.testing.assert_array_equal(a, before, err_msg="the input must not be modified")
