import numpy as np
import pytest


# ---------------------------------------------------------------- iqr_outlier_mask

def test_iqr_outlier_mask_example(sol, no_loops):
    out = sol.iqr_outlier_mask(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]))
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, [False] * 9 + [True])
    no_loops(sol.iqr_outlier_mask)


def test_iqr_outlier_mask_random(sol):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 200), [-15.0, 12.0]])
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    expected = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
    out = sol.iqr_outlier_mask(x)
    np.testing.assert_array_equal(out, expected)
    assert out[-1] and out[-2], "the two planted outliers should be flagged"


# ---------------------------------------------------------------- winsorize

def test_winsorize_example(sol, no_loops):
    x = np.arange(11.0)
    out = sol.winsorize(x, 10, 90)
    np.testing.assert_allclose(out, [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
    np.testing.assert_array_equal(x, np.arange(11.0), err_msg="winsorize must not modify its input")
    no_loops(sol.winsorize)


def test_winsorize_per_column(sol):
    rng = np.random.default_rng(0)
    X = rng.normal([0, 100], [1, 50], size=(200, 2))
    out = sol.winsorize(X, 5, 95, axis=0)
    assert out.shape == X.shape
    for j in range(2):
        lo, hi = np.percentile(X[:, j], [5, 95])
        np.testing.assert_allclose(out[:, j], np.clip(X[:, j], lo, hi),
                                   err_msg=f"column {j} should be clipped at its own percentiles")


def test_winsorize_per_row_and_global(sol):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 50)) * np.array([[1], [10], [100], [1000]])
    out = sol.winsorize(X, 10, 90, axis=1)
    for i in range(4):
        lo, hi = np.percentile(X[i], [10, 90])
        np.testing.assert_allclose(out[i], np.clip(X[i], lo, hi))
    lo, hi = np.percentile(X, [10, 90])
    np.testing.assert_allclose(sol.winsorize(X, 10, 90), np.clip(X, lo, hi), err_msg="axis=None uses one pair of bounds")


# ---------------------------------------------------------------- robust_scale

def test_robust_scale_example(sol, no_loops):
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [100.0, 5.0]])
    out = sol.robust_scale(X)
    assert out.dtype == np.float64 and out.shape == (5, 2)
    np.testing.assert_allclose(out[:, 0], [-1.0, -0.5, 0.0, 0.5, 48.5])
    np.testing.assert_allclose(out[:, 1], 0.0, err_msg="a constant column should just be centered (IQR 0 -> divide by 1)")
    no_loops(sol.robust_scale)


def test_robust_scale_random_int(sol):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 1000, size=(60, 3))
    out = sol.robust_scale(X)
    med = np.median(X, axis=0)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    np.testing.assert_allclose(out, (X - med) / (q3 - q1))


# ---------------------------------------------------------------- quantile_bins

def test_quantile_bins_example(sol, no_loops):
    out = sol.quantile_bins(np.array([5, 1, 4, 2, 3, 6, 8, 7]), 4)
    np.testing.assert_array_equal(out, [2, 0, 1, 0, 1, 2, 3, 3])
    assert np.issubdtype(out.dtype, np.integer)
    no_loops(sol.quantile_bins)


def test_quantile_bins_balanced(sol):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    out = sol.quantile_bins(x, 10)
    assert out.min() == 0 and out.max() == 9, "bins should run from 0 to n_bins - 1 (max in the last bin)"
    np.testing.assert_array_equal(np.bincount(out), np.full(10, 100), err_msg="deciles of distinct values hold 100 each")
    edges = np.quantile(x, np.linspace(0, 1, 11))
    k = 3
    inside = (x >= edges[k]) & (x < edges[k + 1])
    np.testing.assert_array_equal(out == k, inside)
