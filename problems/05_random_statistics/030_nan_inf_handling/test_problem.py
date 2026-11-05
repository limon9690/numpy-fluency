import warnings

import numpy as np
import pytest


def _quiet(func, *args):
    """Call func with every warning turned into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args)


# ---------------------------------------------------------------- nan_column_means

def test_nan_column_means_example(sol, no_loops):
    X = np.array([[1.0, np.nan, 5.0], [3.0, np.nan, np.nan]])
    out = _quiet(sol.nan_column_means, X)
    assert out.dtype == np.float64 and out.shape == (3,)
    np.testing.assert_allclose(out, [2.0, np.nan, 5.0], err_msg="an all-NaN column should give NaN, silently")
    no_loops(sol.nan_column_means)


def test_nan_column_means_random(sol):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    X[rng.random((30, 5)) < 0.3] = np.nan
    out = _quiet(sol.nan_column_means, X)
    np.testing.assert_allclose(out, np.nanmean(X, axis=0))


# ---------------------------------------------------------------- fill_nan_with_column_mean

def test_fill_nan_example(sol, no_loops):
    X = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, np.nan]])
    before = X.copy()
    out = _quiet(sol.fill_nan_with_column_mean, X)
    np.testing.assert_allclose(out, [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    np.testing.assert_array_equal(X, before, err_msg="the input must not be modified")
    no_loops(sol.fill_nan_with_column_mean)


def test_fill_nan_random_keeps_inf(sol):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 4))
    mask = rng.random((20, 4)) < 0.25
    X[mask] = np.nan
    X[0, 3] = np.inf
    out = _quiet(sol.fill_nan_with_column_mean, X)
    assert not np.isnan(out).any(), "no NaN may remain"
    assert out[0, 3] == np.inf, "inf is not NaN and must be kept"
    means = np.nanmean(X, axis=0)
    np.testing.assert_allclose(out[mask], np.broadcast_to(means, X.shape)[mask])
    np.testing.assert_array_equal(out[~mask], X[~mask])


# ---------------------------------------------------------------- finite_stats

def test_finite_stats_example(sol, no_loops):
    out = sol.finite_stats(np.array([1.0, np.nan, 3.0, np.inf, -np.inf]))
    assert out == {"min": 1.0, "max": 3.0, "mean": 2.0, "count": 2}
    assert type(out["min"]) is float and type(out["count"]) is int, "use Python float and int"
    no_loops(sol.finite_stats)


def test_finite_stats_2d_and_empty(sol):
    x = np.array([[np.nan, -2.0], [4.0, 10.0]])
    out = sol.finite_stats(x)
    assert out["count"] == 3 and out["min"] == -2.0 and out["max"] == 10.0
    assert out["mean"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        sol.finite_stats(np.array([np.nan, np.inf]))


# ---------------------------------------------------------------- relative_change

def test_relative_change_example(sol, no_loops):
    out = _quiet(sol.relative_change, np.array([2.0, 0.0, 0.0, 0.0]), np.array([3.0, 1.0, -1.0, 0.0]))
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [0.5, np.inf, -np.inf, np.nan])
    no_loops(sol.relative_change)


def test_relative_change_broadcast_and_int(sol):
    old = np.array([[1], [0]])
    new = np.array([2, 0, -4])
    out = _quiet(sol.relative_change, old, new)
    assert out.shape == (2, 3) and out.dtype == np.float64
    np.testing.assert_array_equal(out, [[1.0, -1.0, -5.0], [np.inf, np.nan, -np.inf]])
