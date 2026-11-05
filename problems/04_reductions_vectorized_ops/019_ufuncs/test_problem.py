import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------- safe_log

def test_safe_log_example(sol, no_loops):
    x = np.array([1.0, 0.0, -1.0, np.e])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # RuntimeWarning or the "where without out" warning fail
        out = sol.safe_log(x, fill=-99.0)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [0.0, -99.0, -99.0, 1.0])
    no_loops(sol.safe_log)


def test_safe_log_default_fill_2d_and_nan(sol):
    x = np.array([[np.nan, 4.0], [0.0, 1e-300]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sol.safe_log(x)
    assert out.shape == (2, 2)
    assert out[0, 0] == -np.inf and out[1, 0] == -np.inf, "NaN and 0 should get the fill (default -inf)"
    np.testing.assert_allclose(out[0, 1], np.log(4.0))
    np.testing.assert_allclose(out[1, 1], np.log(1e-300))


def test_safe_log_int_input(sol):
    out = sol.safe_log(np.array([1, 0, 100]), fill=0.0)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [0.0, 0.0, np.log(100)])


# ---------------------------------------------------------------- running_max

def test_running_max_example(sol, no_loops):
    out = sol.running_max(np.array([[1, 3, 2], [5, 4, 6]]), axis=1)
    np.testing.assert_array_equal(out, [[1, 3, 3], [5, 5, 6]])
    no_loops(sol.running_max)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_running_max_axes(sol, axis):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 6)).astype(np.float32)
    out = sol.running_max(x, axis=axis)
    assert out.shape == x.shape and out.dtype == np.float32
    xs = np.moveaxis(x, axis, 0)
    run = np.empty_like(xs)
    run[0] = xs[0]
    for i in range(1, xs.shape[0]):
        run[i] = np.maximum(run[i - 1], xs[i])
    expected = np.moveaxis(run, 0, axis)
    np.testing.assert_array_equal(out, expected)


def test_running_max_default_axis(sol):
    np.testing.assert_array_equal(sol.running_max(np.array([2, 1, 4, 3])), [2, 2, 4, 4])


# ---------------------------------------------------------------- gcd_all

def test_gcd_all(sol, no_loops):
    out = sol.gcd_all(np.array([[12, 18], [30, -42]]))
    assert type(out) is int, f"return a Python int, got {type(out).__name__}"
    assert out == 6
    assert sol.gcd_all(np.array([7, 13])) == 1
    assert sol.gcd_all(np.array([0, 0])) == 0
    assert sol.gcd_all(np.array([0, 15])) == 15
    no_loops(sol.gcd_all)


# ---------------------------------------------------------------- pairs_within

def test_pairs_within_example(sol, no_loops):
    out = sol.pairs_within(np.array([1.0, 5.0]), np.array([1.5, 4.0, 9.0]), 1.0)
    assert out.dtype == np.bool_, f"expected a boolean table, got {out.dtype}"
    np.testing.assert_array_equal(out, [[True, False, False], [False, True, False]])
    no_loops(sol.pairs_within)


def test_pairs_within_random(sol):
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 10, 7)
    b = rng.uniform(0, 10, 4)
    out = sol.pairs_within(a, b, 2.0)
    assert out.shape == (7, 4), f"expected (len(a), len(b)) = (7, 4), got {out.shape}"
    expected = np.array([[abs(x - y) <= 2.0 for y in b] for x in a])
    np.testing.assert_array_equal(out, expected)
