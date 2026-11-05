import warnings

import numpy as np
import pytest


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


# ---------------------------------------------------------------- logsumexp

def test_logsumexp_extremes(sol, no_loops):
    out = _quiet(sol.logsumexp, np.array([[1000.0, 1000.0], [-1000.0, -1000.0]]), axis=1)
    assert out.shape == (2,), f"the reduced axis should be removed, got shape {out.shape}"
    np.testing.assert_allclose(out, [1000 + np.log(2), -1000 + np.log(2)])
    no_loops(sol.logsumexp)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_logsumexp_random(sol, axis):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6)) * 3
    out = _quiet(sol.logsumexp, x, axis=axis)
    np.testing.assert_allclose(out, np.log(np.exp(x).sum(axis=axis)))


def test_logsumexp_all_neg_inf(sol):
    x = np.array([[-np.inf, -np.inf], [0.0, -np.inf]])
    out = _quiet(sol.logsumexp, x, axis=1)
    assert out[0] == -np.inf, f"a row of all -inf (all probabilities 0) should give -inf, got {out[0]}"
    assert out[1] == pytest.approx(0.0)


# ---------------------------------------------------------------- softplus

def test_softplus(sol, no_loops):
    x = np.array([-40.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
    out = _quiet(sol.softplus, x)
    assert out.dtype == np.float64
    assert out[-1] == 1000.0, "softplus(1000) should be 1000 (no overflow)"
    np.testing.assert_allclose(out[0], np.exp(-40.0), rtol=1e-10,
                               err_msg="softplus(-40) lost all digits; use log1p")
    np.testing.assert_allclose(out[1:5], np.log1p(np.exp(x[1:5])))
    no_loops(sol.softplus)


def test_softplus_shape(sol):
    x = np.linspace(-5, 5, 12).reshape(3, 4)
    out = _quiet(sol.softplus, x)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, np.logaddexp(0, x))


# ---------------------------------------------------------------- quadratic_roots

def test_quadratic_roots_simple(sol, no_loops):
    lo, hi = _quiet(sol.quadratic_roots, np.array([1.0]), np.array([-3.0]), np.array([2.0]))
    np.testing.assert_allclose(lo, [1.0])
    np.testing.assert_allclose(hi, [2.0])
    no_loops(sol.quadratic_roots)


def test_quadratic_roots_tiny_root(sol):
    r1, r2 = 1e-9, 1e8
    a = np.array([1.0, 2.0])
    b = -a * (r1 + r2)
    c = a * r1 * r2
    lo, hi = _quiet(sol.quadratic_roots, a, b, c)
    np.testing.assert_allclose(lo, [r1, r1], rtol=1e-10,
                               err_msg="the small root lost its digits: catastrophic cancellation")
    np.testing.assert_allclose(hi, [r2, r2], rtol=1e-10)


def test_quadratic_roots_signs_and_broadcast(sol):
    rng = np.random.default_rng(0)
    roots = rng.uniform(-10, 10, size=(20, 2))
    roots[:5, 0] *= 1e-7  # some tiny roots of either sign
    a = rng.uniform(0.5, 2, size=20) * rng.choice([-1, 1], size=20)
    b = -a * roots.sum(axis=1)
    c = a * roots.prod(axis=1)
    lo, hi = _quiet(sol.quadratic_roots, a, b, c)
    np.testing.assert_allclose(lo, roots.min(axis=1), rtol=1e-8)
    np.testing.assert_allclose(hi, roots.max(axis=1), rtol=1e-8)
    lo, hi = _quiet(sol.quadratic_roots, 1.0, np.array([0.0, -5.0]), np.array([-4.0, 6.0]))
    np.testing.assert_allclose(lo, [-2.0, 2.0], err_msg="b == 0 must work too (careful with np.sign(0))")
    np.testing.assert_allclose(hi, [2.0, 3.0])


# ---------------------------------------------------------------- count_mismatches

def test_count_mismatches_example(sol, no_loops):
    out = sol.count_mismatches(np.array([1.0, 2.0, np.nan, 0.0]), np.array([1.0 + 1e-9, 2.1, np.nan, 1e-9]))
    assert type(out) is int, f"return a Python int, got {type(out).__name__}"
    assert out == 1, "only 2.0 vs 2.1 differs; NaN vs NaN counts as equal"
    no_loops(sol.count_mismatches)


def test_count_mismatches_tolerances(sol):
    rng = np.random.default_rng(0)
    b = rng.standard_normal(500)
    a = b + rng.normal(0, 1e-4, 500)
    for rtol, atol in [(1e-5, 1e-8), (1e-3, 0.0), (0.0, 1e-4)]:
        expected = int((~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)).sum())
        assert sol.count_mismatches(a, b, rtol=rtol, atol=atol) == expected, f"wrong count for rtol={rtol}, atol={atol}"
