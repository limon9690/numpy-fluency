import numpy as np
import pytest


def _data(n=50, d=4, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d))
    base[:, 1] += 2 * base[:, 0]          # make some columns correlated
    base[:, 3] = -base[:, 2] + 0.1 * base[:, 3]
    return base * np.array([1.0, 10.0, 0.5, 3.0]) + np.array([0.0, 5.0, -2.0, 100.0])


# ---------------------------------------------------------------- sample_std

def test_sample_std(sol):
    assert sol.sample_std(np.array([1.0, 3.0])) == pytest.approx(np.sqrt(2.0)), "use the n - 1 divisor"
    X = _data()
    for axis in (0, 1, None):
        np.testing.assert_allclose(sol.sample_std(X, axis=axis), X.std(axis=axis, ddof=1),
                                   err_msg=f"wrong sample std for axis={axis}")


def test_sample_std_int_input(sol):
    out = sol.sample_std(np.array([[1, 2], [3, 6]]), axis=0)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [np.sqrt(2.0), np.sqrt(8.0)])


# ---------------------------------------------------------------- covariance_matrix

def test_covariance_matrix_example(sol, ban, no_loops):
    ban(np, "cov")
    out = sol.covariance_matrix(np.array([[0.0, 0.0], [2.0, 4.0]]))
    np.testing.assert_allclose(out, [[2.0, 4.0], [4.0, 8.0]])
    no_loops(sol.covariance_matrix)


def test_covariance_matrix_random(sol, ban):
    X = _data()
    expected = np.cov(X, rowvar=False)
    ban(np, "cov")
    out = sol.covariance_matrix(X)
    assert out.shape == (4, 4), f"expected (d, d) = (4, 4), got {out.shape}; columns are the variables"
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(out, out.T, err_msg="a covariance matrix is symmetric")


# ---------------------------------------------------------------- correlation_matrix

def test_correlation_matrix(sol, ban, no_loops):
    X = _data()
    expected = np.corrcoef(X, rowvar=False)
    ban(np, "cov")
    ban(np, "corrcoef")
    out = sol.correlation_matrix(X)
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(np.diag(out), 1.0, err_msg="the diagonal must be 1")
    no_loops(sol.correlation_matrix)


def test_correlation_matrix_example(sol, ban):
    ban(np, "cov")
    ban(np, "corrcoef")
    out = sol.correlation_matrix(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [3.0, 6.0, 2.0]]))
    assert out[0, 1] == pytest.approx(1.0), "column 1 is exactly 2 * column 0"


# ---------------------------------------------------------------- columnwise_corr

def test_columnwise_corr_example(sol, no_loops):
    x = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([[2.0, 3.0], [4.0, 2.0], [6.0, 1.0]])
    out = sol.columnwise_corr(x, y)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [1.0, -1.0])
    no_loops(sol.columnwise_corr)


def test_columnwise_corr_random(sol):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((80, 5))
    y = 0.5 * x + rng.standard_normal((80, 5)) * np.arange(1, 6)
    out = sol.columnwise_corr(x, y)
    expected = [np.corrcoef(x[:, j], y[:, j])[0, 1] for j in range(5)]
    np.testing.assert_allclose(out, expected)
