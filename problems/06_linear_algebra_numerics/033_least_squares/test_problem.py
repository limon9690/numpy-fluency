import numpy as np
import pytest


def _ban_shortcuts(ban):
    ban(np, "polyfit")
    ban(np.linalg, "inv")


# ---------------------------------------------------------------- fit_line

def test_fit_line_exact(sol, ban, no_loops):
    _ban_shortcuts(ban)
    slope, intercept = sol.fit_line(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0]))
    assert type(slope) is float and type(intercept) is float, "return two Python floats"
    assert slope == pytest.approx(2.0) and intercept == pytest.approx(1.0)
    no_loops(sol.fit_line)


def test_fit_line_noisy(sol, ban):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    y = -0.7 * x + 4.0 + rng.normal(0, 0.5, 200)
    expected = np.polyfit(x, y, 1)
    _ban_shortcuts(ban)
    slope, intercept = sol.fit_line(x, y)
    assert slope == pytest.approx(expected[0]) and intercept == pytest.approx(expected[1]), (
        "slope/intercept don't match the least-squares fit; did you forget the column of ones?"
    )


# ---------------------------------------------------------------- fit_plane

def test_fit_plane_exact(sol, ban, no_loops):
    _ban_shortcuts(ban)
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [1.0, 1.0, 6.0]])
    out = sol.fit_plane(pts)
    assert out.shape == (3,) and out.dtype == np.float64
    np.testing.assert_allclose(out, [2.0, 3.0, 1.0], atol=1e-12)
    no_loops(sol.fit_plane)


def test_fit_plane_noisy(sol, ban):
    _ban_shortcuts(ban)
    rng = np.random.default_rng(0)
    xy = rng.uniform(-5, 5, size=(300, 2))
    z = 0.5 * xy[:, 0] - 2.0 * xy[:, 1] + 7.0 + rng.normal(0, 0.01, 300)
    out = sol.fit_plane(np.column_stack([xy, z]))
    np.testing.assert_allclose(out, [0.5, -2.0, 7.0], atol=5e-3)


# ---------------------------------------------------------------- lstsq_multi

def test_lstsq_multi_example(sol, ban, no_loops):
    _ban_shortcuts(ban)
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    X, rss = sol.lstsq_multi(A, np.array([[1.0], [1.0], [2.0]]))
    np.testing.assert_allclose(X, [[1.0], [1.0]])
    np.testing.assert_allclose(rss, [0.0], atol=1e-20)
    no_loops(sol.lstsq_multi)


def test_lstsq_multi_random(sol, ban):
    _ban_shortcuts(ban)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 3))
    B = rng.standard_normal((40, 5))
    X, rss = sol.lstsq_multi(A, B)
    assert X.shape == (3, 5) and rss.shape == (5,)
    for j in range(5):
        xj = np.linalg.lstsq(A, B[:, j], rcond=None)[0]
        np.testing.assert_allclose(X[:, j], xj, atol=1e-12)
        assert rss[j] == pytest.approx(np.sum((A @ xj - B[:, j]) ** 2))


def test_lstsq_multi_rank_deficient_still_has_rss(sol, ban):
    _ban_shortcuts(ban)
    rng = np.random.default_rng(1)
    a = rng.standard_normal((10, 1))
    A = np.hstack([a, 2 * a])  # rank 1: lstsq's own residuals come back empty
    B = rng.standard_normal((10, 2))
    X, rss = sol.lstsq_multi(A, B)
    assert rss.shape == (2,), f"rss must always have one value per column, got shape {rss.shape}"
    np.testing.assert_allclose(rss, ((A @ X - B) ** 2).sum(axis=0))


# ---------------------------------------------------------------- min_norm_solution

def test_min_norm_solution_example(sol, ban, no_loops):
    _ban_shortcuts(ban)
    out = sol.min_norm_solution(np.array([[1.0, 1.0]]), np.array([2.0]))
    np.testing.assert_allclose(out, [1.0, 1.0])
    no_loops(sol.min_norm_solution)


def test_min_norm_solution_random(sol, ban):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 7))
    b = rng.standard_normal(3)
    expected = np.linalg.pinv(A) @ b
    _ban_shortcuts(ban)
    x = sol.min_norm_solution(A, b)
    assert x.shape == (7,)
    np.testing.assert_allclose(A @ x, b, atol=1e-10, err_msg="x must solve the system exactly")
    np.testing.assert_allclose(x, expected, atol=1e-10, err_msg="x should be the smallest-norm solution")
