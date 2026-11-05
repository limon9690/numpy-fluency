import numpy as np
import pytest


def _well_conditioned(n, rng, batch=()):
    """Random matrices that are safely invertible (diagonally dominant)."""
    A = rng.standard_normal(batch + (n, n))
    return A + n * np.eye(n)


def _hilbert(n):
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


# ---------------------------------------------------------------- solve_many

def test_solve_many_example(sol, ban, no_loops):
    ban(np.linalg, "inv")
    out = sol.solve_many(np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([[9.0, 3.0], [8.0, 1.0]]))
    np.testing.assert_allclose(out, [[2.0, 1.0], [3.0, 0.0]], atol=1e-12)
    no_loops(sol.solve_many)


def test_solve_many_random(sol, ban):
    ban(np.linalg, "inv")
    rng = np.random.default_rng(0)
    A = _well_conditioned(6, rng)
    B = rng.standard_normal((6, 4))
    X = sol.solve_many(A, B)
    assert X.shape == (6, 4), f"expected (n, k) = (6, 4), got {X.shape}"
    np.testing.assert_allclose(A @ X, B, atol=1e-10)


# ---------------------------------------------------------------- batched_solve

def test_batched_solve_example(sol, ban, no_loops):
    ban(np.linalg, "inv")
    A = np.stack([np.eye(2), 2 * np.eye(2)])
    out = sol.batched_solve(A, np.array([[1.0, 2.0], [2.0, 4.0]]))
    np.testing.assert_allclose(out, [[1.0, 2.0], [1.0, 2.0]])
    no_loops(sol.batched_solve)


@pytest.mark.parametrize("N, n", [(5, 3), (4, 4)])
def test_batched_solve_random(sol, ban, N, n):
    ban(np.linalg, "inv")
    rng = np.random.default_rng(N + n)
    A = _well_conditioned(n, rng, (N,))
    b = rng.standard_normal((N, n))
    x = sol.batched_solve(A, b)
    assert x.shape == (N, n), f"expected one solution vector per system, shape {(N, n)}, got {x.shape}"
    for i in range(N):
        np.testing.assert_allclose(A[i] @ x[i], b[i], atol=1e-10, err_msg=f"system {i} is not solved")


# ---------------------------------------------------------------- solve_checked

def test_solve_checked_ok(sol, ban):
    ban(np.linalg, "inv")
    rng = np.random.default_rng(0)
    A = _well_conditioned(5, rng)
    b = rng.standard_normal(5)
    x = sol.solve_checked(A, b)
    np.testing.assert_allclose(A @ x, b, atol=1e-10)


def test_solve_checked_rejects_singular_and_ill_conditioned(sol, ban):
    ban(np.linalg, "inv")
    with pytest.raises(ValueError):
        sol.solve_checked(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([1.0, 1.0]))
    H = _hilbert(10)  # cond ~ 1.6e13
    with pytest.raises(ValueError):
        sol.solve_checked(H, H @ np.ones(10), max_cond=1e10)
    x = sol.solve_checked(H, H @ np.ones(10), max_cond=1e14)
    assert x.shape == (10,), "with a high enough max_cond the system should be solved"


# ---------------------------------------------------------------- relative_residual

def test_relative_residual(sol):
    out = sol.relative_residual(np.eye(2), np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert type(out) is float, f"return a Python float, got {type(out).__name__}"
    assert out == 0.0
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    assert sol.relative_residual(A, np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(
        np.linalg.norm([2.0, -2.0]) / 2.0
    )


def test_relative_residual_solve_beats_inv(sol):
    H = _hilbert(10)
    b = H @ np.ones(10)
    r_solve = sol.relative_residual(H, np.linalg.solve(H, b), b)
    r_inv = sol.relative_residual(H, np.linalg.inv(H) @ b, b)
    assert r_solve < 1e-12, "solve should fit the equations to about machine precision"
    assert r_solve < r_inv, "on an ill-conditioned matrix, inv(A) @ b should leave a larger residual"
