import numpy as np
import pytest


# ---------------------------------------------------------------- batched_matvec

def test_batched_matvec_example(sol, no_loops):
    A = np.stack([np.eye(2), 2 * np.eye(2)])
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = sol.batched_matvec(A, x)
    np.testing.assert_allclose(out, [[1.0, 2.0], [6.0, 8.0]])
    no_loops(sol.batched_matvec)


def test_batched_matvec_random_rect(sol):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3, 4))
    x = rng.standard_normal((5, 4))
    out = sol.batched_matvec(A, x)
    assert out.shape == (5, 3), f"expected (B, n) = (5, 3), got {out.shape}"
    np.testing.assert_allclose(out, np.stack([A[b] @ x[b] for b in range(5)]))


def test_batched_matvec_square_batch(sol):
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3, 3))  # B == m: a wrong `A @ x` runs without error here
    x = rng.standard_normal((3, 3))
    np.testing.assert_allclose(sol.batched_matvec(A, x), np.stack([A[b] @ x[b] for b in range(3)]),
                               err_msg="A @ x treats x as one matrix; each item needs its own vector")


# ---------------------------------------------------------------- bilinear

def test_bilinear_example(sol, no_loops):
    out = sol.bilinear(np.array([[1.0, 0.0]]), np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(out, [3.0])
    no_loops(sol.bilinear)


def test_bilinear_random(sol):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 3))
    W = rng.standard_normal((3, 5))
    y = rng.standard_normal((6, 5))
    out = sol.bilinear(x, W, y)
    assert out.shape == (6,), f"expected (B,) = (6,), got {out.shape}"
    np.testing.assert_allclose(out, [x[b] @ W @ y[b] for b in range(6)])


# ---------------------------------------------------------------- batch_trace

def test_batch_trace(sol, ban, no_loops):
    A = np.stack([np.eye(3), np.arange(9.0).reshape(3, 3)])
    ban(np, "trace")
    out = sol.batch_trace(A)
    np.testing.assert_allclose(out, [3.0, 12.0])
    no_loops(sol.batch_trace)


def test_batch_trace_int(sol, ban):
    rng = np.random.default_rng(0)
    A = rng.integers(-5, 5, size=(7, 4, 4))
    expected = [int(np.diag(a).sum()) for a in A]
    ban(np, "trace")
    out = sol.batch_trace(A)
    assert out.shape == (7,)
    assert np.issubdtype(out.dtype, np.integer), f"int input should stay integer, got {out.dtype}"
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- attention_scores

def test_attention_scores_example(sol, no_loops):
    out = sol.attention_scores(np.ones((1, 2, 4)), np.ones((1, 3, 4)))
    assert out.shape == (1, 2, 3), f"expected (B, T, S) = (1, 2, 3), got {out.shape}"
    np.testing.assert_allclose(out, 2.0)
    no_loops(sol.attention_scores)


def test_attention_scores_random(sol):
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((2, 5, 8))
    K = rng.standard_normal((2, 7, 8))
    out = sol.attention_scores(Q, K)
    assert out.shape == (2, 5, 7)
    expected = np.array([[[Q[b, t] @ K[b, s] for s in range(7)] for t in range(5)] for b in range(2)]) / np.sqrt(8)
    np.testing.assert_allclose(out, expected)
