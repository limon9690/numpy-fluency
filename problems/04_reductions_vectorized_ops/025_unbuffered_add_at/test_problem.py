import numpy as np
import pytest


# ---------------------------------------------------------------- scatter_add

def test_scatter_add_repeats(sol, no_loops):
    target = np.zeros(4)
    out = sol.scatter_add(target, np.array([0, 2, 2, 2]), np.array([1.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, [1.0, 0.0, 6.0, 0.0],
                               err_msg="repeated indices must add up; target[idx] += values loses updates")
    np.testing.assert_array_equal(target, np.zeros(4), err_msg="target must not be modified")
    no_loops(sol.scatter_add)


def test_scatter_add_random(sol):
    rng = np.random.default_rng(0)
    target = rng.standard_normal(10)
    idx = rng.integers(0, 10, 100)
    values = rng.standard_normal(100)
    expected = target.copy()
    for i, v in zip(idx, values):
        expected[i] += v
    out = sol.scatter_add(target, idx, values)
    assert out.shape == (10,) and out.dtype == np.float64
    np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------- pixel_counts

def test_pixel_counts_example(sol, no_loops):
    out = sol.pixel_counts(np.array([0, 0, 1]), np.array([2, 2, 0]), 2, 3)
    assert out.dtype == np.int64, f"expected int64, got {out.dtype}"
    np.testing.assert_array_equal(out, [[0, 0, 2], [1, 0, 0]])
    no_loops(sol.pixel_counts)


def test_pixel_counts_random(sol):
    rng = np.random.default_rng(0)
    ys = rng.integers(0, 4, 500)
    xs = rng.integers(0, 6, 500)
    out = sol.pixel_counts(ys, xs, 4, 6)
    assert out.shape == (4, 6), f"expected (h, w) = (4, 6), got {out.shape}"
    assert out.sum() == 500, "every point should be counted once"
    expected = np.zeros((4, 6), dtype=np.int64)
    for y, x in zip(ys, xs):
        expected[y, x] += 1
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- scatter_max

def test_scatter_max_example(sol, no_loops):
    out = sol.scatter_max(3, np.array([0, 2, 0]), np.array([1.0, 5.0, 4.0]))
    assert out.dtype == np.float64 and out.shape == (3,)
    np.testing.assert_array_equal(out, [4.0, -np.inf, 5.0])
    no_loops(sol.scatter_max)


def test_scatter_max_random_negative_values(sol):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 8, 60)
    values = rng.standard_normal(60) - 10  # all negative: starting from 0 would be wrong
    out = sol.scatter_max(9, idx, values)
    expected = np.array([values[idx == j].max() if (idx == j).any() else -np.inf for j in range(9)])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- embedding_grad

def test_embedding_grad_example(sol, no_loops):
    out = sol.embedding_grad(np.array([[1, 1]]), np.array([[[1.0, 2.0], [3.0, 4.0]]]), 3)
    np.testing.assert_allclose(out, [[0, 0], [4, 6], [0, 0]],
                               err_msg="both rows of grad_out go to word 1 and must be summed")
    no_loops(sol.embedding_grad)


def test_embedding_grad_random(sol):
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 7, size=(4, 5))
    grad_out = rng.standard_normal((4, 5, 3)).astype(np.float32)
    out = sol.embedding_grad(indices, grad_out, 10)
    assert out.shape == (10, 3), f"expected (vocab_size, D) = (10, 3), got {out.shape}"
    assert out.dtype == np.float32, f"dtype should follow grad_out (float32), got {out.dtype}"
    expected = np.zeros((10, 3), dtype=np.float64)
    for b in range(4):
        for t in range(5):
            expected[indices[b, t]] += grad_out[b, t]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(out[7:], 0, err_msg="unused words should have zero gradient")
