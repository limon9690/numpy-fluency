import numpy as np
import pytest


# ---------------------------------------------------------------- add_bias_column

def test_add_bias_column_float32(sol, no_loops):
    X = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    out = sol.add_bias_column(X)
    assert out.shape == (2, 3), f"expected (2, 3), got {out.shape}"
    assert out.dtype == np.float32, f"dtype should stay float32, got {out.dtype} (np.ones defaults to float64)"
    np.testing.assert_array_equal(out, [[1, 2, 3], [1, 4, 5]])
    no_loops(sol.add_bias_column)


def test_add_bias_column_random(sol):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    out = sol.add_bias_column(X)
    assert out.shape == (10, 5) and out.dtype == np.float64
    np.testing.assert_array_equal(out[:, 0], np.ones(10))
    np.testing.assert_array_equal(out[:, 1:], X)


# ---------------------------------------------------------------- split_batches

def test_split_batches_example(sol, no_loops):
    x = np.arange(7)
    out = sol.split_batches(x, 3)
    assert isinstance(out, list), "return a list of arrays"
    assert [b.tolist() for b in out] == [[0, 1, 2], [3, 4, 5], [6]]
    no_loops(sol.split_batches)


def test_split_batches_views_and_exact_multiple(sol):
    x = np.arange(24).reshape(12, 2)
    out = sol.split_batches(x, 4)
    assert len(out) == 3, f"12 rows in batches of 4 should give 3 batches (no empty one), got {len(out)}"
    for k, b in enumerate(out):
        assert b.shape == (4, 2)
        np.testing.assert_array_equal(b, x[4 * k:4 * k + 4])
        assert np.shares_memory(b, x), f"batch {k} should be a view of x"


def test_split_batches_big_batch(sol):
    out = sol.split_batches(np.arange(5), 10)
    assert len(out) == 1 and out[0].tolist() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- interleave

def test_interleave_example(sol, no_loops):
    out = sol.interleave(np.array([1, 2, 3]), np.array([10, 20, 30]))
    np.testing.assert_array_equal(out, [1, 10, 2, 20, 3, 30])
    no_loops(sol.interleave)


def test_interleave_2d(sol):
    a = np.arange(8).reshape(4, 2)
    b = -np.arange(8).reshape(4, 2)
    out = sol.interleave(a, b)
    assert out.shape == (8, 2), f"expected (8, 2), got {out.shape}"
    np.testing.assert_array_equal(out[0::2], a, err_msg="even rows should be a")
    np.testing.assert_array_equal(out[1::2], b, err_msg="odd rows should be b")


# ---------------------------------------------------------------- join_channels

def test_join_channels(sol, no_loops):
    rng = np.random.default_rng(0)
    r, g, b = (rng.integers(0, 256, size=(4, 5), dtype=np.uint8) for _ in range(3))
    out = sol.join_channels(r, g, b)
    assert out.shape == (4, 5, 3), f"expected (4, 5, 3), got {out.shape}"
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., 0], r)
    np.testing.assert_array_equal(out[..., 1], g)
    np.testing.assert_array_equal(out[..., 2], b)
    no_loops(sol.join_channels)
