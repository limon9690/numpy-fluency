import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------- softmax

def test_softmax_example_and_large_values(sol, no_loops):
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sol.softmax(x, axis=1)
    e = np.exp([1.0, 2.0, 3.0])
    np.testing.assert_allclose(out[0], e / e.sum())
    np.testing.assert_allclose(out[1], [1 / 3] * 3, err_msg="large inputs overflowed; subtract the max first")
    no_loops(sol.softmax)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_axes(sol, axis):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5)) * 10
    out = sol.softmax(x, axis=axis)
    assert out.shape == x.shape
    np.testing.assert_allclose(out.sum(axis=axis), 1.0, err_msg=f"values along axis={axis} should sum to 1")
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    np.testing.assert_allclose(out, e / e.sum(axis=axis, keepdims=True))


def test_softmax_dtypes(sol):
    out = sol.softmax(np.array([1.0, 2.0], dtype=np.float32))
    assert out.dtype == np.float32, f"float32 input should stay float32, got {out.dtype}"
    out = sol.softmax(np.array([[0, 0, 0, 0]]))
    assert out.dtype == np.float64, f"int input should give float64, got {out.dtype}"
    np.testing.assert_allclose(out, [[0.25] * 4])


# ---------------------------------------------------------------- argmax_2d

def test_argmax_2d(sol, no_loops):
    out = sol.argmax_2d(np.array([[1, 5, 2], [7, 0, 3]]))
    assert out == (1, 0), f"expected (1, 0), got {out}"
    assert all(type(v) is int for v in out), "return Python ints, not NumPy integers"
    no_loops(sol.argmax_2d)


def test_argmax_2d_ties_and_random(sol):
    assert sol.argmax_2d(np.array([[0, 9], [9, 9]])) == (0, 1), "on ties, take the first in C order"
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 11))
    r, c = sol.argmax_2d(a)
    assert a[r, c] == a.max()


# ---------------------------------------------------------------- channel_mean_std

def test_channel_mean_std_example(sol, no_loops):
    images = np.zeros((2, 4, 4, 3), np.uint8)
    images[..., 1] = 10
    mean, std = sol.channel_mean_std(images)
    assert mean.shape == std.shape == (3,), f"expected shape (C,) = (3,), got {mean.shape}"
    assert mean.dtype == std.dtype == np.float64
    np.testing.assert_allclose(mean, [0, 10, 0])
    np.testing.assert_allclose(std, [0, 0, 0])
    no_loops(sol.channel_mean_std)


def test_channel_mean_std_random_uint8(sol):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(5, 8, 9, 4), dtype=np.uint8)
    mean, std = sol.channel_mean_std(images)
    flat = images.reshape(-1, 4).astype(np.float64)
    np.testing.assert_allclose(mean, flat.mean(axis=0))
    np.testing.assert_allclose(std, flat.std(axis=0), err_msg="use the population std (ddof=0)")


# ---------------------------------------------------------------- center_per_sample

def test_center_per_sample_example(sol, no_loops):
    out = sol.center_per_sample(np.array([[1, 3], [10, 20]]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[-1, 1], [-5, 5]])
    no_loops(sol.center_per_sample)


@pytest.mark.parametrize("shape", [(6, 4), (3, 4, 5), (2, 3, 4, 5), (5,)])
def test_center_per_sample_any_ndim(sol, shape):
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, size=shape)
    out = sol.center_per_sample(x)
    assert out.shape == shape, f"shape should stay {shape}, got {out.shape}"
    means = out.reshape(shape[0], -1).mean(axis=1)
    np.testing.assert_allclose(means, 0, atol=1e-12, err_msg="every sample should have mean 0")
    expected = x - x.reshape(shape[0], -1).mean(axis=1).reshape((-1,) + (1,) * (len(shape) - 1))
    np.testing.assert_allclose(out, expected)
