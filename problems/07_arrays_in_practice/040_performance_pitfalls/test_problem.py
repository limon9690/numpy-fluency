import tracemalloc

import numpy as np
import pytest


def _peak_bytes(func, *args):
    """Run func(*args) and return (result, peak bytes allocated while it ran)."""
    tracemalloc.start()
    try:
        result = func(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


# ---------------------------------------------------------------- pairwise_sq_dist_fast

def test_pairwise_sq_dist_fast_values(sol, no_loops):
    out = sol.pairwise_sq_dist_fast(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[25.0, 1.0]], atol=1e-12)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    Y = rng.standard_normal((15, 5))
    out = sol.pairwise_sq_dist_fast(X, Y)
    assert out.shape == (20, 15)
    np.testing.assert_allclose(out, ((X[:, None] - Y[None]) ** 2).sum(-1), atol=1e-10)
    no_loops(sol.pairwise_sq_dist_fast)


def test_pairwise_sq_dist_fast_non_negative(sol):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 8)) * 1e3
    D = sol.pairwise_sq_dist_fast(X, X)  # the diagonal is 0 up to rounding
    assert (D >= 0).all(), "clip at 0: rounding in the expanded formula can give tiny negative values"
    np.testing.assert_allclose(np.diag(D), 0, atol=1e-6 * 1e6)


def test_pairwise_sq_dist_fast_memory(sol):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 64))
    Y = rng.standard_normal((300, 64))
    full_broadcast = 300 * 300 * 64 * 8  # ~46 MB for the (n, m, d) intermediate
    out, peak = _peak_bytes(sol.pairwise_sq_dist_fast, X, Y)
    assert out.shape == (300, 300)
    assert peak < full_broadcast / 4, (
        f"peak memory {peak / 1e6:.1f} MB: an (n, m, d) array was built ({full_broadcast / 1e6:.0f} MB); "
        "use |x|^2 + |y|^2 - 2 x.y"
    )


# ---------------------------------------------------------------- standardize_inplace_f32

def test_standardize_inplace_f32_values(sol, no_loops):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert sol.standardize_inplace_f32(x) is None, "work in place and return None"
    assert x.dtype == np.float32
    np.testing.assert_allclose(x, [-1.2247449, 0.0, 1.2247449], rtol=1e-5, atol=1e-6)
    no_loops(sol.standardize_inplace_f32)


def test_standardize_inplace_f32_memory(sol):
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(1000, 1000)).astype(np.float32)  # 4 MB
    expected = ((x.astype(np.float64) - x.mean(dtype=np.float64)) / x.std(dtype=np.float64))
    _, peak = _peak_bytes(sol.standardize_inplace_f32, x)
    assert x.dtype == np.float32, "x must stay float32"
    np.testing.assert_allclose(x, expected, rtol=1e-3, atol=1e-4)
    assert peak < x.nbytes / 4, (
        f"peak extra memory {peak / 1e6:.2f} MB for a {x.nbytes / 1e6:.0f} MB array: "
        "avoid full-size temporaries (x - m, x.std(), astype)"
    )


# ---------------------------------------------------------------- lookup

def _ban_fake_vectorization(ban):
    ban(np, "vectorize")
    ban(np, "frompyfunc")
    ban(np, "apply_along_axis")


def test_lookup_example(sol, ban, no_loops):
    _ban_fake_vectorization(ban)
    out = sol.lookup(np.array([5, 1, 9]), np.array([1, 5, 3]), np.array([10.0, 50.0, 30.0]), -1.0)
    np.testing.assert_array_equal(out, [50.0, 10.0, -1.0])
    assert out.dtype == np.float64
    no_loops(sol.lookup)


def test_lookup_random_2d(sol, ban):
    _ban_fake_vectorization(ban)
    rng = np.random.default_rng(0)
    table_keys = rng.permutation(1000)[:300]
    table_values = rng.integers(0, 10_000, size=300)
    keys = rng.integers(-10, 1010, size=(40, 25))
    out = sol.lookup(keys, table_keys, table_values, -1)
    assert out.shape == keys.shape, f"result should have the shape of keys {keys.shape}, got {out.shape}"
    assert out.dtype == table_values.dtype
    table = dict(zip(table_keys.tolist(), table_values.tolist()))
    expected = np.array([[table.get(k, -1) for k in row] for row in keys.tolist()])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- channel_means_uint8

def test_channel_means_uint8_values(sol, no_loops):
    imgs = np.zeros((2, 4, 4, 3), np.uint8)
    imgs[..., 2] = 255
    out = sol.channel_means_uint8(imgs)
    assert out.dtype == np.float64 and out.shape == (3,)
    np.testing.assert_allclose(out, [0.0, 0.0, 255.0], err_msg="sums of uint8 must not overflow")
    no_loops(sol.channel_means_uint8)


def test_channel_means_uint8_memory(sol):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(50, 100, 100, 3), dtype=np.uint8)  # 1.5 MB
    expected = imgs.reshape(-1, 3).astype(np.float64).mean(axis=0)
    out, peak = _peak_bytes(sol.channel_means_uint8, imgs)
    np.testing.assert_allclose(out, expected)
    assert peak < imgs.nbytes / 2, (
        f"peak extra memory {peak / 1e6:.2f} MB for a {imgs.nbytes / 1e6:.1f} MB batch: "
        "don't convert the whole batch to float; pass dtype= to the reduction"
    )
