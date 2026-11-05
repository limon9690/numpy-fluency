import numpy as np
import pytest


# ---------------------------------------------------------------- ensure_batch_image

def test_ensure_batch_image_2d(sol):
    img = np.arange(12).reshape(3, 4)
    out = sol.ensure_batch_image(img)
    assert out.shape == (1, 3, 4, 1), f"(H, W) should become (1, H, W, 1), got {out.shape}"
    np.testing.assert_array_equal(out[0, ..., 0], img)
    assert np.shares_memory(out, img), "return a view, not a copy"


def test_ensure_batch_image_3d_and_4d(sol):
    img = np.zeros((5, 6, 3), dtype=np.uint8)
    out = sol.ensure_batch_image(img)
    assert out.shape == (1, 5, 6, 3), f"(H, W, C) should become (1, H, W, C), got {out.shape}"
    assert out.dtype == np.uint8 and np.shares_memory(out, img)
    batch = np.zeros((2, 5, 6, 3))
    out = sol.ensure_batch_image(batch)
    assert out.shape == (2, 5, 6, 3) and np.shares_memory(out, batch), "4-D input should come back unchanged"


@pytest.mark.parametrize("shape", [(5,), (1, 2, 3, 4, 5), ()])
def test_ensure_batch_image_bad_ndim(sol, shape):
    with pytest.raises(ValueError):
        sol.ensure_batch_image(np.zeros(shape))


# ---------------------------------------------------------------- squeeze_except

@pytest.mark.parametrize("shape, axis, expected", [
    ((1, 3, 1, 5), 0, (1, 3, 5)),
    ((1, 3, 1, 5), 2, (3, 1, 5)),
    ((1, 3, 1, 5), -2, (3, 1, 5)),
    ((1, 3, 1, 5), 1, (3, 5)),
    ((1, 1, 1), -1, (1,)),
    ((4, 5), 0, (4, 5)),
])
def test_squeeze_except(sol, shape, axis, expected):
    x = np.arange(np.prod(shape)).reshape(shape)
    out = sol.squeeze_except(x, axis)
    assert out.shape == expected, f"squeeze_except(shape {shape}, axis={axis}): expected {expected}, got {out.shape}"
    np.testing.assert_array_equal(out.ravel(), x.ravel())
    assert np.shares_memory(out, x), "return a view"


# ---------------------------------------------------------------- broadcast_along

def test_broadcast_along_shapes(sol, no_loops):
    b = np.array([1.0, 2.0, 3.0])
    out = sol.broadcast_along(b, 4, 1)
    assert out.shape == (1, 3, 1, 1), f"expected (1, 3, 1, 1), got {out.shape}"
    assert np.shares_memory(out, b), "return a view of v"
    assert sol.broadcast_along(b, 3, -1).shape == (1, 1, 3)
    assert sol.broadcast_along(b, 1, 0).shape == (3,)
    assert sol.broadcast_along(b, 3, 0).shape == (3, 1, 1)
    no_loops(sol.broadcast_along)


def test_broadcast_along_adds_per_channel(sol):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4, 5))
    for axis in range(4):
        v = np.arange(x.shape[axis], dtype=np.float64) * 10
        out = x + sol.broadcast_along(v, 4, axis)
        idx = [None] * 4
        idx[axis] = slice(None)
        np.testing.assert_allclose(out, x + v[tuple(idx)], err_msg=f"wrong alignment for axis={axis}")
