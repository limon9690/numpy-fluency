import numpy as np
import pytest


# ---------------------------------------------------------------- submatrix

def test_submatrix_example(sol, no_loops):
    a = np.arange(12).reshape(3, 4)
    out = sol.submatrix(a, np.array([0, 2]), np.array([1, 3]))
    assert out.shape == (2, 2), f"expected the 2x2 block, got shape {out.shape} (paired, not crossed?)"
    np.testing.assert_array_equal(out, [[1, 3], [9, 11]])
    no_loops(sol.submatrix)


def test_submatrix_order_repeats_and_masks(sol):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, size=(6, 7))
    rows, cols = np.array([5, 0, 5]), np.array([6, 2])
    out = sol.submatrix(a, rows, cols)
    expected = np.array([[a[r, c] for c in cols] for r in rows])
    np.testing.assert_array_equal(out, expected)
    rmask = np.array([True, False, True, True, False, False])
    cmask = np.array([False, True, False, False, True, True, False])
    out = sol.submatrix(a, rmask, cmask)
    np.testing.assert_array_equal(out, a[rmask][:, cmask], err_msg="boolean masks should select a block too")


# ---------------------------------------------------------------- channel_pixels

def test_channel_pixels(sol, no_loops):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    ys, xs = np.array([0, 3, 3]), np.array([1, 4, 0])
    out = sol.channel_pixels(img, ys, xs)
    assert out.shape == (3, 3), f"expected shape (3, 3), got {out.shape}"
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.stack([img[0, 1], img[3, 4], img[3, 0]]))
    no_loops(sol.channel_pixels)


# ---------------------------------------------------------------- select_frame_channels

def test_select_frame_channels_shape(sol, no_loops):
    video = np.arange(10 * 4 * 5 * 3).reshape(10, 4, 5, 3)
    out = sol.select_frame_channels(video, 2, [2, 0])
    assert out.shape == (4, 5, 2), (
        f"expected (H, W, len(channels)) = (4, 5, 2), got {out.shape}; "
        "video[t, :, :, channels] moves the channel axis to the front"
    )
    no_loops(sol.select_frame_channels)


def test_select_frame_channels_values(sol):
    rng = np.random.default_rng(0)
    video = rng.standard_normal((6, 3, 4, 5))
    channels = [4, 1, 1]
    out = sol.select_frame_channels(video, 5, channels)
    assert out.shape == (3, 4, 3)
    for j, c in enumerate(channels):
        np.testing.assert_array_equal(out[..., j], video[5, :, :, c], err_msg=f"wrong data for output channel {j}")


# ---------------------------------------------------------------- batch_gather

def test_batch_gather_example(sol, ban, no_loops):
    ban(np, "take_along_axis")
    ban(np, "take")
    x = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    idx = np.array([[3, 0], [1, 1]])
    out = sol.batch_gather(x, idx)
    assert out.shape == (2, 2, 3), f"expected shape (B, K, D) = (2, 2, 3), got {out.shape}"
    np.testing.assert_array_equal(out[0], x[0, [3, 0]])
    np.testing.assert_array_equal(out[1], x[1, [1, 1]])
    no_loops(sol.batch_gather)


def test_batch_gather_random(sol, ban):
    ban(np, "take_along_axis")
    ban(np, "take")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 9, 4)).astype(np.float32)
    idx = rng.integers(0, 9, size=(5, 6))
    out = sol.batch_gather(x, idx)
    assert out.shape == (5, 6, 4) and out.dtype == np.float32
    expected = np.stack([x[b][idx[b]] for b in range(5)])
    np.testing.assert_array_equal(out, expected)
