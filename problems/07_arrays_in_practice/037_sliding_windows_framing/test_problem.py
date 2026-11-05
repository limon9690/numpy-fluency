import numpy as np
import pytest


# ---------------------------------------------------------------- frames

def test_frames_example(sol, no_loops):
    x = np.arange(7)
    out = sol.frames(x, 3, 2)
    np.testing.assert_array_equal(out, [[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    assert np.shares_memory(out, x), "frames should be a view of the signal (no copy)"
    assert not out.flags.writeable, "the frames view should be read-only"
    no_loops(sol.frames)


@pytest.mark.parametrize("n, L, hop", [(100, 20, 10), (101, 20, 10), (50, 50, 7), (30, 4, 1), (30, 5, 6)])
def test_frames_counts(sol, n, L, hop):
    x = np.random.default_rng(0).standard_normal(n)
    out = sol.frames(x, L, hop)
    n_frames = 1 + (n - L) // hop
    assert out.shape == (n_frames, L), f"n={n}, L={L}, hop={hop}: expected {(n_frames, L)}, got {out.shape}"
    for i in range(n_frames):
        np.testing.assert_array_equal(out[i], x[i * hop:i * hop + L])


# ---------------------------------------------------------------- moving_max

def test_moving_max(sol, no_loops):
    out = sol.moving_max(np.array([1, 3, 2, 5, 4, 1]), 3)
    np.testing.assert_array_equal(out, [3, 5, 5, 5])
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40).astype(np.float32)
    out = sol.moving_max(x, 5)
    assert out.dtype == np.float32 and out.shape == (36,)
    np.testing.assert_array_equal(out, [x[i:i + 5].max() for i in range(36)])
    no_loops(sol.moving_max)


# ---------------------------------------------------------------- patches_2d

def test_patches_2d(sol, no_loops):
    img = np.arange(16).reshape(4, 4)
    out = sol.patches_2d(img, 2)
    assert out.shape == (3, 3, 2, 2), f"expected (H-k+1, W-k+1, k, k) = (3, 3, 2, 2), got {out.shape}"
    np.testing.assert_array_equal(out[0, 1], [[1, 2], [5, 6]])
    assert np.shares_memory(out, img) and not out.flags.writeable, "patches should be a read-only view"
    no_loops(sol.patches_2d)


def test_patches_2d_rect(sol):
    img = np.random.default_rng(0).integers(0, 256, size=(5, 7), dtype=np.uint8)
    out = sol.patches_2d(img, 3)
    assert out.shape == (3, 5, 3, 3)
    for i in range(3):
        for j in range(5):
            np.testing.assert_array_equal(out[i, j], img[i:i + 3, j:j + 3])


# ---------------------------------------------------------------- box_blur_valid

def test_box_blur_valid_example(sol, no_loops):
    out = sol.box_blur_valid(np.arange(16).reshape(4, 4), 2)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[2.5, 3.5, 4.5], [6.5, 7.5, 8.5], [10.5, 11.5, 12.5]])
    no_loops(sol.box_blur_valid)


def test_box_blur_valid_uint8(sol):
    img = np.full((6, 6), 250, dtype=np.uint8)
    img[0, 0] = 10
    out = sol.box_blur_valid(img, 3)
    assert out.shape == (4, 4)
    assert out[1, 1] == 250.0, "uint8 sums must not overflow"
    assert out[0, 0] == pytest.approx((250 * 8 + 10) / 9)
