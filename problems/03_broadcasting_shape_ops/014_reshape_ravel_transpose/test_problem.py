import numpy as np
import pytest


# ---------------------------------------------------------------- to_patches

def test_to_patches_example(sol, no_loops):
    img = np.arange(16).reshape(4, 4)
    out = sol.to_patches(img, 2)
    assert out.shape == (4, 2, 2), f"expected (4, 2, 2), got {out.shape}"
    np.testing.assert_array_equal(out[0], [[0, 1], [4, 5]])
    np.testing.assert_array_equal(out[1], [[2, 3], [6, 7]],
                                  err_msg="patch 1 is the top-right block; a plain reshape gives [[4,5],[6,7]]")
    np.testing.assert_array_equal(out[2], [[8, 9], [12, 13]])
    no_loops(sol.to_patches)


def test_to_patches_rectangular(sol):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(6, 9), dtype=np.uint8)
    out = sol.to_patches(img, 3)
    assert out.shape == (6, 3, 3) and out.dtype == np.uint8
    k = 0
    for r in range(0, 6, 3):
        for c in range(0, 9, 3):
            np.testing.assert_array_equal(out[k], img[r:r + 3, c:c + 3], err_msg=f"patch {k} is wrong")
            k += 1


# ---------------------------------------------------------------- hwc_to_chw

def test_hwc_to_chw_3d(sol, no_loops):
    rng = np.random.default_rng(0)
    img = rng.standard_normal((4, 5, 3))
    out = sol.hwc_to_chw(img)
    assert out.shape == (3, 4, 5), f"expected (3, 4, 5), got {out.shape}"
    for c in range(3):
        np.testing.assert_array_equal(out[c], img[..., c], err_msg=f"channel {c} is wrong")
    assert np.shares_memory(out, img), "hwc_to_chw should return a view, not a copy"
    no_loops(sol.hwc_to_chw)


def test_hwc_to_chw_4d(sol):
    img = np.arange(2 * 4 * 5 * 3).reshape(2, 4, 5, 3)
    out = sol.hwc_to_chw(img)
    assert out.shape == (2, 3, 4, 5), f"expected (2, 3, 4, 5), got {out.shape}"
    np.testing.assert_array_equal(out, img.transpose(0, 3, 1, 2))
    assert np.shares_memory(out, img)


# ---------------------------------------------------------------- flatten_batch

def test_flatten_batch_values(sol, no_loops):
    x = np.arange(5 * 3 * 4).reshape(5, 3, 4)
    out = sol.flatten_batch(x)
    assert out.shape == (5, 12), f"expected (5, 12), got {out.shape}"
    np.testing.assert_array_equal(out[2], x[2].ravel())
    no_loops(sol.flatten_batch)


def test_flatten_batch_empty_batch(sol):
    out = sol.flatten_batch(np.zeros((0, 3, 4), dtype=np.float32))
    assert out.shape == (0, 12), f"B == 0 should give (0, 12), got {out.shape}"
    assert out.dtype == np.float32


def test_flatten_batch_1d_and_2d(sol):
    assert sol.flatten_batch(np.arange(4)).shape == (4, 1), "1-D input (B,) should give (B, 1)"
    x = np.ones((3, 7))
    assert sol.flatten_batch(x).shape == (3, 7)


# ---------------------------------------------------------------- increment_flat

def test_increment_flat_contiguous(sol):
    a = np.zeros((2, 3))
    ret = sol.increment_flat(a, 4, 2.5)
    assert ret is None, "increment_flat works in place and returns None"
    expected = np.zeros((2, 3))
    expected[1, 1] = 2.5
    np.testing.assert_array_equal(a, expected)


def test_increment_flat_non_contiguous(sol):
    base = np.zeros((2, 3))
    a = base.T  # shape (3, 2), not C-contiguous
    sol.increment_flat(a, 1, 5.0)
    assert a[0, 1] == 5.0, (
        "the change didn't reach `a`; a.ravel() makes a copy for non-contiguous arrays"
    )
    assert base[1, 0] == 5.0, "the change should also be visible in the base array"
    assert a.sum() == 5.0, "only one element should change"


def test_increment_flat_strided_view_repeated(sol):
    base = np.arange(20).reshape(4, 5)
    a = base[::2, 1::2]  # shape (2, 2): [[1, 3], [11, 13]]
    for i in range(4):
        sol.increment_flat(a, i, 100)
    np.testing.assert_array_equal(a, [[101, 103], [111, 113]])
    assert base[0, 0] == 0 and base[0, 2] == 2, "elements outside the view must not change"
