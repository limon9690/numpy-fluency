import numpy as np
import pytest


# ---------------------------------------------------------------- crop_center

def test_crop_center_example(sol, no_loops):
    img = np.arange(16).reshape(4, 4)
    out = sol.crop_center(img, 2, 2)
    np.testing.assert_array_equal(out, [[5, 6], [9, 10]])
    assert np.shares_memory(out, img), "the crop should be a view (basic slicing), not a copy"
    no_loops(sol.crop_center)


def test_crop_center_odd_sizes_and_channels(sol):
    img = np.arange(7 * 6 * 3).reshape(7, 6, 3)
    out = sol.crop_center(img, 3, 3)
    assert out.shape == (3, 3, 3), f"expected shape (3, 3, 3), got {out.shape}"
    np.testing.assert_array_equal(out, img[2:5, 1:4])
    np.testing.assert_array_equal(sol.crop_center(img, 7, 6), img)


# ---------------------------------------------------------------- flip_axis

def test_flip_axis_matches_np_flip(sol, ban, no_loops):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, size=(3, 4, 5))
    expected = {ax: np.flip(a, ax) for ax in range(-3, 3)}
    ban(np, "flip")
    for ax in range(-3, 3):
        out = sol.flip_axis(a, ax)
        np.testing.assert_array_equal(out, expected[ax], err_msg=f"wrong result for axis={ax}")
        assert np.shares_memory(out, a), f"axis={ax}: the result should be a view of `a`"
    no_loops(sol.flip_axis)


def test_flip_axis_1d(sol, ban):
    ban(np, "flip")
    np.testing.assert_array_equal(sol.flip_axis(np.arange(4), 0), [3, 2, 1, 0])


# ---------------------------------------------------------------- checkerboard

@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_checkerboard(sol, n):
    out = sol.checkerboard(n)
    assert out.shape == (n, n), f"expected shape ({n}, {n}), got {out.shape}"
    assert out.dtype == np.int8, f"expected int8, got {out.dtype}"
    i, j = np.indices((n, n))
    np.testing.assert_array_equal(out, (i + j) % 2, err_msg="top-left must be 0, and neighbours must differ")


def test_checkerboard_no_loops(sol, no_loops):
    sol.checkerboard(4)
    no_loops(sol.checkerboard)


# ---------------------------------------------------------------- last_k

def test_last_k_basic(sol, no_loops):
    a = np.arange(10).reshape(5, 2)
    out = sol.last_k(a, 2)
    np.testing.assert_array_equal(out, [[6, 7], [8, 9]])
    assert np.shares_memory(out, a), "the result should be a view"
    np.testing.assert_array_equal(sol.last_k(a, 5), a)
    no_loops(sol.last_k)


def test_last_k_zero_is_empty(sol):
    a = np.arange(10).reshape(5, 2)
    out = sol.last_k(a, 0)
    assert out.shape == (0, 2), f"k=0 should give shape (0, 2), got {out.shape} (a[-0:] is the whole array!)"
