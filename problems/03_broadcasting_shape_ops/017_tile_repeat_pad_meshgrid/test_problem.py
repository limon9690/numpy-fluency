import numpy as np
import pytest


# ---------------------------------------------------------------- upsample_nearest

def test_upsample_nearest_example(sol, no_loops):
    out = sol.upsample_nearest(np.array([[1, 2], [3, 4]]), 2)
    np.testing.assert_array_equal(out, [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    no_loops(sol.upsample_nearest)


def test_upsample_nearest_rgb(sol):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    out = sol.upsample_nearest(img, 3)
    assert out.shape == (9, 12, 3), f"expected (9, 12, 3), got {out.shape}"
    assert out.dtype == np.uint8
    i, j = np.indices((9, 12))
    np.testing.assert_array_equal(out, img[i // 3, j // 3], err_msg="out[i, j] should equal img[i // k, j // k]")


def test_upsample_nearest_k1(sol):
    img = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(sol.upsample_nearest(img, 1), img)


# ---------------------------------------------------------------- pad_to_multiple

def test_pad_to_multiple_2d(sol, no_loops):
    img = np.arange(1, 16).reshape(3, 5)
    out = sol.pad_to_multiple(img, 4)
    assert out.shape == (4, 8), f"expected (4, 8), got {out.shape}"
    np.testing.assert_array_equal(out[:3, :5], img, err_msg="the original pixels should stay at the top-left")
    assert out[3:].sum() == 0 and out[:, 5:].sum() == 0, "padding should be zeros at the bottom and right"
    no_loops(sol.pad_to_multiple)


def test_pad_to_multiple_rgb(sol):
    img = np.ones((5, 6, 3), dtype=np.uint8)
    out = sol.pad_to_multiple(img, 4)
    assert out.shape == (8, 8, 3), f"expected (8, 8, 3) (channels not padded), got {out.shape}"
    assert out.dtype == np.uint8
    assert out.sum() == 5 * 6 * 3


def test_pad_to_multiple_already_multiple(sol):
    img = np.ones((8, 12, 3))
    out = sol.pad_to_multiple(img, 4)
    assert out.shape == (8, 12, 3), "no padding is needed when H and W are already multiples"
    np.testing.assert_array_equal(out, img)


# ---------------------------------------------------------------- coordinate_grid

def test_coordinate_grid_example(sol, no_loops):
    out = sol.coordinate_grid(2, 3)
    assert out.shape == (2, 3, 2), f"expected (2, 3, 2), got {out.shape}"
    assert np.issubdtype(out.dtype, np.integer), f"expected an integer dtype, got {out.dtype}"
    np.testing.assert_array_equal(out[..., 0], [[0, 0, 0], [1, 1, 1]], err_msg="channel 0 should be the row index")
    np.testing.assert_array_equal(out[..., 1], [[0, 1, 2], [0, 1, 2]], err_msg="channel 1 should be the column index")
    no_loops(sol.coordinate_grid)


def test_coordinate_grid_non_square(sol):
    out = sol.coordinate_grid(5, 7)
    assert out.shape == (5, 7, 2), f"expected (5, 7, 2), got {out.shape}; meshgrid's default 'xy' swaps axes"
    for i, j in [(0, 6), (4, 0), (3, 5)]:
        assert tuple(out[i, j]) == (i, j), f"out[{i}, {j}] should be ({i}, {j}), got {tuple(out[i, j])}"


# ---------------------------------------------------------------- repeat_and_cycle

def test_repeat_and_cycle(sol, no_loops):
    each, cycled = sol.repeat_and_cycle(np.array([0, 1, 2]), 2)
    np.testing.assert_array_equal(each, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(cycled, [0, 1, 2, 0, 1, 2])
    no_loops(sol.repeat_and_cycle)


def test_repeat_and_cycle_dtype(sol):
    labels = np.array([7, 3], dtype=np.int16)
    each, cycled = sol.repeat_and_cycle(labels, 3)
    assert each.dtype == cycled.dtype == np.int16
    assert each.shape == cycled.shape == (6,)
    np.testing.assert_array_equal(each, [7, 7, 7, 3, 3, 3])
    np.testing.assert_array_equal(cycled, [7, 3, 7, 3, 7, 3])
