import numpy as np
import pytest


# ---------------------------------------------------------------- one_hot

def test_one_hot_example(sol, no_loops):
    out = sol.one_hot(np.array([1, 3, 0]), 4)
    assert out.dtype == np.float32, f"expected float32, got {out.dtype}"
    assert out.shape == (3, 4), f"expected shape (3, 4), got {out.shape}"
    np.testing.assert_array_equal(out, [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    no_loops(sol.one_hot)


def test_one_hot_random(sol):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=200)
    out = sol.one_hot(labels, 10)
    np.testing.assert_array_equal(out, np.eye(10, dtype=np.float32)[labels])
    np.testing.assert_array_equal(out.sum(axis=1), np.ones(200), err_msg="every row needs exactly one 1")


def test_one_hot_unused_classes(sol):
    out = sol.one_hot(np.array([0, 0]), 5)
    assert out.shape == (2, 5), "num_classes sets the width even if some classes never appear"


# ---------------------------------------------------------------- pick_per_row

def test_pick_per_row_example(sol, no_loops):
    a = np.array([[10, 11, 12], [20, 21, 22]])
    out = sol.pick_per_row(a, np.array([2, 0]))
    np.testing.assert_array_equal(out, [12, 20])
    no_loops(sol.pick_per_row)


def test_pick_per_row_random(sol):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 7)).astype(np.float32)
    cols = rng.integers(0, 7, size=50)
    out = sol.pick_per_row(a, cols)
    assert out.shape == (50,), f"expected shape (50,), got {out.shape}; did you build an (n, n) block?"
    assert out.dtype == np.float32
    expected = np.array([a[i, c] for i, c in enumerate(cols)])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- apply_lut

def test_apply_lut_invert(sol, no_loops):
    invert = (255 - np.arange(256)).astype(np.uint8)
    img = np.array([[0, 10], [255, 128]], dtype=np.uint8)
    out = sol.apply_lut(img, invert)
    assert out.dtype == np.uint8 and out.shape == (2, 2)
    np.testing.assert_array_equal(out, [[255, 245], [0, 127]])
    no_loops(sol.apply_lut)


def test_apply_lut_gamma_3d(sol):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(6, 5, 3), dtype=np.uint8)
    gamma = np.round(255 * (np.arange(256) / 255) ** 0.5).astype(np.uint8)
    out = sol.apply_lut(img, gamma)
    assert out.shape == img.shape, f"expected shape {img.shape}, got {out.shape}"
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.vectorize(lambda v: gamma[v])(img).astype(np.uint8))


# ---------------------------------------------------------------- inverse_permutation

def test_inverse_permutation_example(sol, ban, no_loops):
    ban(np, "argsort")
    out = sol.inverse_permutation(np.array([2, 0, 1]))
    np.testing.assert_array_equal(out, [1, 2, 0])
    no_loops(sol.inverse_permutation)


def test_inverse_permutation_random(sol, ban):
    rng = np.random.default_rng(0)
    p = rng.permutation(100)
    expected = np.argsort(p)
    ban(np, "argsort")
    inv = sol.inverse_permutation(p)
    assert inv.shape == p.shape and inv.dtype == p.dtype
    np.testing.assert_array_equal(inv, expected)
    np.testing.assert_array_equal(p[inv], np.arange(100), err_msg="p[inv] should be 0..n-1")
    np.testing.assert_array_equal(inv[p], np.arange(100), err_msg="inv[p] should be 0..n-1")
