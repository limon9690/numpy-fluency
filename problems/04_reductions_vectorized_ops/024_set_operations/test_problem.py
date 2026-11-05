import numpy as np
import pytest


# ---------------------------------------------------------------- new_ids

def test_new_ids(sol, no_loops):
    out = sol.new_ids(np.array([1, 2, 3]), np.array([3, 7, 5, 7]))
    np.testing.assert_array_equal(out, [5, 7], err_msg="result should be sorted and without duplicates")
    assert sol.new_ids(np.array([1, 2]), np.array([2, 1])).size == 0
    no_loops(sol.new_ids)


def test_new_ids_random(sol):
    rng = np.random.default_rng(0)
    old = rng.integers(0, 50, 40)
    cur = rng.integers(0, 60, 40)
    np.testing.assert_array_equal(sol.new_ids(old, cur), sorted(set(cur.tolist()) - set(old.tolist())))


# ---------------------------------------------------------------- keep_known

def test_keep_known_example(sol, no_loops):
    x = np.array([[1, 2], [3, 4]])
    before = x.copy()
    out = sol.keep_known(x, np.array([2, 3]), -1)
    np.testing.assert_array_equal(out, [[-1, 2], [3, -1]])
    assert out.shape == x.shape and out.dtype == x.dtype
    np.testing.assert_array_equal(x, before, err_msg="keep_known must not modify its input")
    no_loops(sol.keep_known)


def test_keep_known_keeps_order_and_duplicates(sol):
    x = np.array([9, 4, 9, 1, 4, 7])
    out = sol.keep_known(x, np.array([4, 9]), 0)
    np.testing.assert_array_equal(out, [9, 4, 9, 0, 4, 0], err_msg="order and repeats of x must be kept")


# ---------------------------------------------------------------- common_with_positions

def test_common_with_positions_example(sol, no_loops):
    common, ia, ib = sol.common_with_positions(np.array([5, 1, 5, 3]), np.array([3, 5, 5, 7]))
    np.testing.assert_array_equal(common, [3, 5])
    np.testing.assert_array_equal(ia, [3, 0], err_msg="ia should point at the FIRST occurrence in a")
    np.testing.assert_array_equal(ib, [0, 1])
    no_loops(sol.common_with_positions)


def test_common_with_positions_random(sol):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 30, 25)
    b = rng.integers(0, 30, 25)
    common, ia, ib = sol.common_with_positions(a, b)
    expected = sorted(set(a.tolist()) & set(b.tolist()))
    np.testing.assert_array_equal(common, expected)
    np.testing.assert_array_equal(a[ia], common)
    np.testing.assert_array_equal(b[ib], common)
    np.testing.assert_array_equal(ia, [a.tolist().index(v) for v in expected])
    np.testing.assert_array_equal(ib, [b.tolist().index(v) for v in expected])


# ---------------------------------------------------------------- jaccard

def test_jaccard(sol, no_loops):
    out = sol.jaccard(np.array([1, 2, 2, 3]), np.array([2, 3, 4]))
    assert type(out) is float, f"return a Python float, got {type(out).__name__}"
    assert out == pytest.approx(0.5)
    assert sol.jaccard(np.array([1, 1]), np.array([1])) == pytest.approx(1.0)
    assert sol.jaccard(np.array([1, 2]), np.array([3])) == pytest.approx(0.0)
    assert sol.jaccard(np.array([], dtype=int), np.array([], dtype=int)) == 1.0
    no_loops(sol.jaccard)
