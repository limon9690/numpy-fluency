import numpy as np
import pytest


# ---------------------------------------------------------------- sort_rows_by_column

def test_sort_rows_by_column_example(sol, no_loops):
    data = np.array([[2, 0], [1, 1], [2, 2], [1, 3]])
    out = sol.sort_rows_by_column(data, 0)
    np.testing.assert_array_equal(out, [[1, 1], [1, 3], [2, 0], [2, 2]])
    no_loops(sol.sort_rows_by_column)


def test_sort_rows_by_column_is_stable(sol):
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 5, size=1000)
    data = np.stack([keys, np.arange(1000)], axis=1)  # column 1 = original position
    out = sol.sort_rows_by_column(data, 0)
    assert out.shape == data.shape and out.dtype == data.dtype
    assert np.all(np.diff(out[:, 0]) >= 0), "rows should be sorted by column 0"
    for k in range(5):
        pos = out[out[:, 0] == k, 1]
        assert np.all(np.diff(pos) > 0), (
            f"rows with key {k} lost their original order; use kind='stable'"
        )


def test_sort_rows_by_column_float_col(sol):
    data = np.array([[0.0, 3.5], [1.0, -1.0], [2.0, 0.0]])
    np.testing.assert_array_equal(sol.sort_rows_by_column(data, 1)[:, 0], [1.0, 2.0, 0.0])


# ---------------------------------------------------------------- order_by_two_keys

def test_order_by_two_keys_example(sol, no_loops):
    out = sol.order_by_two_keys(np.array([2, 1, 2, 1]), np.array([30, 40, 20, 25]))
    np.testing.assert_array_equal(out, [3, 1, 2, 0],
                                  err_msg="primary decides first; careful, lexsort's LAST key is primary")
    no_loops(sol.order_by_two_keys)


def test_order_by_two_keys_random(sol):
    rng = np.random.default_rng(0)
    p = rng.integers(0, 4, size=300)
    s = rng.integers(0, 4, size=300)
    out = sol.order_by_two_keys(p, s)
    expected = sorted(range(300), key=lambda i: (p[i], s[i], i))
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- k_smallest

def test_k_smallest_example(sol, ban, no_loops):
    x = np.array([7, 2, 9, 1, 5])
    ban(np, "sort")
    ban(np, "argsort")
    out = sol.k_smallest(x, 3)
    assert out.shape == (3,) and out.dtype == x.dtype
    assert sorted(out.tolist()) == [1, 2, 5]
    no_loops(sol.k_smallest)


@pytest.mark.parametrize("k", [1, 10, 500])
def test_k_smallest_random(sol, ban, k):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    expected = sorted(x.tolist())[:k]
    ban(np, "sort")
    ban(np, "argsort")
    out = sol.k_smallest(x, k)
    assert out.shape == (k,), f"expected {k} values, got shape {out.shape}"
    np.testing.assert_array_equal(sorted(out.tolist()), expected)


# ---------------------------------------------------------------- rank

def test_rank_example(sol, no_loops):
    out = sol.rank(np.array([30, 10, 20, 10]))
    np.testing.assert_array_equal(out, [3, 0, 2, 1], err_msg="the earlier 10 should get the lower rank")
    no_loops(sol.rank)


def test_rank_random_with_ties(sol):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 10, size=400)
    out = sol.rank(x)
    order = sorted(range(400), key=lambda i: (x[i], i))
    expected = np.empty(400, dtype=int)
    expected[order] = np.arange(400)
    np.testing.assert_array_equal(out, expected)
