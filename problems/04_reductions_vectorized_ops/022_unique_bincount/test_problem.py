import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------- encode_labels

def test_encode_labels_example(sol, no_loops):
    classes, codes = sol.encode_labels(np.array([30, 10, 30, 20]))
    np.testing.assert_array_equal(classes, [10, 20, 30])
    np.testing.assert_array_equal(codes, [2, 0, 2, 1])
    assert np.issubdtype(codes.dtype, np.integer)
    no_loops(sol.encode_labels)


def test_encode_labels_2d_keeps_shape(sol):
    rng = np.random.default_rng(0)
    labels = rng.choice([-1.5, 0.0, 2.25, 7.0], size=(4, 6))
    classes, codes = sol.encode_labels(labels)
    assert codes.shape == (4, 6), f"codes should have the shape of labels (4, 6), got {codes.shape}"
    np.testing.assert_array_equal(classes, np.unique(labels))
    np.testing.assert_array_equal(classes[codes], labels, err_msg="classes[codes] should rebuild labels")


# ---------------------------------------------------------------- unique_rows_counts

def test_unique_rows_counts_example(sol):
    rows, counts = sol.unique_rows_counts(np.array([[1, 2], [3, 4], [1, 2]]))
    np.testing.assert_array_equal(rows, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(counts, [2, 1])


def test_unique_rows_counts_random(sol):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=(50, 3))
    rows, counts = sol.unique_rows_counts(a)
    assert rows.ndim == 2 and rows.shape[1] == 3, f"rows should be 2-D with 3 columns, got {rows.shape}"
    assert counts.sum() == 50
    seen = {}
    for r in a.tolist():
        seen[tuple(r)] = seen.get(tuple(r), 0) + 1
    assert [tuple(r) for r in rows.tolist()] == sorted(seen), "rows should be distinct and sorted"
    assert counts.tolist() == [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------------- most_frequent

def test_most_frequent(sol, ban, no_loops):
    ban(np, "unique")
    out = sol.most_frequent(np.array([3, 1, 3, 1, 2]))
    assert type(out) is int, f"return a Python int, got {type(out).__name__}"
    assert out == 1, "on ties, return the smallest value"
    assert sol.most_frequent(np.array([5])) == 5
    assert sol.most_frequent(np.array([0, 9, 9])) == 9
    no_loops(sol.most_frequent)


def test_most_frequent_random(sol, ban):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 20, size=1000)
    counts = {v: (x == v).sum() for v in range(20)}
    best = max(counts.values())
    expected = min(v for v, c in counts.items() if c == best)
    ban(np, "unique")
    assert sol.most_frequent(x) == expected


# ---------------------------------------------------------------- group_mean

def test_group_mean_example(sol, ban, no_loops):
    ban(np, "unique")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sol.group_mean(np.array([0, 1, 0, 3]), np.array([10.0, 5.0, 20.0, 7.0]), 4)
    assert out.dtype == np.float64 and out.shape == (4,)
    np.testing.assert_allclose(out, [15.0, 5.0, np.nan, 7.0], err_msg="empty groups should be NaN")
    no_loops(sol.group_mean)


def test_group_mean_random(sol, ban):
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 6, size=200)
    values = rng.standard_normal(200)
    expected = np.array([values[groups == g].mean() if (groups == g).any() else np.nan for g in range(8)])
    ban(np, "unique")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sol.group_mean(groups, values, 8)
    assert out.shape == (8,), "n_groups sets the length, even when the last groups are empty"
    np.testing.assert_allclose(out, expected)
