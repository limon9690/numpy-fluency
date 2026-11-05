import numpy as np
import pytest


# ---------------------------------------------------------------- bucketize

def test_bucketize_example(sol, no_loops):
    out = sol.bucketize(np.array([-5, 0, 9.9, 10, 25]), np.array([0, 10, 20]))
    np.testing.assert_array_equal(out, [-1, 0, 0, 1, 2])
    assert np.issubdtype(out.dtype, np.integer)
    no_loops(sol.bucketize)


def test_bucketize_matches_digitize(sol):
    rng = np.random.default_rng(0)
    edges = np.array([-1.0, 0.0, 0.5, 2.0])
    grid = rng.normal(0, 1.5, size=(10, 10))
    out = sol.bucketize(grid, edges)
    assert out.shape == (10, 10), "result should have the shape of x"
    np.testing.assert_array_equal(out, np.digitize(grid, edges) - 1)
    x = np.concatenate([grid.ravel(), edges])  # include values exactly on the edges
    np.testing.assert_array_equal(sol.bucketize(x, edges), np.digitize(x, edges) - 1,
                                  err_msg="values equal to an edge belong to the bin that starts there")


# ---------------------------------------------------------------- count_in_windows

def test_count_in_windows_example(sol, no_loops):
    out = sol.count_in_windows(np.array([1, 2, 2, 5, 9]), np.array([0, 2, 6]), np.array([2, 6, 9]))
    np.testing.assert_array_equal(out, [1, 3, 0], err_msg="windows are [start, end): start included, end excluded")
    no_loops(sol.count_in_windows)


def test_count_in_windows_random(sol):
    rng = np.random.default_rng(0)
    times = np.sort(rng.integers(0, 100, size=300))
    starts = rng.integers(0, 100, size=40)
    ends = starts + rng.integers(0, 30, size=40)
    out = sol.count_in_windows(times, starts, ends)
    expected = [((times >= s) & (times < e)).sum() for s, e in zip(starts, ends)]
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- histogram_counts

def test_histogram_counts_example(sol, ban, no_loops):
    ban(np, "histogram")
    counts, edges = sol.histogram_counts(np.array([0.1, 0.4, 0.5, 1.0, 3.0]), 2, 0.0, 1.0)
    np.testing.assert_array_equal(counts, [2, 2], err_msg="the last bin includes hi; values outside are dropped")
    np.testing.assert_allclose(edges, [0.0, 0.5, 1.0])
    assert edges.dtype == np.float64
    no_loops(sol.histogram_counts)


@pytest.mark.parametrize("n_bins, lo, hi", [(10, -2.0, 2.0), (4, 0.0, 1.0), (7, -1.0, 3.5)])
def test_histogram_counts_matches_numpy(sol, ban, n_bins, lo, hi):
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1.2, 500), np.linspace(lo, hi, n_bins + 1)])
    exp_counts, exp_edges = np.histogram(x, n_bins, range=(lo, hi))
    ban(np, "histogram")
    counts, edges = sol.histogram_counts(x, n_bins, lo, hi)
    assert counts.shape == (n_bins,), f"expected {n_bins} counts, got shape {counts.shape}"
    np.testing.assert_allclose(edges, exp_edges)
    np.testing.assert_array_equal(counts, exp_counts)


# ---------------------------------------------------------------- resample_linear

def test_resample_linear_example(sol, no_loops):
    out = sol.resample_linear(np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 0.0]), np.array([0.5, 1.5, 3.0]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [5.0, 5.0, 0.0])
    no_loops(sol.resample_linear)


def test_resample_linear_shape_and_clamp(sol):
    t = np.array([0.0, 2.0, 3.0])
    y = np.array([1.0, 5.0, -1.0])
    t_new = np.array([[-1.0, 0.0, 1.0], [2.5, 3.0, 10.0]])
    out = sol.resample_linear(t, y, t_new)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[1.0, 1.0, 3.0], [2.0, -1.0, -1.0]])


def test_resample_linear_unsorted_raises(sol):
    with pytest.raises(ValueError):
        sol.resample_linear(np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0]), np.array([0.5]))
    with pytest.raises(ValueError):
        sol.resample_linear(np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.0]), np.array([0.5]))
