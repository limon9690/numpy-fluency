import numpy as np
import pytest


def _ban_window_helpers(ban):
    ban(np, "convolve")
    ban(np.lib.stride_tricks, "sliding_window_view")


# ---------------------------------------------------------------- moving_average

def test_moving_average_example(sol, ban, no_loops):
    _ban_window_helpers(ban)
    out = sol.moving_average(np.array([3, 1, 4, 1, 5]), 3)
    assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
    np.testing.assert_allclose(out, [8 / 3, 2.0, 10 / 3])
    no_loops(sol.moving_average)


@pytest.mark.parametrize("w", [1, 4, 50])
def test_moving_average_random(sol, ban, w):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    expected = np.array([x[i:i + w].mean() for i in range(len(x) - w + 1)])
    _ban_window_helpers(ban)
    out = sol.moving_average(x, w)
    assert out.shape == (51 - w,), f"expected length n - w + 1 = {51 - w}, got {out.shape}"
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_moving_average_uint8_no_overflow(sol):
    x = np.full(10, 200, dtype=np.uint8)
    np.testing.assert_allclose(sol.moving_average(x, 4), np.full(7, 200.0),
                               err_msg="uint8 input overflowed; accumulate in a wide dtype")


# ---------------------------------------------------------------- running_mean

def test_running_mean(sol, no_loops):
    out = sol.running_mean(np.array([2, 4, 6, 0]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [2.0, 3.0, 4.0, 3.0])
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30)
    np.testing.assert_allclose(sol.running_mean(x), [x[:i + 1].mean() for i in range(30)])
    no_loops(sol.running_mean)


# ---------------------------------------------------------------- lengths_to_offsets

def test_lengths_to_offsets(sol, no_loops):
    out = sol.lengths_to_offsets(np.array([2, 3, 1]))
    assert out.dtype == np.int64, f"expected int64, got {out.dtype}"
    np.testing.assert_array_equal(out, [0, 2, 5, 6])
    np.testing.assert_array_equal(sol.lengths_to_offsets(np.array([0, 4, 0])), [0, 0, 4, 4])
    np.testing.assert_array_equal(sol.lengths_to_offsets(np.array([], dtype=np.int64)), [0])
    no_loops(sol.lengths_to_offsets)


def test_lengths_to_offsets_slices_packed_array(sol):
    lengths = np.array([3, 1, 2], dtype=np.int32)
    packed = np.arange(6) * 10
    off = sol.lengths_to_offsets(lengths)
    assert off.dtype == np.int64
    pieces = [packed[off[k]:off[k + 1]].tolist() for k in range(3)]
    assert pieces == [[0, 10, 20], [30], [40, 50]]


# ---------------------------------------------------------------- rising_edges

def test_rising_edges_example(sol, no_loops):
    out = sol.rising_edges(np.array([False, False, True, True, False, True]))
    np.testing.assert_array_equal(out, [2, 5])
    assert np.issubdtype(out.dtype, np.integer)
    no_loops(sol.rising_edges)


def test_rising_edges_edge_cases(sol):
    assert sol.rising_edges(np.array([True, True, False])).size == 0, (
        "starting True is not a rising edge, and falling edges don't count"
    )
    assert sol.rising_edges(np.array([False, False])).size == 0
    rng = np.random.default_rng(0)
    s = rng.random(200) < 0.5
    expected = [i for i in range(1, 200) if not s[i - 1] and s[i]]
    np.testing.assert_array_equal(sol.rising_edges(s), expected)
