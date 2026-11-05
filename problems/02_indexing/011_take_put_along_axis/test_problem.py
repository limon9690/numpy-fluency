import numpy as np
import pytest


def _distinct(shape, seed=0):
    """Random float array with all values distinct (so top-k has no ties)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.prod(shape)).reshape(shape).astype(np.float64) / 10


# ---------------------------------------------------------------- topk_last

def test_topk_last_example(sol, no_loops):
    x = np.array([[1, 9, 3, 7], [8, 2, 6, 4]])
    values, indices = sol.topk_last(x, 2)
    np.testing.assert_array_equal(values, [[9, 7], [8, 6]], err_msg="values should be the k largest, descending")
    np.testing.assert_array_equal(indices, [[1, 3], [0, 2]])
    assert values.dtype == x.dtype
    assert np.issubdtype(indices.dtype, np.integer), "indices should be integers"
    no_loops(sol.topk_last)


def test_topk_last_3d(sol):
    x = _distinct((3, 4, 6))
    values, indices = sol.topk_last(x, 3)
    assert values.shape == indices.shape == (3, 4, 3)
    np.testing.assert_array_equal(values, np.sort(x, axis=-1)[..., ::-1][..., :3])
    np.testing.assert_array_equal(np.take_along_axis(x, indices, -1), values,
                                  err_msg="values must be what the indices point to")


def test_topk_last_unsigned(sol):
    x = np.array([[3, 250, 0, 7]], dtype=np.uint8)
    values, indices = sol.topk_last(x, 2)
    np.testing.assert_array_equal(values, [[250, 7]], err_msg="careful: -x wraps around for uint8")
    np.testing.assert_array_equal(indices, [[1, 3]])


# ---------------------------------------------------------------- pick_along

def test_pick_along_example(sol, no_loops):
    x = np.array([[10, 11, 12], [20, 21, 22]])
    np.testing.assert_array_equal(sol.pick_along(x, np.array([2, 0]), axis=1), [12, 20])
    np.testing.assert_array_equal(sol.pick_along(x, np.array([1, 0, 1]), axis=0), [20, 11, 22])
    no_loops(sol.pick_along)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_pick_along_3d(sol, axis):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4, 5))
    ax = axis % 3
    out_shape = tuple(s for i, s in enumerate(x.shape) if i != ax)
    idx = rng.integers(0, x.shape[ax], size=out_shape)
    out = sol.pick_along(x, idx, axis)
    assert out.shape == out_shape, f"axis={axis}: expected shape {out_shape}, got {out.shape}"
    expected = np.empty(out_shape)
    for pos in np.ndindex(out_shape):
        full = list(pos)
        full.insert(ax, idx[pos])
        expected[pos] = x[tuple(full)]
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------- keep_topk

def test_keep_topk_example(sol, no_loops):
    x = np.array([[1, 9, 3, 7], [8, 2, 6, 4]])
    before = x.copy()
    out = sol.keep_topk(x, 2)
    np.testing.assert_array_equal(out, [[0, 9, 0, 7], [8, 0, 6, 0]])
    assert out.dtype == x.dtype
    np.testing.assert_array_equal(x, before, err_msg="keep_topk must not modify its input")
    no_loops(sol.keep_topk)


def test_keep_topk_random(sol):
    x = _distinct((20, 8)).astype(np.float32) + 1
    out = sol.keep_topk(x, 3)
    assert out.shape == x.shape and out.dtype == np.float32
    for row_in, row_out in zip(x, out):
        top = np.sort(row_in)[-3:]
        expected = np.where(np.isin(row_in, top), row_in, 0)
        np.testing.assert_array_equal(row_out, expected)


# ---------------------------------------------------------------- argmax_mask

def test_argmax_mask_example_ties(sol, no_loops):
    x = np.array([[1, 5, 5], [7, 2, 0]])
    out = sol.argmax_mask(x, axis=1)
    assert out.dtype == np.bool_, f"expected a boolean mask, got {out.dtype}"
    np.testing.assert_array_equal(out, [[False, True, False], [True, False, False]],
                                  err_msg="exactly one True per row, the first maximum on ties")
    no_loops(sol.argmax_mask)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_argmax_mask_3d(sol, axis):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, size=(3, 4, 5))  # many ties on purpose
    out = sol.argmax_mask(x, axis)
    assert out.shape == x.shape
    np.testing.assert_array_equal(out.sum(axis=axis), np.ones(np.delete(x.shape, axis)),
                                  err_msg="every slice along the axis needs exactly one True")
    np.testing.assert_array_equal(np.argmax(out, axis=axis), np.argmax(x, axis=axis))
