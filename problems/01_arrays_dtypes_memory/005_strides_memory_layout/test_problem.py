import numpy as np
import pytest

SHAPES = [(5,), (2, 3), (3, 1, 4), (2, 3, 4, 5), (7, 1), ()]


def _address(a, index):
    """Memory address of a[index], read from a 1-element view that starts at that element."""
    view = a[tuple(slice(i, i + 1) for i in index)]
    return view.__array_interface__["data"][0]


# ---------------------------------------------------------------- strides_for

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_strides_for_matches_numpy(sol, shape, order, dtype):
    itemsize = np.dtype(dtype).itemsize
    out = sol.strides_for(shape, itemsize, order)
    expected = np.empty(shape, dtype=dtype, order=order).strides
    assert isinstance(out, tuple), f"strides_for should return a tuple, got {type(out).__name__}"
    assert out == expected, f"shape {shape}, itemsize {itemsize}, order {order}: expected {expected}, got {out}"


def test_strides_for_default_order_is_c(sol):
    assert sol.strides_for((4, 5), 8) == (40, 8)


# ---------------------------------------------------------------- element_offset

def test_element_offset_example(sol):
    a = np.zeros((2, 3), dtype=np.int32)
    assert sol.element_offset(a, (1, 2)) == 20
    assert sol.element_offset(a.T, (2, 1)) == 20, "a.T[2, 1] is the same element as a[1, 2]"
    assert sol.element_offset(a, (0, 0)) == 0


@pytest.mark.parametrize("make_view", [
    lambda a: a,
    lambda a: a.T,
    lambda a: a[:, ::2],
    lambda a: a[::-1],
    lambda a: a[1:, ::-2],
    lambda a: np.moveaxis(a.reshape(2, 3, 4), 0, -1),
], ids=["plain", "transpose", "step", "reversed", "mixed", "moveaxis"])
def test_element_offset_matches_real_addresses(sol, make_view):
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    v = make_view(base)
    start = _address(v, (0,) * v.ndim)
    for index in np.ndindex(v.shape):
        got = sol.element_offset(v, index)
        assert int(got) == _address(v, index) - start, f"wrong offset for index {index} with strides {v.strides}"


def test_element_offset_returns_int(sol):
    out = sol.element_offset(np.zeros((3, 3)), (2, 2))
    assert isinstance(out, (int, np.integer)), f"expected an int, got {type(out).__name__}"
    assert out == 64


# ---------------------------------------------------------------- ensure_c_contiguous

def test_ensure_c_contiguous_copies_when_needed(sol):
    x = np.arange(12, dtype=np.float32).reshape(3, 4).T
    out = sol.ensure_c_contiguous(x)
    assert out.flags.c_contiguous, "result must be C-contiguous"
    assert out.shape == x.shape and out.dtype == x.dtype
    np.testing.assert_array_equal(out, x)
    assert not x.flags.c_contiguous, "the input must not be changed"


def test_ensure_c_contiguous_no_copy_when_already(sol):
    x = np.arange(12).reshape(3, 4)
    out = sol.ensure_c_contiguous(x)
    assert out.flags.c_contiguous
    assert np.shares_memory(out, x), "x is already C-contiguous; it should not be copied"


def test_ensure_c_contiguous_sliced(sol):
    x = np.arange(20).reshape(4, 5)[:, 1:4]
    out = sol.ensure_c_contiguous(x)
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, x)


# ---------------------------------------------------------------- repeat_rows_view

def test_repeat_rows_view_values(sol):
    row = np.array([1, 2, 3], dtype=np.int16)
    out = sol.repeat_rows_view(row, 4)
    assert out.shape == (4, 3), f"expected shape (4, 3), got {out.shape}"
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, np.tile(row, (4, 1)))


def test_repeat_rows_view_no_copy(sol):
    row = np.arange(5.0)
    out = sol.repeat_rows_view(row, 1000)
    assert np.shares_memory(out, row), "the result should be a view of `row`, not a copy"
    assert out.strides[0] == 0, f"axis 0 should have stride 0, got strides {out.strides}"
    assert not out.flags.writeable, "the result should be read-only"
    row[0] = 42.0
    assert np.all(out[:, 0] == 42.0), "every row should see changes to `row` (same memory)"
