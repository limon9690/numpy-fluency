import numpy as np
import pytest


# ---------------------------------------------------------------- safe_arange

def test_safe_arange_avoids_extra_point(sol):
    out = sol.safe_arange(1, 1.3, 0.1)
    assert isinstance(out, np.ndarray), "safe_arange should return a NumPy array"
    assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
    assert out.shape == (3,), f"expected 3 points (1.0, 1.1, 1.2), got shape {out.shape}"
    np.testing.assert_allclose(out, [1.0, 1.1, 1.2])


def test_safe_arange_matches_formula(sol):
    for start, stop, step in [(0, 1, 0.1), (0, 5, 1), (-2, 2, 0.5), (1, 0, -0.25), (0.3, 2.1, 0.3)]:
        n = int(round((stop - start) / step))
        out = sol.safe_arange(start, stop, step)
        assert out.shape == (n,), f"safe_arange{(start, stop, step)}: expected {n} points, got {out.shape}"
        np.testing.assert_allclose(out, start + step * np.arange(n), atol=1e-12,
                                   err_msg=f"wrong values for safe_arange{(start, stop, step)}")


def test_safe_arange_int_inputs_give_float(sol):
    out = sol.safe_arange(0, 5, 1)
    assert out.dtype == np.float64, f"integer inputs should still give float64, got {out.dtype}"
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_safe_arange_empty(sol):
    out = sol.safe_arange(2, 1, 0.5)
    assert out.shape == (0,), f"an empty range should give shape (0,), got {out.shape}"
    assert out.dtype == np.float64, f"even an empty result should be float64, got {out.dtype}"


# ---------------------------------------------------------------- linspace_with_step

def test_linspace_with_step_values(sol):
    points, step = sol.linspace_with_step(0, 1, 5)
    assert points.dtype == np.float64 and points.shape == (5,), "expected 5 float64 points"
    np.testing.assert_allclose(points, [0, 0.25, 0.5, 0.75, 1.0])
    assert type(step) is float, f"step should be a Python float, got {type(step).__name__}"
    assert step == pytest.approx(0.25)


def test_linspace_with_step_endpoints_exact(sol):
    points, step = sol.linspace_with_step(1.0, 1.3, 4)
    assert points[0] == 1.0 and points[-1] == 1.3, "both endpoints must be included exactly"
    assert step == pytest.approx(0.1)
    points, step = sol.linspace_with_step(10, -10, 3)
    np.testing.assert_allclose(points, [10, 0, -10])
    assert step == pytest.approx(-10.0)


# ---------------------------------------------------------------- blank_like

def test_blank_like_int_input_gives_float(sol):
    a = np.array([[1, 2, 3], [4, 5, 6]])
    out = sol.blank_like(a, 0.5)
    assert out.dtype == np.float64, f"expected float64 even for int input, got {out.dtype}"
    assert out.shape == (2, 3), f"expected shape (2, 3), got {out.shape}"
    np.testing.assert_array_equal(out, np.full((2, 3), 0.5),
                                  err_msg="0.5 was lost; did the int dtype of `a` truncate it?")


def test_blank_like_shape_and_independence(sol):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    before = a.copy()
    out = sol.blank_like(a, -1)
    assert out.shape == a.shape and out.dtype == np.float64
    assert np.all(out == -1.0)
    assert not np.shares_memory(out, a), "result must be a new array, not a view of `a`"
    np.testing.assert_array_equal(a, before, err_msg="blank_like must not modify its input")


# ---------------------------------------------------------------- as_float32_matrix

def test_as_float32_matrix_basic(sol):
    out = sol.as_float32_matrix([[1, 2], [3, 4], [5, 6]])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32, f"expected float32, got {out.dtype}"
    assert out.shape == (3, 2), f"expected shape (3, 2), got {out.shape}"
    np.testing.assert_array_equal(out, np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))


def test_as_float32_matrix_single_row(sol):
    out = sol.as_float32_matrix([[0.5, 1.5, 2.5]])
    assert out.shape == (1, 3) and out.dtype == np.float32


def test_as_float32_matrix_ragged_raises(sol):
    with pytest.raises(ValueError):
        sol.as_float32_matrix([[1, 2], [3]])


def test_as_float32_matrix_wrong_ndim_raises(sol):
    with pytest.raises(ValueError):
        sol.as_float32_matrix([1, 2, 3])
    with pytest.raises(ValueError):
        sol.as_float32_matrix([[[1, 2]], [[3, 4]]])
