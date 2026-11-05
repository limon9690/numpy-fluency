import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------- safe_divide

def test_safe_divide_example(sol, no_loops):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a test failure
        out = sol.safe_divide(np.array([1, 2, 3]), np.array([2, 0, 4]), fill=-1.0)
    assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
    np.testing.assert_allclose(out, [0.5, -1.0, 0.75])
    no_loops(sol.safe_divide)


def test_safe_divide_no_warnings_with_zero_over_zero(sol):
    a = np.array([[0.0, 1.0], [-3.0, 0.0]])
    b = np.array([[0.0, 0.0], [1.5, 2.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sol.safe_divide(a, b)
    np.testing.assert_allclose(out, [[0.0, 0.0], [-2.0, 0.0]],
                               err_msg="default fill is 0.0; 0/0 must not produce NaN or a warning")


def test_safe_divide_broadcasts(sol):
    a = np.arange(6.0).reshape(2, 3)
    b = np.array([1.0, 0.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sol.safe_divide(a, b, fill=np.inf)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[0.0, np.inf, 1.0], [3.0, np.inf, 2.5]])


# ---------------------------------------------------------------- bounding_box

def test_bounding_box_example(sol):
    m = np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]], dtype=bool)
    box = sol.bounding_box(m)
    assert isinstance(box, tuple) and len(box) == 2, "expected a tuple of 2 slices"
    assert all(isinstance(s, slice) for s in box), "every entry should be a slice"
    assert [(s.start, s.stop) for s in box] == [(1, 3), (1, 3)], f"got {box}"


def test_bounding_box_crops(sol):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 10, size=(12, 15))
    m = np.zeros((12, 15), bool)
    m[3, 9] = m[7, 4] = m[5, 6] = True
    box = sol.bounding_box(m)
    np.testing.assert_array_equal(img[box], img[3:8, 4:10], err_msg="x[box] should crop rows 3-7 and cols 4-9")


def test_bounding_box_3d_and_single_pixel(sol):
    m = np.zeros((4, 5, 6), bool)
    m[2, 1, 5] = True
    box = sol.bounding_box(m)
    assert len(box) == 3
    assert [(s.start, s.stop) for s in box] == [(2, 3), (1, 2), (5, 6)]


def test_bounding_box_empty(sol):
    assert sol.bounding_box(np.zeros((3, 3), bool)) is None, "no True element should give None"


# ---------------------------------------------------------------- first_true

def test_first_true(sol, no_loops):
    assert sol.first_true(np.array([False, False, True, True])) == 2
    assert sol.first_true(np.array([True, False])) == 0
    out = sol.first_true(np.array([False, False]))
    assert out == -1, f"no True should give -1, got {out} (np.argmax returns 0 here!)"
    assert sol.first_true(np.zeros(0, bool)) == -1
    no_loops(sol.first_true)


def test_first_true_returns_int(sol):
    out = sol.first_true(np.arange(10) > 6)
    assert isinstance(out, int), f"expected a Python int, got {type(out).__name__}"
    assert out == 7


# ---------------------------------------------------------------- piecewise_curve

def test_piecewise_curve_example(sol, no_loops):
    out = sol.piecewise_curve(np.array([-2.0, 0.5, 3.0]))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [4.0, 0.5, 1.0])
    no_loops(sol.piecewise_curve)


def test_piecewise_curve_boundaries_and_shape(sol):
    x = np.array([[-1.0, 0.0], [0.999, 1.0]])
    out = sol.piecewise_curve(x)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[1.0, 0.0], [0.999, 1.0]])
    x = np.linspace(-3, 3, 101)
    expected = np.array([v**2 if v < 0 else (v if v < 1 else 1.0) for v in x])
    np.testing.assert_allclose(sol.piecewise_curve(x), expected)
