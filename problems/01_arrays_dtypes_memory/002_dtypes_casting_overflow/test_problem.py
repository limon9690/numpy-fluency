import numpy as np
import pytest


# ---------------------------------------------------------------- brighten

def test_brighten_saturates_high(sol):
    img = np.array([250, 100, 3, 255], dtype=np.uint8)
    out = sol.brighten(img, 10)
    assert out.dtype == np.uint8, f"expected uint8, got {out.dtype}"
    np.testing.assert_array_equal(out, [255, 110, 13, 255],
                                  err_msg="values above 255 must stick at 255, not wrap around")


def test_brighten_saturates_low(sol):
    img = np.array([[250, 100], [3, 0]], dtype=np.uint8)
    out = sol.brighten(img, -5)
    assert out.shape == (2, 2) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[245, 95], [0, 0]],
                                  err_msg="values below 0 must stick at 0, not wrap to 251+")


def test_brighten_large_amount_and_input_untouched(sol):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    before = img.copy()
    for amount in (300, -300, 0, 77):
        out = sol.brighten(img, amount)
        expected = np.clip(img.astype(np.int64) + amount, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(out, expected, err_msg=f"wrong result for amount={amount}")
    np.testing.assert_array_equal(img, before, err_msg="brighten must not modify its input")


# ---------------------------------------------------------------- round_half_up

def test_round_half_up_halves(sol):
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    out = sol.round_half_up(x)
    assert out.dtype == np.float64, f"expected float64, got {out.dtype}"
    np.testing.assert_array_equal(out, [1, 2, 3, -1, -2, -3],
                                  err_msg="halves should go away from zero (np.round goes to even)")


def test_round_half_up_non_halves(sol):
    x = np.array([[2.4, 2.6, -2.4], [-2.6, 0.0, 7.0]])
    out = sol.round_half_up(x)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, [[2, 3, -2], [-3, 0, 7]])


def test_round_half_up_float32_input(sol):
    out = sol.round_half_up(np.array([0.5, -3.5], dtype=np.float32))
    assert out.dtype == np.float64, f"result should be float64, got {out.dtype}"
    np.testing.assert_array_equal(out, [1, -4])


# ---------------------------------------------------------------- to_int_safe

def test_to_int_safe_rounds_half_even(sol):
    out = sol.to_int_safe(np.array([1.6, 2.5, -1.7, 3.5, -0.4]), np.int8)
    assert out.dtype == np.int8, f"expected int8, got {out.dtype}"
    np.testing.assert_array_equal(out, [2, 2, -2, 4, 0],
                                  err_msg="round with np.round first (half to even); astype alone truncates")


def test_to_int_safe_edges_fit(sol):
    out = sol.to_int_safe(np.array([[127.4, -128.4], [0.0, 254.6]]), np.int16)
    assert out.dtype == np.int16 and out.shape == (2, 2)
    np.testing.assert_array_equal(out, [[127, -128], [0, 255]])
    out = sol.to_int_safe(np.array([255.4, 0.2]), np.uint8)
    np.testing.assert_array_equal(out, np.array([255, 0], dtype=np.uint8))


def test_to_int_safe_overflow(sol):
    with pytest.raises(OverflowError):
        sol.to_int_safe(np.array([1.0, 127.6]), np.int8)
    with pytest.raises(OverflowError):
        sol.to_int_safe(np.array([-0.6]), np.uint8)


def test_to_int_safe_nan_inf(sol):
    with pytest.raises(ValueError):
        sol.to_int_safe(np.array([1.0, np.nan]), np.int32)
    with pytest.raises(ValueError):
        sol.to_int_safe(np.array([np.inf]), np.int64)


# ---------------------------------------------------------------- scale_keep_dtype

def test_scale_keep_dtype_numpy_scalar(sol):
    a = np.linspace(0, 1, 7, dtype=np.float32)
    before = a.copy()
    out = sol.scale_keep_dtype(a, np.float64(0.1))
    assert out.dtype == np.float32, (
        f"expected float32, got {out.dtype}: a NumPy float64 scalar promotes float32 arrays"
    )
    np.testing.assert_allclose(out, a.astype(np.float64) * 0.1, rtol=1e-6)
    np.testing.assert_array_equal(a, before, err_msg="scale_keep_dtype must not modify its input")


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_scale_keep_dtype_all_float_dtypes(sol, dtype):
    a = np.arange(6, dtype=dtype).reshape(2, 3)
    for factor in (2.0, np.float64(0.5), np.float32(3.0)):
        out = sol.scale_keep_dtype(a, factor)
        assert out.dtype == dtype, f"input {np.dtype(dtype)} with factor {factor!r} gave {out.dtype}"
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out.astype(np.float64), a.astype(np.float64) * float(factor), rtol=1e-3)
