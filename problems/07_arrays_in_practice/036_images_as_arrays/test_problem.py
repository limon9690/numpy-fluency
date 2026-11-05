import numpy as np
import pytest


def _image(h=6, w=8, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------- to_float

def test_to_float_uint8(sol, no_loops):
    img = _image()
    out = sol.to_float(img)
    assert out.dtype == np.float32, f"expected float32, got {out.dtype}"
    assert out.shape == img.shape
    np.testing.assert_allclose(out, img / 255.0, rtol=1e-6, err_msg="uint8 0..255 should map to 0..1")
    assert 0.0 <= out.min() and out.max() <= 1.0
    no_loops(sol.to_float)


def test_to_float_float_input(sol):
    f = np.array([[0.0, 0.25, 1.0]])
    out = sol.to_float(f)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, f, err_msg="float input is already 0..1: don't divide by 255 again")


# ---------------------------------------------------------------- to_uint8

def test_to_uint8_clip_and_round(sol, no_loops):
    out = sol.to_uint8(np.array([-0.2, 0.0, 0.5, 0.999, 1.3]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [0, 0, 128, 255, 255],
                                  err_msg="clip to [0, 1], scale by 255, round (0.999 -> 255, not 254)")
    no_loops(sol.to_uint8)


def test_to_uint8_round_trip(sol):
    img = _image()
    back = sol.to_uint8(img.astype(np.float32) / 255)
    assert back.shape == img.shape
    np.testing.assert_array_equal(back, img, err_msg="uint8 -> float -> uint8 should give the same pixels")


# ---------------------------------------------------------------- rgb_to_gray

def test_rgb_to_gray_uint8(sol, no_loops):
    out = sol.rgb_to_gray(np.array([[[255, 0, 0], [255, 255, 255]]], np.uint8))
    assert out.dtype == np.uint8, f"uint8 input should give uint8, got {out.dtype}"
    assert out.shape == (1, 2)
    np.testing.assert_array_equal(out, [[76, 255]])
    no_loops(sol.rgb_to_gray)


def test_rgb_to_gray_random_and_batch(sol):
    w = np.array([0.299, 0.587, 0.114])
    img = _image()
    out = sol.rgb_to_gray(img)
    assert out.shape == (6, 8)
    expected = np.round(img.astype(np.float64) @ w)
    assert np.abs(out.astype(int) - expected).max() <= 0, "values should be the rounded weighted sum"
    batch = np.stack([_image(seed=s) for s in range(3)]).astype(np.float32) / 255
    out = sol.rgb_to_gray(batch)
    assert out.shape == (3, 6, 8), f"a batch (N, H, W, 3) should give (N, H, W), got {out.shape}"
    assert out.dtype == np.float32, f"float32 input should stay float32, got {out.dtype}"
    np.testing.assert_allclose(out, batch @ w.astype(np.float32), rtol=1e-5)


# ---------------------------------------------------------------- alpha_blend

def test_alpha_blend_example(sol, no_loops):
    fg = np.full((1, 1, 3), 200, np.uint8)
    bg = np.full((1, 1, 3), 100, np.uint8)
    out = sol.alpha_blend(fg, bg, np.array([[0.25]]))
    assert out.dtype == np.uint8 and out.shape == (1, 1, 3)
    np.testing.assert_array_equal(out, [[[125, 125, 125]]])
    no_loops(sol.alpha_blend)


def test_alpha_blend_random(sol):
    fg, bg = _image(seed=1), _image(seed=2)
    alpha = np.random.default_rng(3).random((6, 8))
    alpha[0, 0], alpha[0, 1] = 1.0, 0.0
    out = sol.alpha_blend(fg, bg, alpha)
    assert out.shape == (6, 8, 3) and out.dtype == np.uint8
    expected = np.round(alpha[..., None] * fg + (1 - alpha[..., None]) * bg).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out[0, 0], fg[0, 0], err_msg="alpha = 1 should give the foreground")
    np.testing.assert_array_equal(out[0, 1], bg[0, 1], err_msg="alpha = 0 should give the background")


def test_alpha_blend_no_overflow(sol):
    fg = np.full((2, 2, 3), 250, np.uint8)
    bg = np.full((2, 2, 3), 240, np.uint8)
    out = sol.alpha_blend(fg, bg, np.full((2, 2), 0.5))
    np.testing.assert_array_equal(out, np.full((2, 2, 3), 245, np.uint8),
                                  err_msg="bright pixels wrapped around: do the math in float")
