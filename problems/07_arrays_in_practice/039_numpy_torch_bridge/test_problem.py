import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _shares(t, a):
    """True if tensor t starts at the same memory address as array a."""
    return t.data_ptr() == a.__array_interface__["data"][0]


# ---------------------------------------------------------------- to_tensor_shared

def test_to_tensor_shared(sol):
    a = np.zeros(3, dtype=np.float32)
    t = sol.to_tensor_shared(a)
    assert isinstance(t, torch.Tensor)
    a[0] = 5.0
    assert t[0].item() == 5.0, "the tensor should share memory with the array (torch.tensor copies)"
    assert t.dtype == torch.float32


def test_to_tensor_shared_keeps_dtype(sol):
    for dtype, tdtype in [(np.float64, torch.float64), (np.int64, torch.int64), (np.uint8, torch.uint8)]:
        a = np.arange(6, dtype=dtype).reshape(2, 3)
        t = sol.to_tensor_shared(a)
        assert t.dtype == tdtype, f"{np.dtype(dtype)} should stay {tdtype}, got {t.dtype}"
        assert tuple(t.shape) == (2, 3) and _shares(t, a)


# ---------------------------------------------------------------- image_to_tensor

def test_image_to_tensor_single(sol):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    t = sol.image_to_tensor(img)
    assert t.dtype == torch.float32, f"expected torch.float32, got {t.dtype}"
    assert tuple(t.shape) == (3, 4, 5), f"(H, W, C) should become (C, H, W), got {tuple(t.shape)}"
    assert t.is_contiguous(), "the result should be contiguous"
    np.testing.assert_allclose(t.numpy(), np.moveaxis(img, -1, 0) / 255.0, rtol=1e-6)


def test_image_to_tensor_batch(sol):
    img = np.full((2, 4, 6, 3), 255, dtype=np.uint8)
    img[1, ..., 0] = 0
    t = sol.image_to_tensor(img)
    assert tuple(t.shape) == (2, 3, 4, 6), f"(N, H, W, C) should become (N, C, H, W), got {tuple(t.shape)}"
    assert t.max().item() == 1.0 and t[1, 0].max().item() == 0.0
    assert t.is_contiguous()


# ---------------------------------------------------------------- tensor_to_numpy

def test_tensor_to_numpy_requires_grad(sol):
    t = torch.ones(2, 3, requires_grad=True)
    a = sol.tensor_to_numpy(t)
    assert isinstance(a, np.ndarray), f"expected an ndarray, got {type(a).__name__}"
    assert a.dtype == np.float32 and a.shape == (2, 3)
    np.testing.assert_array_equal(a, np.ones((2, 3)))


def test_tensor_to_numpy_no_copy(sol):
    t = torch.arange(5, dtype=torch.float64)
    a = sol.tensor_to_numpy(t)
    assert a.dtype == np.float64
    t[0] = 42.0
    assert a[0] == 42.0, "for a CPU tensor without grad, the array should share memory with the tensor"


# ---------------------------------------------------------------- safe_from_numpy

def _quiet(func, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args)


def test_safe_from_numpy_negative_strides(sol):
    a = np.arange(6).reshape(2, 3)[:, ::-1]
    t = _quiet(sol.safe_from_numpy, a)
    np.testing.assert_array_equal(t.numpy(), a, err_msg="values must match the (reversed) array")
    assert tuple(t.shape) == (2, 3) and t.dtype == torch.int64


def test_safe_from_numpy_read_only(sol):
    row = np.arange(3, dtype=np.float32)
    a = np.broadcast_to(row, (4, 3))
    t = _quiet(sol.safe_from_numpy, a)  # from_numpy would warn about a non-writable array
    np.testing.assert_array_equal(t.numpy(), a)
    t[0, 0] = 99.0
    assert row[0] == 0.0, "writing to the tensor must not change a read-only array: copy it"


def test_safe_from_numpy_shares_when_possible(sol):
    base = np.arange(20.0).reshape(4, 5)
    a = base[:, ::2]  # positive, non-contiguous strides: PyTorch can share these
    t = _quiet(sol.safe_from_numpy, a)
    np.testing.assert_array_equal(t.numpy(), a)
    base[0, 0] = -1.0
    assert t[0, 0].item() == -1.0, "positive strides are supported: share memory instead of copying"
    c = np.ones((3, 3), dtype=np.float32)
    assert _shares(_quiet(sol.safe_from_numpy, c), c), "a plain contiguous array should not be copied"
