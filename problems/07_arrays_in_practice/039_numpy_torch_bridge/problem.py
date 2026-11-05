"""039 · NumPy and PyTorch bridge (shared memory, dtype, layout)."""

import numpy as np
import torch


def to_tensor_shared(a: np.ndarray) -> torch.Tensor:
    """A tensor that shares memory with the NumPy array `a` (no copy, same dtype).

    `a` is writeable and has no negative strides. Changing `a` afterwards changes the tensor.

    Example:
        a = np.zeros(3, np.float32); t = to_tensor_shared(a); a[0] = 5
        t -> tensor([5., 0., 0.])
    """
    raise NotImplementedError


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC image(s) to the float32 CHW layout PyTorch models expect.

    (H, W, C) -> (C, H, W) and (N, H, W, C) -> (N, C, H, W). Values are divided by 255 to lie in
    [0, 1]. The result is a contiguous torch.float32 tensor.

    Example:
        image_to_tensor(np.full((2, 3, 3), 255, np.uint8)).shape -> torch.Size([3, 2, 3])  (all ones)
    """
    raise NotImplementedError


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """A NumPy array with the values of tensor `t`.

    Must work when `t` requires grad (and would work for a GPU tensor). For a CPU tensor that
    doesn't require grad, the result shares memory with `t` (no copy). Same dtype and shape.

    Example:
        tensor_to_numpy(torch.ones(2, requires_grad=True)) -> array([1., 1.], dtype=float32)
    """
    raise NotImplementedError


def safe_from_numpy(a: np.ndarray) -> torch.Tensor:
    """A tensor with the values, shape and dtype of ANY NumPy array, without errors or warnings.

    Share memory when torch.from_numpy can do it safely: `a` is writeable and has no negative
    strides (positive non-contiguous strides are fine). Otherwise (e.g. a[::-1], or a read-only
    np.broadcast_to result) build the tensor from a copy.

    Example:
        safe_from_numpy(np.arange(4)[::-1]) -> tensor([3, 2, 1, 0])
    """
    raise NotImplementedError
