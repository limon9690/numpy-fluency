"""039 · NumPy and PyTorch bridge (shared memory, dtype, layout) (reference solution)."""

import numpy as np
import torch


def to_tensor_shared(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a)  # torch.tensor(a) would copy


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    chw = np.moveaxis(img, -1, -3)  # works with or without a leading batch axis
    # One contiguous float32 copy: fixes layout and dtype together, then scale to [0, 1].
    out = np.ascontiguousarray(chw, dtype=np.float32)
    out /= 255
    return torch.from_numpy(out)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    # detach() drops the autograd link, cpu() is a no-op for CPU tensors: still zero-copy.
    return t.detach().cpu().numpy()


def safe_from_numpy(a: np.ndarray) -> torch.Tensor:
    if not a.flags.writeable or min(a.strides, default=0) < 0:
        a = np.array(a)  # fresh, writeable, C-ordered copy
    return torch.from_numpy(a)
