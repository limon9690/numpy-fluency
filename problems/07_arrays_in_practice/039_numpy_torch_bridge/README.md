# 039 · NumPy and PyTorch bridge (shared memory, dtype, layout) ★

**Category:** Arrays in practice · **Difficulty:** Easy

## What it is

`torch.from_numpy(a)` turns a NumPy array into a tensor, and `t.numpy()` turns a CPU tensor
into an array. Both **share memory**: no copy, and a change on one side shows up on the other.
`torch.tensor(a)` always copies. The rest of the bridge is about the three things that differ:
**dtype defaults**, **memory layout** (strides) and **axis order** for images.

## Why it exists

Data loading and preprocessing usually happen in NumPy; training happens in PyTorch; analysis
of the outputs goes back to NumPy. Crossing that line many times per second must be cheap
(sharing, not copying), and the crossing is where most "expected Float but found Double" and
"negative strides" errors come from.

## How it works

**Sharing vs copying:**

| call | copies? |
|---|---|
| `torch.from_numpy(a)` | no, shares memory (same dtype) |
| `torch.as_tensor(a)` | no if it can share, yes otherwise |
| `torch.tensor(a)` | always copies |
| `t.numpy()` | no (CPU tensor without grad only) |

```python
a = np.zeros(3, dtype=np.float32)
t = torch.from_numpy(a)
a[0] = 5.0
t                      # tensor([5., 0., 0.])   same memory
```

**dtype.** NumPy's default float is **float64**, PyTorch's is **float32**. `from_numpy` keeps
the dtype, so `torch.from_numpy(np.zeros((1, 2)))` is a float64 tensor, and feeding it to a
float32 layer fails with something like `mat1 and mat2 must have the same dtype ... Double and
Float`. Convert explicitly: `.astype(np.float32)` before, or `.float()` after.

**Layout.** Tensors support arbitrary **positive** strides, so `from_numpy` works on sliced and
transposed arrays. **Negative** strides (from `a[::-1]` or `np.flip`) raise
`ValueError: At least one stride in the given numpy array is negative`. Make a contiguous copy
first: `np.ascontiguousarray(a)`. Read-only arrays (e.g. from `np.broadcast_to`) work but give a
warning, because PyTorch can't promise not to write to them; copy them too.

**Images.** NumPy/PIL/OpenCV images are `(H, W, C)` uint8 in 0..255. PyTorch models expect
`(C, H, W)` float32 in 0..1, with a batch axis in front: `(N, C, H, W)`.

```
(H, W, C) uint8  --moveaxis(-1, -3)-->  (C, H, W) uint8  --float32, /255-->  (C, H, W) float32
```

`permute`/`moveaxis` only change strides, so the result isn't contiguous (005, 014). A
contiguous copy (`np.ascontiguousarray` or `t.contiguous()`) avoids surprises later (`.view()`
fails on non-contiguous tensors).

**Back to NumPy:** `t.numpy()` fails if `t` requires grad ("Can't call numpy() on Tensor that
requires grad") or lives on the GPU. The safe chain is `t.detach().cpu().numpy()`. On a CPU
tensor this is still zero-copy.

## Gotchas

- Shared memory cuts both ways: an in-place op on the tensor (`t += 1`, `t.relu_()`) changes
  your NumPy array.
- `np.float64` data silently becomes float64 tensors. Decide on float32 at the boundary.
- `a[::-1]`, `np.flip(a)` and `img[..., ::-1]` (BGR to RGB) all make negative strides.
- `axis` (NumPy) vs `dim` (PyTorch); `keepdims` vs `keepdim`.
- PyTorch scalars: `t.item()` gives a Python number, like NumPy's `.item()` (003).

## Builds on

002, 005, 014, 036.

## Your task

Implement in `problem.py` (tests skip if PyTorch isn't installed):

1. `to_tensor_shared(a)` - a tensor that shares memory with `a` (same dtype).
2. `image_to_tensor(img)` - uint8 `(H, W, C)` -> contiguous float32 `(C, H, W)` tensor in
   `[0, 1]`; a batch `(N, H, W, C)` -> `(N, C, H, W)`.
3. `tensor_to_numpy(t)` - a NumPy array with `t`'s values, even if `t` requires grad. For a CPU
   tensor without grad, no copy.
4. `safe_from_numpy(a)` - a tensor from **any** array: share memory when PyTorch allows it
   (writeable, no negative strides), otherwise copy. Never raise or warn.

## Hints

<details>
<summary>Hint 1</summary>

`image_to_tensor`: `np.moveaxis(img, -1, -3)`, then `np.ascontiguousarray(..., dtype=np.float32) / 255`,
then `torch.from_numpy`.

</details>

<details>
<summary>Hint 2</summary>

`tensor_to_numpy`: `t.detach().cpu().numpy()`.

</details>

<details>
<summary>Hint 3</summary>

`safe_from_numpy`: check `a.flags.writeable` and `min(a.strides, default=0) >= 0`; if either
fails, pass a copy (`np.array(a)` makes a fresh, writeable, C-ordered one).

</details>

## Self-check

1. What is the difference between `torch.from_numpy(a)` and `torch.tensor(a)`?
2. Why does a model complain about "Double" when you feed it data from NumPy?
3. Why does `torch.from_numpy(img[..., ::-1])` fail?
4. When does `t.numpy()` raise, and what's the fix?

<details>
<summary>Answers</summary>

1. `from_numpy` shares memory with `a` (no copy); `torch.tensor` always copies.
2. NumPy defaults to float64 and `from_numpy` keeps it; the model's weights are float32.
3. Reversing an axis gives a negative stride, which tensors don't support. Copy first.
4. When `t` requires grad or is on a GPU; use `t.detach().cpu().numpy()`.

</details>

## Go deeper

- [torch.from_numpy](https://pytorch.org/docs/stable/generated/torch.from_numpy.html)
