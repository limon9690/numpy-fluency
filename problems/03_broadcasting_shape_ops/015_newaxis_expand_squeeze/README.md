# 015 · newaxis, expand_dims, squeeze

**Category:** Broadcasting and shape ops · **Difficulty:** Easy

## What it is

Three ways to add or remove axes of length 1:

- `x[None]`, `x[:, None]`, `x[..., None]` - insert a length-1 axis where the `None` is
  (`np.newaxis` is just another name for `None`).
- `np.expand_dims(x, axis)` - the same, with the position as a number (or a tuple of numbers).
- `np.squeeze(x, axis=...)` - remove length-1 axes.

All three return views.

## Why it exists

Length-1 axes are how you control broadcasting (012): a `(C,)` bias must become `(C, 1, 1)` to
be added per channel to a `(C, H, W)` image. They also adapt data to an API: a model wants a
batch `(1, H, W, C)`, you have one image `(H, W, C)`; a function returns `(N, 1)` and you want `(N,)`.

## How it works

```python
v = np.array([1, 2, 3])        # (3,)
v[None]                        # (1, 3)   row
v[:, None]                     # (3, 1)   column
np.expand_dims(v, 0)           # (1, 3)
np.expand_dims(v, (0, 2))      # (1, 3, 1)
```

```
v[:, None]:       v[None]:
1                 1 2 3
2
3
```

**Per-channel bias** for a batch `x` of shape `(N, C, H, W)`:

```
b                      (C,)
b[None, :, None, None] (1, C, 1, 1)    <- lines up with axis 1 of x
x + b[...]             (N, C, H, W)
```

To line a vector up with **any** axis of an `ndim`-dimensional array, build a shape of all 1s
with `len(v)` at that axis and `reshape`:

```python
shape = [1] * ndim
shape[axis] = len(v)
v.reshape(shape)               # e.g. ndim=4, axis=1 -> (1, C, 1, 1)
```

**squeeze**:

```python
x = np.zeros((1, 3, 1, 5))
np.squeeze(x).shape            # (3, 5)        all length-1 axes gone
np.squeeze(x, axis=2).shape    # (1, 3, 5)     only axis 2
np.squeeze(x, axis=1)          # ValueError: axis 1 has length 3
```

## Gotchas

- **Plain `squeeze()` eats your batch axis.** A batch of one image `(1, H, W, 1)` becomes
  `(H, W)`. Always pass `axis=` in code that handles batches.
- `x[None]` adds an axis at the **front**; `x[..., None]` at the **end**. They are easy to mix up
  for 1-D input, where both look plausible.
- PyTorch: `unsqueeze(dim)` = `expand_dims`; `squeeze(dim)` on a non-1 axis silently does
  nothing there, while NumPy raises. Indexing with `None` works the same in both.
- `np.expand_dims` with a negative axis counts positions in the **result**: `axis=-1` appends.

## Builds on

012, 014.

## Your task

Implement in `problem.py`, with **no Python loops over array elements**:

1. `ensure_batch_image(img)` - return a 4-D `(N, H, W, C)` view: `(H, W)` -> `(1, H, W, 1)`,
   `(H, W, C)` -> `(1, H, W, C)`, 4-D unchanged. Any other `ndim`: `ValueError`.
2. `squeeze_except(x, axis)` - remove every length-1 axis **except** `axis` (which is kept even if
   its length is 1). `axis` may be negative.
3. `broadcast_along(v, ndim, axis)` - reshape the 1-D `v` into an `ndim`-dimensional view that
   broadcasts along `axis` of an `ndim`-dimensional array (e.g. `(C,)` -> `(1, C, 1, 1)` for
   `ndim=4, axis=1`).

## Hints

<details>
<summary>Hint 1</summary>

`ensure_batch_image`: `img[None, ..., None]` for 2-D and `img[None]` for 3-D.

</details>

<details>
<summary>Hint 2</summary>

`squeeze_except`: build a tuple of the axes to remove (length 1 and not `axis`), then pass it to
`np.squeeze(x, axis=...)`. Normalize `axis` with `axis % x.ndim` first.

</details>

## Self-check

1. What shape is `np.zeros(5)[:, None, None]`?
2. Why is `np.squeeze(pred)` risky on a batch of predictions of shape `(N, 1)`?
3. How do you add a `(C,)` per-channel mean to an image of shape `(H, W, C)`? And `(C, H, W)`?
4. Do `x[None]` and `np.expand_dims(x, 0)` copy the data?

<details>
<summary>Answers</summary>

1. `(5, 1, 1)`.
2. When `N == 1` it removes the batch axis too, giving a 0-d array instead of shape `(1,)`.
3. `(H, W, C)`: just `img + mean`, it lines up with the last axis. `(C, H, W)`: `img + mean[:, None, None]`.
4. No, both return views.

</details>

## Go deeper

- [numpy.expand_dims](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
