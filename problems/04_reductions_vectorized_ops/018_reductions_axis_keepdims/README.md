# 018 · Reductions with axis and keepdims ★

**Category:** Reductions and vectorized ops · **Difficulty:** Easy

## What it is

A **reduction** turns many numbers into fewer: `sum`, `mean`, `max`, `min`, `std`, `argmax`,
`any`, `all`. The `axis` argument says which axis disappears. `keepdims=True` keeps that axis
with length 1, so the result still broadcasts against the input.

## Why it exists

Almost every statistic is a reduction: the mean of each feature, the max of each row, the
brightness of each image. Doing it in one call along an axis is far faster than a Python loop,
and `keepdims` lets you use the result right away ("subtract each row's max").

## How it works

```python
a = np.array([[1, 5, 2],
              [7, 0, 3]])        # shape (2, 3)
```

| call | result | shape |
|---|---|---|
| `a.sum()` | `18` | `()` |
| `a.sum(axis=0)` | `[8, 5, 5]` | `(3,)` - one per column |
| `a.sum(axis=1)` | `[8, 10]` | `(2,)` - one per row |
| `a.max(axis=1, keepdims=True)` | `[[5], [7]]` | `(2, 1)` |
| `a.argmax()` | `3` | flat index! |

```
axis=0 collapses DOWN the rows       axis=1 collapses ACROSS the columns
  1  5  2                              1  5  2  -> 8
  7  0  3                              7  0  3  -> 10
  |  |  |
  8  5  5
```

**Several axes at once:** `axis=(0, 1, 2)` on a batch of images `(N, H, W, C)` gives one number
per channel, shape `(C,)`. `axis=tuple(range(1, x.ndim))` reduces "everything but the batch".

**keepdims for broadcasting:**

```python
m = a.max(axis=1, keepdims=True)    # (2, 1)
a - m                               # (2, 3): each row minus its own max
```

```
-4  0 -3
 0 -7 -4
```

**Stable softmax** is the classic use: `exp` of large numbers overflows, so subtract each row's
max first. The result is mathematically the same, because the factor `exp(-max)` cancels.

**argmax gives a flat index** when you don't pass `axis`. Turn it into coordinates with
`np.unravel_index(i, a.shape)`: index 3 in a `(2, 3)` array is `(1, 0)`.

## Gotchas

- NumPy's `axis` = PyTorch's `dim`; `keepdims` = PyTorch's `keepdim` (no "s").
- `a.max(axis=1)` returns values only. PyTorch's `t.max(dim=1)` returns `(values, indices)`.
- `np.max(a, b)` does **not** compare two arrays (the second argument is `axis`). For an
  element-wise maximum use `np.maximum(a, b)`.
- `a.max()` on an empty array raises `ValueError`; `a.sum()` gives 0.
- Sums of small ints are accumulated in a wider int (`uint8.sum()` is `uint64`), but
  `a.sum(dtype=np.uint8)` would wrap (002).
- `np.std` defaults to `ddof=0`; PyTorch's `std` to `correction=1` (029).

## Builds on

012, 013, 015.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `softmax(x, axis)` - numerically stable softmax along `axis` (subtract the max first).
   Float input keeps its dtype; int input gives float64.
2. `argmax_2d(a)` - the `(row, col)` of the largest element of a 2-D array, as Python ints
   (first one on ties).
3. `channel_mean_std(images)` - per-channel mean and std (`ddof=0`) of an `(N, H, W, C)` batch;
   two float64 arrays of shape `(C,)`.
4. `center_per_sample(x)` - subtract from each `x[i]` its own mean over all its elements; any
   `ndim >= 1`, float64 result, same shape.

## Hints

<details>
<summary>Hint 1</summary>

`softmax`: `z = x - x.max(axis=axis, keepdims=True)`, then `e = np.exp(z)` and
`e / e.sum(axis=axis, keepdims=True)`.

</details>

<details>
<summary>Hint 2</summary>

`argmax_2d`: `np.unravel_index(a.argmax(), a.shape)`, then convert each part with `int()`.

</details>

<details>
<summary>Hint 3</summary>

`channel_mean_std`: `axis=(0, 1, 2)` and `dtype=np.float64`. `center_per_sample`:
`axis=tuple(range(1, x.ndim))` with `keepdims=True`.

</details>

## Self-check

1. For `a` of shape `(4, 5, 6)`, what is the shape of `a.sum(axis=(0, 2))`? With `keepdims=True`?
2. Why subtract the max before `exp` in softmax, and why doesn't it change the answer?
3. What does `np.argmax` return for a 2-D array when you don't pass `axis`?
4. What does `np.max(a, 0)` do?

<details>
<summary>Answers</summary>

1. `(5,)`; with `keepdims=True`, `(1, 5, 1)`.
2. `exp(1000)` overflows to inf. `exp(x - m) / sum(exp(x - m))` equals the original because the
   factor `exp(-m)` appears in numerator and denominator.
3. A flat index into the C-ordered array; use `np.unravel_index` for coordinates.
4. The max along axis 0. It does not compare `a` with 0; that is `np.maximum(a, 0)`.

</details>

## Go deeper

- [numpy.ufunc.reduce](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html)
