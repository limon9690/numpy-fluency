# 017 · tile, repeat, pad, meshgrid

**Category:** Broadcasting and shape ops · **Difficulty:** Medium

## What it is

Four ways to build a bigger array from a smaller one:

- `np.repeat(a, k, axis)` - repeat each **element** `k` times in place: `1 2 -> 1 1 2 2`.
- `np.tile(a, reps)` - repeat the **whole** array: `1 2 -> 1 2 1 2`.
- `np.pad(a, pad_width, mode)` - add a border around an array.
- `np.meshgrid(x, y, indexing=...)` - coordinate grids from 1-D coordinate vectors.

## Why it exists

Nearest-neighbour upsampling is `repeat`. Making labels for a dataset (`0 0 0 1 1 1 2 2 2`) is
`repeat`; cycling through classes is `tile`. Images must often be padded to a size divisible by a
patch or stride. Coordinate grids let you evaluate a function at every pixel (distance from a
centre, a lens distortion, a Gaussian).

## How it works

**repeat vs tile:**

```python
a = np.array([1, 2, 3])
np.repeat(a, 2)          # [1 1 2 2 3 3]
np.tile(a, 2)            # [1 2 3 1 2 3]
```

On a 2-D image, repeating along both axes makes every pixel a `k × k` block:

```python
img = np.array([[1, 2],
                [3, 4]])
np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
```

```
1 1 2 2
1 1 2 2
3 3 4 4
3 3 4 4
```

Without `axis`, `repeat` flattens first: `np.repeat(img, 2)` is `[1 1 2 2 3 3 4 4]`.

**pad** takes one `(before, after)` pair **per axis**:

```python
np.pad(img, ((0, 1), (2, 0)))           # 1 row at the bottom, 2 columns on the left
```

```
0 0 1 2
0 0 3 4
0 0 0 0
```

For an `(H, W, C)` image, pad only the first two axes: `((top, bottom), (left, right), (0, 0))`.
Other modes: `"edge"`, `"reflect"`, `"constant"` with `constant_values=...`.

**meshgrid** and the `indexing` trap:

```python
ys = np.arange(3); xs = np.arange(4)
Y, X = np.meshgrid(ys, xs, indexing="ij")   # Y, X shape (3, 4): Y[i, j] = i, X[i, j] = j
```

```
Y:            X:
0 0 0 0       0 1 2 3
1 1 1 1       0 1 2 3
2 2 2 2       0 1 2 3
```

The **default** is `indexing="xy"` (from MATLAB): the outputs come out with shape `(4, 3)`,
transposed relative to the image. For image rows and columns, always pass `indexing="ij"`.

## Gotchas

- `repeat` and `tile` copy the data; for read-only repeats, `np.broadcast_to` (005) is free.
- `np.repeat(a, 2)` without `axis` flattens a 2-D array.
- `np.pad` with a single pair (`np.pad(img, (1, 1))`) pads **every** axis, including channels.
- `meshgrid` default `"xy"` swaps the first two output axes. A square test image hides this bug.
- PyTorch: `repeat_interleave` = `np.repeat`, but `Tensor.repeat` = `np.tile`. The names cross over.

## Builds on

012, 014, 016.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `upsample_nearest(img, k)` - every pixel becomes a `k × k` block; works for `(H, W)` and
   `(H, W, C)`; result `(k*H, k*W[, C])`.
2. `pad_to_multiple(img, m)` - pad with zeros at the bottom and right so `H` and `W` become
   multiples of `m` (no padding if they already are). Channels, if any, aren't padded.
3. `coordinate_grid(h, w)` - an `(h, w, 2)` int array with `out[i, j] = (i, j)`.
4. `repeat_and_cycle(labels, k)` - return `(each, cycled)`: each label repeated `k` times in place,
   and the whole array repeated `k` times.

## Hints

<details>
<summary>Hint 1</summary>

`upsample_nearest`: `np.repeat` twice, with `axis=0` and `axis=1`. The channel axis is untouched.

</details>

<details>
<summary>Hint 2</summary>

`pad_to_multiple`: the amount to add is `(-H) % m`. Build the pad width as a list with
`(0, 0)` for every extra axis.

</details>

<details>
<summary>Hint 3</summary>

`coordinate_grid`: `np.meshgrid(np.arange(h), np.arange(w), indexing="ij")`, then `np.stack(..., axis=-1)`.

</details>

## Self-check

1. What do `np.repeat([1, 2], 3)` and `np.tile([1, 2], 3)` return?
2. How much padding brings a height of 30 up to a multiple of 8? Of a height of 32?
3. Why can a bug with `meshgrid`'s default indexing pass unnoticed?
4. Which PyTorch method behaves like `np.tile`?

<details>
<summary>Answers</summary>

1. `[1 1 1 2 2 2]` and `[1 2 1 2 1 2]`.
2. `(-30) % 8 = 2`; `(-32) % 8 = 0`.
3. On square grids the transposed result has the right shape; only the values are swapped.
4. `Tensor.repeat`. (PyTorch's `repeat_interleave` is `np.repeat`.)

</details>

## Go deeper

- [numpy.pad](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
