# 006 · Slicing and basic indexing

**Category:** Indexing · **Difficulty:** Easy

## What it is

**Basic indexing** uses integers, slices (`start:stop:step`), `...` (Ellipsis) and `None`.
In a multi-dimensional array you give one index per axis, separated by commas: `a[1, 2:5]`.
Basic indexing always returns a **view** (004).

## Why it exists

Most array work is "take this region": crop an image, drop the first row, take every second
sample, reverse a signal. Slicing does all of these without copying data, so it's both the most
common and the cheapest operation in NumPy.

## How it works

```python
a = np.arange(20).reshape(4, 5)
```

```
     col 0  1  2  3  4
row 0:   0  1  2  3  4
row 1:   5  6  7  8  9
row 2:  10 11 12 13 14
row 3:  15 16 17 18 19
```

| expression | result | shape |
|---|---|---|
| `a[1]` | row 1 | `(5,)` |
| `a[:, 1]` | column 1 | `(4,)` |
| `a[1:3, 1:4]` | rows 1-2, cols 1-3 | `(2, 3)` |
| `a[::2, ::2]` | every other row and col | `(2, 3)` |
| `a[::-1]` | rows in reverse order | `(4, 5)` |
| `a[-1]` | last row | `(5,)` |
| `a[1:2]` | row 1, kept 2-D | `(1, 5)` |

`a[1:3, 1:4]` picks:

```
 6  7  8
11 12 13
```

**Ellipsis** means "all the axes I didn't write": for an image batch `x` of shape `(N, H, W, C)`,
`x[..., 0]` is channel 0 of every image, shape `(N, H, W)`.

**Slice objects.** `a[1:3]` is really `a[slice(1, 3)]`, and `a[1:3, ::-1]` is
`a[(slice(1, 3), slice(None, None, -1))]`. Building that tuple in code lets you slice along an
axis chosen at runtime:

```python
idx = [slice(None)] * a.ndim        # ":" for every axis
idx[axis] = slice(None, None, -1)   # "::-1" on one axis
a[tuple(idx)]
```

**Assignment through a slice** writes into the array: `a[::2, ::2] = 0` zeroes those positions.

## Gotchas

- An integer index **drops** an axis (`a[1]` is 1-D); a length-1 slice keeps it (`a[1:2]` is 2-D).
- `a[-k:]` with `k = 0` is `a[-0:]` = `a[0:]` = **the whole array**, not an empty one.
- Out-of-range slices don't raise: `a[10:20]` is just empty. Out-of-range integers do raise.
- `a[1][2]` works but makes an intermediate array; write `a[1, 2]`.
- PyTorch supports positive steps only: `t[::-1]` is an error there (use `torch.flip`, which
  copies). In NumPy, `a[::-1]` is a free view.

## Builds on

004.

## Your task

Implement in `problem.py` (no Python loops; `np.flip` is banned):

1. `crop_center(img, h, w)` - the central `h × w` region of a `(H, W)` or `(H, W, C)` array, as a
   view. Top-left corner at `((H - h) // 2, (W - w) // 2)`.
2. `flip_axis(a, axis)` - `a` reversed along `axis` (negative allowed), as a view.
3. `checkerboard(n)` - an `n × n` int8 array of 0s and 1s, 0 in the top-left corner.
4. `last_k(a, k)` - the last `k` entries along axis 0, as a view; `k = 0` gives an empty array.

## Hints

<details>
<summary>Hint 1</summary>

`crop_center`: compute `top` and `left`, then `img[top:top + h, left:left + w]`. The channel axis,
if there is one, is untouched by a 2-index slice.

</details>

<details>
<summary>Hint 2</summary>

`checkerboard`: start from `np.zeros((n, n), np.int8)`, then assign 1 with two step-2 slices:
odd columns of even rows, and even columns of odd rows.

</details>

<details>
<summary>Hint 3</summary>

`last_k`: avoid negative starts. `a[len(a) - k:]` works for every `k` from 0 to `len(a)`.

</details>

## Self-check

1. What are the shapes of `a[2]`, `a[2:3]` and `a[:, 2]` for `a` of shape `(4, 5)`?
2. What does `x[..., ::-1]` do to an `(H, W, 3)` RGB image?
3. Why is `a[-k:]` a bug when `k` can be 0?
4. Is `a[::2]` a view or a copy?

<details>
<summary>Answers</summary>

1. `(5,)`, `(1, 5)` and `(4,)`.
2. It reverses the last axis: RGB becomes BGR.
3. `-0` is `0`, so `a[-0:]` is the whole array instead of nothing.
4. A view: basic slicing never copies.

</details>

## Go deeper

- [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html)
