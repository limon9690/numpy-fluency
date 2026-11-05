# 014 · reshape, ravel vs flatten, transpose, moveaxis

**Category:** Broadcasting and shape ops · **Difficulty:** Easy

## What it is

These functions change **how you look at** the data, not the data itself:

- `reshape` - same elements, read in C order, poured into a new shape.
- `ravel` / `flatten` - make it 1-D. `ravel` returns a view when it can; `flatten` always copies.
- `transpose` / `.T` / `np.moveaxis` / `swapaxes` - reorder the axes.

## Why it exists

Different code wants different layouts: images are `(H, W, C)` in NumPy/PIL/OpenCV but
`(C, H, W)` in PyTorch; a linear layer wants `(batch, features)`; a vision transformer wants an
image cut into patches. Shape ops convert between them, usually for free (a view, 004-005).

## How it works

**reshape reads and writes in C order** (last axis fastest):

```python
np.arange(6).reshape(2, 3)        # 0 1 2 / 3 4 5
np.arange(6).reshape(3, -1)       # -1 = "work it out": (3, 2)
```

```
(2, 3):       (3, 2):
0 1 2         0 1
3 4 5         2 3
              4 5
```

Reshape is **not** transpose: `(2, 3) -> (3, 2)` keeps the reading order; `.T` swaps rows and
columns.

**moveaxis** moves one axis to a new position and keeps the others in order:

```python
img = np.zeros((224, 224, 3))     # H, W, C
np.moveaxis(img, -1, 0).shape     # (3, 224, 224)   C, H, W
img.transpose(2, 0, 1).shape      # same thing, all axes listed
```

Both are views with swapped strides, so the result is usually **not** C-contiguous.

**Cutting an image into patches** is "reshape, transpose, reshape". For a `4 × 4` image and
`2 × 2` patches:

```
image (4, 4)            reshape (2, 2, 2, 2)       swap axes 1 and 2      reshape (4, 2, 2)
 0  1 |  2  3           (patch row, row in patch,  (patch row, patch col,  4 patches:
 4  5 |  6  7            patch col, col in patch)   row in patch, col)     [0 1; 4 5], [2 3; 6 7],
------+------                                                              [8 9; 12 13], [10 11; 14 15]
 8  9 | 10 11
12 13 | 14 15
```

A plain `reshape(4, 2, 2)` would give `[0 1; 2 3]` as the first "patch": wrong, because it keeps
C reading order across the whole row.

**ravel vs flatten:**

```python
a = np.arange(6).reshape(2, 3)
a.ravel()          # view (a is contiguous)
a.T.ravel()        # copy! a.T's elements aren't in C order in memory
a.flatten()        # always a copy
a.flat[4]          # element 4 in C order, always writes into a
```

## Gotchas

- `x.reshape(len(x), -1)` fails when `len(x) == 0`: with 0 elements, `-1` could be anything.
  Compute the size explicitly.
- `ravel()` may be a view **or** a copy; writing into it may or may not change the original.
  `a.flat[i] = v` or `a[np.unravel_index(i, a.shape)] = v` always write into `a`.
- `reshape` after `.T` copies (the memory order doesn't match). Fine for reading, a trap for writing.
- PyTorch: `view` refuses to copy and errors on non-contiguous input; `reshape` copies if needed,
  like NumPy's. `permute` is NumPy's `transpose`.

## Builds on

004, 005.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `to_patches(img, p)` - `(H, W)` image, `H` and `W` multiples of `p` -> `(H//p * W//p, p, p)`,
   patches in row-major order of the patch grid.
2. `hwc_to_chw(img)` - `(H, W, C)` -> `(C, H, W)`, and `(N, H, W, C)` -> `(N, C, H, W)`. Return a view.
3. `flatten_batch(x)` - `(B, ...)` -> `(B, rest)` where `rest` is the product of the other axes.
   Must work for `B == 0` and for 1-D input (gives `(B, 1)`).
4. `increment_flat(a, i, value)` - in place, add `value` to the element at C-order flat
   index `i` of `a`, even when `a` is not contiguous. Return `None`.

## Hints

<details>
<summary>Hint 1</summary>

`to_patches`: `img.reshape(H // p, p, W // p, p)`, then `swapaxes(1, 2)`, then reshape to `(-1, p, p)`.

</details>

<details>
<summary>Hint 2</summary>

`hwc_to_chw`: `np.moveaxis(img, -1, -3)` covers both the 3-D and the 4-D case.

</details>

<details>
<summary>Hint 3</summary>

`flatten_batch`: `math.prod(x.shape[1:])` is 1 for an empty tuple. `increment_flat`: `a.flat`.

</details>

## Self-check

1. What is the difference between `a.reshape(3, 2)` and `a.T` for `a` of shape `(2, 3)`?
2. When does `ravel()` return a copy?
3. Why does `np.zeros((0, 4, 5)).reshape(0, -1)` fail?
4. Is `np.moveaxis(img, -1, 0)` C-contiguous?

<details>
<summary>Answers</summary>

1. Reshape keeps C reading order (`0 1 / 2 3 / 4 5`); `.T` swaps rows and columns (`0 3 / 1 4 / 2 5`).
2. When the elements aren't laid out in C order in memory (e.g. after `.T` or a step slice).
3. 0 elements fit any second length, so `-1` is ambiguous.
4. No: it's a view with reordered strides. Use `np.ascontiguousarray` if you need contiguous memory.

</details>

## Go deeper

- [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
