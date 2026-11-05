# 009 · where, nonzero, argwhere

**Category:** Indexing · **Difficulty:** Easy

## What it is

Three tools built on masks:

- `np.where(cond, a, b)` - an element-wise "if/else": take from `a` where `cond` is True,
  else from `b`.
- `np.nonzero(mask)` - the **positions** of the True elements, as one index array per axis.
- `np.argwhere(mask)` - the same positions as a `(k, ndim)` table, one row per True element.

## Why it exists

Masking (008) answers "which values?". These answer "build a new array from a condition" and
"**where** are they?" You need positions to crop to an object, find the first event in a signal,
or report which samples failed a check.

## How it works

**`np.where` with 3 arguments** (all three broadcast together):

```python
x = np.array([-2, 3, -1, 4])
np.where(x > 0, x, 0)        # array([0, 3, 0, 4])   (a ReLU)
```

Nested `where`s or `np.select` handle more than two cases:

```python
np.select([x < 0, x < 2], [-1, 0], default=1)    # -1 if x<0, else 0 if x<2, else 1
```

**`np.nonzero` / `np.argwhere`** on a 2-D mask:

```python
m = np.array([[0, 1, 0],
              [1, 0, 1]], dtype=bool)
```

```
np.nonzero(m)   ->  (array([0, 1, 1]),      rows
                     array([1, 0, 2]))      cols

np.argwhere(m)  ->  0 1                    one (row, col) per True
                    1 0
                    1 2
```

The `nonzero` tuple is ready to use as a fancy index: `x[np.nonzero(m)]` equals `x[m]`.
`argwhere` is easier to read and to reduce: `np.argwhere(m).min(axis=0)` is the smallest row and
smallest column that contain a True. That gives a **bounding box**.

`np.where(cond)` with **one** argument is the same as `np.nonzero(cond)`.
`np.flatnonzero(m)` gives positions in the flattened array.

## Gotchas

- `np.where(b != 0, a / b, 0)` still computes `a / b` **everywhere** and warns about division by
  zero. NumPy evaluates both branches before choosing. Make the denominator safe first, or use
  `np.divide(a, b, out=..., where=b != 0)` (019).
- `np.argmax(mask)` returns `0` when nothing is True, the same as when element 0 is True. Check
  `mask.any()` (or use `flatnonzero`) before trusting it.
- `nonzero` returns a **tuple** even for 1-D input: `np.nonzero(v)[0]`.
- `np.where(cond, x, 0)` follows promotion rules: an int `x` with fill `0.5` gives float64.
- PyTorch: `torch.where(cond, a, b)` is the same; `torch.nonzero` returns the `argwhere` layout
  `(k, ndim)`, not NumPy's tuple.

## Builds on

006, 007, 008.

## Your task

Implement in `problem.py`, with **no Python loops over elements** (in `bounding_box`, building
the tuple of slices with a loop over the axes is fine):

1. `safe_divide(a, b, fill)` - `a / b` as float64, with `fill` where `b == 0`. Must not emit any
   division warnings.
2. `bounding_box(mask)` - a tuple of slices, one per axis, so that `x[box]` is the smallest
   box containing every True of `mask`. `None` if there is no True. Any number of dimensions.
3. `first_true(mask)` - index of the first True in a 1-D mask, or `-1` if there is none.
4. `piecewise_curve(x)` - `x**2` where `x < 0`, `x` where `0 <= x < 1`, and `1` where `x >= 1`.

## Hints

<details>
<summary>Hint 1</summary>

For `safe_divide`, replace zeros in the denominator before dividing:
`safe_b = np.where(b == 0, 1, b)`, then use `np.where` again to put `fill` back.

</details>

<details>
<summary>Hint 2</summary>

For `bounding_box`, `idx = np.argwhere(mask)`; `idx.min(axis=0)` and `idx.max(axis=0)` give the
corners. The stop of a slice is exclusive, so add 1.

</details>

<details>
<summary>Hint 3</summary>

For `first_true`, `np.flatnonzero(mask)` is empty when there is no True.

</details>

## Self-check

1. Why does `np.where(b != 0, a / b, 0)` still warn about dividing by zero?
2. What is the difference between the outputs of `np.nonzero` and `np.argwhere`?
3. What does `np.argmax(np.zeros(5, bool))` return, and why is that a trap?
4. What does `np.where(m)` do with a single argument?

<details>
<summary>Answers</summary>

1. Both branches are full arrays computed before `where` picks; `a / b` runs on every element.
2. `nonzero` gives a tuple of `ndim` index arrays (one per axis); `argwhere` gives one
   `(k, ndim)` array with a row per True element.
3. `0`, which looks like "element 0 is True".
4. The same as `np.nonzero(m)`.

</details>

## Go deeper

- [numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
