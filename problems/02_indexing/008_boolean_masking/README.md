# 008 · Boolean masking ★

**Category:** Indexing · **Difficulty:** Medium

## What it is

A **mask** is a boolean array, usually made by a comparison like `x > 0`. Indexing with it,
`x[mask]`, keeps the elements where the mask is `True`. Assigning through it, `x[mask] = 0`,
changes exactly those elements. You combine conditions with `&` (and), `|` (or) and `~` (not).

## Why it exists

"Keep the valid samples", "clip the outliers", "drop rows with NaN", "zero the negative pixels":
data cleaning is mostly masking. It replaces an `if` inside a loop with one vectorized step.

## How it works

```python
x = np.array([[ 3, -1,  7],
              [-4,  5, -2]])
mask = x > 0
```

```
mask:              x[mask]:
 True False  True   [3, 7, 5]     <- 1-D, in row-major order
False  True False
```

**Selecting** with a mask of the same shape always gives a **1-D copy**: NumPy can't keep a 2-D
shape when rows keep different numbers of elements.

**Assigning** writes in place and keeps the shape:

```python
x[x < 0] = 0
```

```
3 0 7
0 5 0
```

**Combining conditions** needs `&`, `|`, `~` and **parentheses**:

```python
(x > 0) & (x < 6)      # correct
x > 0 & x < 6          # wrong: & binds tighter, this is x > (0 & x) < 6 -> error
(x > 0) and (x < 6)    # ValueError: truth value of an array is ambiguous
```

**Row masks.** A 1-D mask on axis 0 selects whole rows and keeps the other axes:

```python
data = np.array([[1, 50],
                 [2, 10],
                 [3, 70]])
data[data[:, 1] > 20]  # rows where column 1 > 20
```

```
1 50
3 70
```

**Counting.** `True` counts as 1: `mask.sum()` or `np.count_nonzero(mask)` counts matches,
and `mask.mean()` is the fraction of matches.

**NaN.** `NaN == NaN` is `False`, so you can't find NaNs with `==`. Use `np.isnan(x)`.

## Gotchas

- `and`, `or`, `not` don't work on arrays; use `&`, `|`, `~` with parentheses.
- `~` on an **int** array is bitwise NOT (`~1 == -2`), not logical NOT. Use it on booleans only.
- `x[mask]` is a copy; `x[mask][0] = 5` changes nothing (004).
- The mask must match the shape of the axes it indexes, or you get an `IndexError`.
- PyTorch: same syntax (`t[mask]`), also returns 1-D. Masks there should be `torch.bool`
  (uint8 masks are deprecated).

## Builds on

006, 007.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `count_in_range(x, low, high)` - how many elements satisfy `low <= x <= high`.
2. `replace_outside(x, low, high, fill)` - a copy of `x` where values below `low` or above
   `high` are replaced by `fill`. Same shape and dtype; `x` unchanged.
3. `select_rows(data, col, threshold)` - the rows of a 2-D array whose value in column `col` is
   greater than `threshold`.
4. `drop_nan_pairs(x, y)` - two 1-D arrays of equal length; drop every position where `x` or `y`
   is NaN, and return the two cleaned arrays.

## Hints

<details>
<summary>Hint 1</summary>

Build the mask first and give it a name. `inside = (x >= low) & (x <= high)`.

</details>

<details>
<summary>Hint 2</summary>

For `replace_outside`: `out = x.copy()`, then assign through the mask on `out`. `np.where` (009)
also works.

</details>

<details>
<summary>Hint 3</summary>

For `drop_nan_pairs`: `keep = ~(np.isnan(x) | np.isnan(y))`, then index both arrays with `keep`.

</details>

## Self-check

1. What shape does `x[x > 0]` have when `x` is `(3, 4)`?
2. Why does `x > 0 & x < 6` fail?
3. How do you compute the fraction of positive elements in one expression?
4. Why doesn't `x[x == np.nan]` find any NaNs?

<details>
<summary>Answers</summary>

1. 1-D, with as many elements as there are `True` values.
2. `&` has higher precedence than `>`, so Python evaluates `0 & x` first, then a chained comparison.
3. `(x > 0).mean()`.
4. NaN is not equal to anything, itself included; use `np.isnan(x)`.

</details>

## Go deeper

- [Boolean array indexing](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing)
