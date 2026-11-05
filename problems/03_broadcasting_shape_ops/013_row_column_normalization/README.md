# 013 · Row and column scaling and normalization ★

**Category:** Broadcasting and shape ops · **Difficulty:** Easy

## What it is

Normalizing means rescaling each row or each column with its own statistic: divide each row by
its sum, standardize each column to mean 0 and std 1, squeeze each column into [0, 1]. In NumPy
it's always the same two steps: **reduce along an axis**, then **broadcast the result back**.

## Why it exists

Features measured in different units (metres, grams, counts) must be put on the same scale
before you compare or combine them. Count tables become probabilities by normalizing rows.
Images are standardized per channel. This is everyday preprocessing.

## How it works

Data with 3 rows (samples) and 2 columns (features):

```
a =  1  10
     2  20
     3  60
```

**Columns** are reduced with `axis=0` (you collapse the rows):

```python
a.mean(axis=0)            # [ 2., 30.]   shape (2,)
a - a.mean(axis=0)        # (3, 2) - (2,) -> broadcasts along rows, works
```

```
-1  -20
 0  -10
 1   30
```

**Rows** are reduced with `axis=1`, and here is the trap:

```python
a.sum(axis=1)                       # [11, 22, 63]  shape (3,)
a / a.sum(axis=1)                   # (3, 2) / (3,) -> ValueError
a / a.sum(axis=1, keepdims=True)    # (3, 2) / (3, 1) -> works
```

`keepdims=True` keeps the reduced axis with length 1, so the result lines up with the axis it
came from. `a.sum(axis=1)[:, None]` does the same thing.

```
row sums (3, 1):     a / row sums:
11                   0.0909  0.9091
22                   0.0909  0.9091
63                   0.0476  0.9524
```

**Division by zero.** A constant column has std 0; an all-zero row has sum 0. Replace those
denominators with 1 before dividing (the numerators there are 0 anyway after centering, or for
an all-zero row), so you get 0 instead of NaN and a warning.

```python
std = a.std(axis=0)
safe = np.where(std == 0, 1, std)
```

## Gotchas

- **The square-matrix trap.** For an `(n, n)` matrix and weights `w` of shape `(n,)`,
  `a * w` scales the **columns**, not the rows, and raises no error. Use `w[:, None]`.
- NumPy's `axis=0` means "reduce over rows, one result per column", the same as PyTorch's `dim=0`.
- `np.std` uses `ddof=0` (population) by default; `torch.std` uses the sample std (`correction=1`).
  More in 029.
- Integer input: `a / something` gives float64. Float32 input stays float32 (002).

## Builds on

012. Uses `keepdims=True`, introduced here and covered fully in 018.

## Your task

Implement in `problem.py`, with **no Python loops**. Results are float (float64 for integer
input; float32 input stays float32). The input is never modified.

1. `row_normalize(a)` - divide each row of a non-negative 2-D array by its sum; all-zero rows
   stay all zero.
2. `standardize_columns(a)` - each column to mean 0, std 1 (`ddof=0`); constant columns become 0.
3. `minmax_scale_columns(a)` - each column to [0, 1]: `(x - min) / (max - min)`; constant
   columns become 0.
4. `scale_rows(a, w)` - multiply row `i` of `a` by `w[i]`.

## Hints

<details>
<summary>Hint 1</summary>

Write the reduction with `keepdims=True` every time; then the broadcast always lines up and you
don't have to think about which axis needs a `None`.

</details>

<details>
<summary>Hint 2</summary>

Guard each denominator with `np.where(d == 0, 1, d)` before dividing.

</details>

## Self-check

1. For `a` of shape `(100, 5)`, what shapes do `a.mean(axis=0)` and `a.mean(axis=1, keepdims=True)` have?
2. Why does `a / a.sum(axis=1)` fail for a `(3, 2)` array, but silently do the wrong thing for `(3, 3)`?
3. Why replace a zero std with 1 rather than adding a tiny epsilon?
4. What dtype do you get when you standardize a float32 array?

<details>
<summary>Answers</summary>

1. `(5,)` and `(100, 1)`.
2. `(3,)` is aligned with the **last** axis (columns). Length 2 vs 3 fails; 3 vs 3 "works" but
   divides each column by a row sum.
3. The numerator is exactly 0 there, so 0/1 gives the intended 0 and the other columns aren't
   changed at all. (An epsilon is also common; it slightly changes every result.)
4. float32: the mean and std of float32 data are float32.

</details>

## Go deeper

- [numpy.mean, keepdims](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
