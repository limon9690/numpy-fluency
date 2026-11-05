# 030 · NaN and inf: isfinite, nan* functions, errstate

**Category:** Random and statistics · **Difficulty:** Medium

## What it is

Floats have special values: `nan` ("not a number": missing or undefined) and `inf` / `-inf`
(too big, or division by zero). NumPy gives you tools to find them (`np.isnan`, `np.isinf`,
`np.isfinite`), to compute while ignoring NaN (`np.nanmean`, `np.nansum`, `np.nanmax`, ...), and
to control the warnings they cause (`np.errstate`).

## Why it exists

Real datasets have missing values, and NumPy marks them with NaN. NaN is "contagious": one NaN
makes a whole sum, mean or dot product NaN. You need to detect it, skip it or fill it, and
decide where warnings like "divide by zero" are a bug versus expected.

## How it works

```python
x = np.array([1.0, np.nan, 3.0, np.inf])
x.mean()               # nan         one NaN poisons the result
np.nanmean(x)          # inf         NaN skipped, inf still counts
np.isnan(x)            # [F, T, F, F]
np.isfinite(x)         # [T, F, T, F]   False for nan, inf and -inf
x[np.isfinite(x)].mean()   # 2.0
```

**Where they come from:**

| expression | result | warning |
|---|---|---|
| `1.0 / 0.0` (arrays) | `inf` | divide by zero |
| `0.0 / 0.0` | `nan` | invalid value |
| `np.inf - np.inf` | `nan` | invalid value |
| `np.log(0.0)` | `-inf` | divide by zero |
| `np.sqrt(-1.0)` | `nan` | invalid value |

**Comparisons with NaN are always False**, even `np.nan == np.nan`. Never test with `==`; use
`np.isnan`.

**Per-column statistics with NaN:** count the valid entries and sum them separately.

```
X:            valid:       nansum:     count:     mean:
 1  nan       T  F
 3  nan       T  F         [4, 0]      [2, 0]     [2, nan]   <- 0/0 must be handled
```

`np.nanmean` does this for you, but warns "Mean of empty slice" for an all-NaN column.

**errstate** turns the floating-point warnings off (or into errors) inside a `with` block:

```python
with np.errstate(divide="ignore", invalid="ignore"):
    r = a / b            # inf / nan where b == 0, but no warnings
with np.errstate(all="raise"):
    r = a / b            # FloatingPointError instead of a warning: good for debugging
```

It controls the ufunc floating-point warnings (`divide`, `invalid`, `over`, `under`). It does
**not** silence ordinary Python warnings such as `nanmean`'s "Mean of empty slice".

## Gotchas

- Integer arrays can't hold NaN. `np.array([1, 2])` followed by `a[0] = np.nan` raises
  `ValueError`. Missing values need a float dtype.
- `np.nan*` functions skip NaN but **not** inf.
- `np.errstate(all="ignore")` around a big block hides real bugs. Keep it tight around the line
  where inf/NaN is expected.
- `np.max` returns NaN if any value is NaN; `np.nanmax` skips it. `np.maximum(a, b)` propagates
  NaN, `np.fmax(a, b)` ignores it.
- PyTorch has `torch.isnan`, `torch.nanmean`, `torch.nansum`, but no `errstate`: it never warns
  on division by zero.

## Builds on

008, 009, 018, 019.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `nan_column_means(X)` - the mean of each column of a 2-D float array ignoring NaN; all-NaN
   columns give NaN. No warnings of any kind.
2. `fill_nan_with_column_mean(X)` - a copy where every NaN is replaced by its column's NaN-ignoring
   mean (0.0 for all-NaN columns).
3. `finite_stats(x)` - a dict `{"min", "max", "mean", "count"}` over the finite values only
   (no NaN, no ±inf), as Python numbers. `ValueError` if there are none.
4. `relative_change(old, new)` - `(new - old) / old` as float64, giving `inf`, `-inf` or `nan`
   where `old == 0`, without emitting warnings.

## Hints

<details>
<summary>Hint 1</summary>

`nan_column_means`: `valid = ~np.isnan(X)`, `sums = np.where(valid, X, 0).sum(axis=0)`,
`counts = valid.sum(axis=0)`, then divide with `out=`/`where=` (019).

</details>

<details>
<summary>Hint 2</summary>

`fill_nan_with_column_mean`: `np.where(np.isnan(X), means, X)` broadcasts the `(d,)` means
across the rows.

</details>

<details>
<summary>Hint 3</summary>

`relative_change`: put the division inside `with np.errstate(divide="ignore", invalid="ignore"):`.

</details>

## Self-check

1. What is `np.nan == np.nan`?
2. What do `np.nansum([np.nan, np.inf])` and `np.nanmean([np.nan, np.nan])` return?
3. Which warnings does `np.errstate` control, and which can't it silence?
4. Why can't you store NaN in an `int64` array?

<details>
<summary>Answers</summary>

1. `False`. Use `np.isnan`.
2. `inf` (inf isn't skipped), and `nan` with a "Mean of empty slice" `RuntimeWarning`.
3. Floating-point errors from ufuncs: divide, invalid, over, under. It can't silence warnings
   raised with Python's `warnings` module, like `nanmean`'s.
4. NaN is a special floating-point bit pattern; integers have no such value.

</details>

## Go deeper

- [numpy.errstate](https://numpy.org/doc/stable/reference/generated/numpy.errstate.html)
