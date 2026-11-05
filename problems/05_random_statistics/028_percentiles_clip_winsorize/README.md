# 028 · Percentiles, clip and winsorize

**Category:** Random and statistics · **Difficulty:** Medium

## What it is

- `np.percentile(x, q)` - the value below which `q` percent of the data lies (`q` in 0..100).
  `np.quantile(x, q)` is the same with `q` in 0..1. The median is the 50th percentile.
- `np.clip(x, lo, hi)` - squash values into `[lo, hi]`.
- **Winsorizing** = clipping at percentiles: e.g. replace everything below the 5th percentile
  by the 5th percentile, and everything above the 95th by the 95th.

## Why it exists

Real data has outliers: a sensor glitch, a typo, one huge transaction. The mean and std react
strongly to them; percentiles and the median barely move. So robust statistics (median,
interquartile range) and outlier handling (clip, winsorize) are built from percentiles.

## How it works

```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
np.percentile(x, [25, 50, 75])      # [3.25, 5.5, 7.75]
```

Between data points NumPy interpolates linearly by default. `method="lower"`, `"higher"`,
`"nearest"` return an actual data value instead (`np.percentile(x, 25, method="lower")` is `3`).

**IQR and Tukey's fences.** The interquartile range is `IQR = Q3 - Q1`. A common rule calls a
value an outlier if it's below `Q1 - 1.5·IQR` or above `Q3 + 1.5·IQR`:

```
 Q1 - 1.5 IQR     Q1        median       Q3      Q3 + 1.5 IQR
------|-----------[=========|==========]-----------|------
  outliers   <-        middle 50%         ->         outliers
```

**Per column**: `axis=0` gives one percentile per column, and `keepdims=True` makes it broadcast
back (018):

```python
X = np.array([[1,  2],
              [3, 40]])
np.percentile(X, 50, axis=0, keepdims=True)    # [[ 2., 21.]]  shape (1, 2)
```

**clip** takes scalars or arrays (they broadcast), and `None` for "no bound":

```python
np.clip([1, 5, 9], 2, 6)          # [2 5 6]
np.clip([1, 5, 9], None, 4)       # [1 4 4]
```

**Winsorize** = `np.clip(x, np.percentile(x, 5), np.percentile(x, 95))`. Unlike dropping
outliers, it keeps the array length, so it stays aligned with other arrays.

**Robust scaling**: `(x - median) / IQR` per column. Like standardizing (013), but one wild
value can't stretch the scale.

**Quantile bins**: the percentiles at `0, 1/n, 2/n, ..., 1` cut the data into `n` groups of
roughly equal size ("deciles" for `n = 10`). `np.searchsorted` (023) assigns each value its group.

## Gotchas

- `percentile` wants 0..100 and `quantile` wants 0..1. `np.percentile(x, 0.95)` is almost the
  minimum, not the 95th percentile.
- The old `interpolation=` keyword is gone in recent NumPy 2; use `method=`.
- NaN anywhere makes the result NaN; use `np.nanpercentile` (030).
- `np.clip` doesn't modify `x` unless you pass `out=x`.
- PyTorch: `torch.quantile` (0..1 only) and `torch.clamp`/`torch.clip`.

## Builds on

013, 018, 023.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `iqr_outlier_mask(x)` - True where a value of the 1-D `x` is outside Tukey's fences
   `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` (default linear percentiles).
2. `winsorize(x, lower_pct, upper_pct, axis)` - clip at the `lower_pct` and `upper_pct`
   percentiles, over the whole array (`axis=None`) or per slice along `axis`.
3. `robust_scale(X)` - per column `(x - median) / IQR`; a column with `IQR == 0` is only
   centered (divide by 1).
4. `quantile_bins(x, n_bins)` - a bin number in `0..n_bins-1` for each value: bin `k` holds
   values between the `k/n` and `(k+1)/n` quantiles (each bin includes its lower edge).

## Hints

<details>
<summary>Hint 1</summary>

`q1, q3 = np.percentile(x, [25, 75])` unpacks both at once.

</details>

<details>
<summary>Hint 2</summary>

`winsorize`: `np.percentile(x, [lower_pct, upper_pct], axis=axis, keepdims=True)` returns both
bounds, each shaped to broadcast against `x`. Then `np.clip`.

</details>

<details>
<summary>Hint 3</summary>

`quantile_bins`: `edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))`, then
`np.searchsorted(edges[1:-1], x, side="right")`.

</details>

## Self-check

1. What does `np.percentile(x, 0.95)` compute?
2. Why are the median and IQR called "robust"?
3. Why winsorize instead of deleting outliers?
4. What shape does `np.percentile(X, [5, 95], axis=0)` have for `X` of shape `(100, 3)`?

<details>
<summary>Answers</summary>

1. The 0.95th percentile, close to the minimum (you wanted `95` or `np.quantile(x, 0.95)`).
2. A few extreme values barely change them, unlike the mean and std.
3. It keeps every row, so the array stays aligned with labels and other features.
4. `(2, 3)`: one row per requested percentile.

</details>

## Go deeper

- [numpy.percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
