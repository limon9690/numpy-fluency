# 029 · var and std with ddof, cov, corrcoef

**Category:** Random and statistics · **Difficulty:** Medium

## What it is

- `np.var` / `np.std` - spread of the data. `ddof` ("delta degrees of freedom") chooses the
  divisor: `n - ddof`. `ddof=0` (NumPy's default) is the population formula; `ddof=1` is the
  sample formula.
- `np.cov` - the covariance matrix: how every pair of variables varies together.
- `np.corrcoef` - the correlation matrix: covariance rescaled to [-1, 1].

## Why it exists

You constantly summarize data: "how noisy is this measurement?", "do these two features move
together?". The trap isn't the math, it's the conventions: which divisor, and whether variables
are rows or columns. Different libraries pick differently, and the numbers silently disagree.

## How it works

```python
x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])   # mean 5, sum of squared deviations 32
np.var(x)            # 32 / 8 = 4.0     ddof=0
np.var(x, ddof=1)    # 32 / 7 = 4.571   ddof=1
np.std(x)            # 2.0
```

Why `n - 1`? The deviations are measured from the **sample** mean, which is pulled toward the
data, so they come out a bit too small. Dividing by `n - 1` corrects that on average (an
"unbiased" variance estimate). For large `n` the difference hardly matters.

**Covariance matrix** of data `X` with shape `(n_samples, d)`:

```
Xc = X - X.mean(axis=0)          center every column
C  = Xc.T @ Xc / (n - 1)         shape (d, d)
C[i, i] = variance of column i,  C[i, j] = covariance of columns i and j
```

**Correlation** divides by the standard deviations: `R[i, j] = C[i, j] / (std_i * std_j)`.
The diagonal is all 1s. In matrix form: `R = C / np.outer(s, s)` with `s = sqrt(diag(C))`.

**The rows-vs-columns trap:** `np.cov(X)` treats every **row** as a variable (`rowvar=True`).
For the usual `(n_samples, d)` layout you need `np.cov(X, rowvar=False)` or `np.cov(X.T)`,
otherwise you get an `(n, n)` matrix of "sample vs sample" covariances.

**Correlation of matching columns.** For two arrays `x`, `y` of shape `(n, d)`, the
correlation between `x[:, j]` and `y[:, j]` for all `j` at once is

```
mean over rows of (standardized x) * (standardized y)        with ddof=0 standardization
```

No `(2d, 2d)` matrix needed.

## Gotchas

- **Defaults differ.** `np.var`/`np.std`: `ddof=0`. `np.cov`: divides by `n - 1`.
  `torch.var`/`torch.std`: `correction=1` (sample). pandas `.std()`: `ddof=1`.
- `np.cov` and `np.corrcoef` assume **rows are variables** unless `rowvar=False`.
- The `n - 1` cancels out in a correlation, so `ddof` doesn't matter for `corrcoef`, as long as
  you use the same `ddof` on top and bottom.
- A constant column has std 0, so its correlations are 0/0 = NaN (with a warning).
- `ddof=1` with a single sample divides by 0.

## Builds on

013, 018. (`Xc.T @ Xc` is a plain matrix product, the same `@` as in PyTorch.)

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `sample_std(x, axis)` - the sample standard deviation (`n - 1` divisor) along `axis`.
2. `covariance_matrix(X)` - the `(d, d)` sample covariance of `X` `(n, d)`, columns = variables.
   `np.cov` is banned.
3. `correlation_matrix(X)` - the `(d, d)` correlation matrix of the columns. `np.cov` and
   `np.corrcoef` are banned. (No constant columns in the tests.)
4. `columnwise_corr(x, y)` - shape `(d,)`: correlation of `x[:, j]` with `y[:, j]`.

## Hints

<details>
<summary>Hint 1</summary>

`sample_std` is `np.std(x, axis=axis, ddof=1)`.

</details>

<details>
<summary>Hint 2</summary>

`covariance_matrix`: center with `X - X.mean(axis=0)`, then `Xc.T @ Xc / (n - 1)`. Reuse it
for `correlation_matrix` and divide by `np.outer(s, s)`.

</details>

<details>
<summary>Hint 3</summary>

`columnwise_corr`: `zx = (x - x.mean(0)) / x.std(0)` and the same for `y`, then
`(zx * zy).mean(axis=0)`.

</details>

## Self-check

1. What are `np.var([1, 3])` and `np.var([1, 3], ddof=1)`?
2. What shape does `np.cov(X)` have for `X` of shape `(100, 3)`? Why?
3. Does `ddof` change a correlation coefficient?
4. Why does `torch.std` give a different number than `np.std` on the same data?

<details>
<summary>Answers</summary>

1. `1.0` and `2.0`.
2. `(100, 100)`: `np.cov` treats rows as variables. Use `rowvar=False` to get `(3, 3)`.
3. No: the same divisor appears in the covariance and in both standard deviations, and cancels.
4. PyTorch defaults to the sample std (`n - 1`), NumPy to the population std (`n`).

</details>

## Go deeper

- [numpy.cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html)
