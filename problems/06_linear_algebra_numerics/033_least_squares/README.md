# 033 · Least squares with lstsq

**Category:** Linear algebra and numerics · **Difficulty:** Medium

## What it is

`np.linalg.lstsq(A, b)` finds the `x` that makes `A @ x` as close as possible to `b`: it
minimizes `‖A x − b‖²`. It works for any shape of `A`: more equations than unknowns (the usual
"fit" case), fewer (many exact solutions: it returns the smallest one), or rank-deficient.

## Why it exists

`solve` (032) needs a square, invertible matrix. Real data gives you **more equations than
unknowns**, for example 100 noisy points and a line with 2 parameters. No `x` fits every
equation, so you ask for the best compromise. Calibrating a sensor, fitting a trend, finding the
plane through a point cloud, or removing a linear drift are all least squares.

## How it works

**Fit a line** `y ≈ m·x + c` to points. Each point gives one equation `m·x_i + c·1 = y_i`.
Stack them into a **design matrix** with a column of ones for the intercept:

```
      A (n, 2)         unknowns     b (n,)
  [ x_0   1 ]          [ m ]        [ y_0 ]
  [ x_1   1 ]    @     [ c ]   ≈    [ y_1 ]
  [ ...  ...]                       [ ... ]
```

```python
A = np.column_stack([x, np.ones_like(x)])
(m, c), res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
```

`lstsq` returns four things: the solution, the residual sum of squares, the rank of `A`, and its
singular values. `rcond=None` (the NumPy 2 default) sets the cutoff for tiny singular values.

**The residuals are often empty.** The `res` output is an empty array when `A` is rank-deficient
or has no more rows than columns. Computing `((A @ x - b) ** 2).sum(axis=0)` yourself always works.

**Several targets at once.** If `B` has shape `(n, k)`, `lstsq(A, B)` fits all `k` columns
in one call and returns `X` of shape `(d, k)`.

**Underdetermined systems.** With fewer equations than unknowns, infinitely many `x` fit
exactly. `lstsq` returns the one with the **smallest norm** `‖x‖`, the same as
`np.linalg.pinv(A) @ b`:

```
x + y = 2       solutions: (2, 0), (0, 2), (1, 1), ...
lstsq picks (1, 1): the shortest vector on the line x + y = 2
```

**Why not the normal equations?** Textbooks write `x = inv(AᵀA) Aᵀb`. Forming `AᵀA` **squares**
the condition number (032), so you lose twice as many digits. `lstsq` uses an SVD and avoids that.

## Gotchas

- Forgetting the column of ones forces the fitted line through the origin.
- The second output (residuals) is empty in the rank-deficient and square/underdetermined cases.
- `rcond`: in NumPy 2 the default (`None`) is the sensible machine-precision cutoff. Old code
  passes `rcond=None` to silence a NumPy 1.x warning, or `rcond=-1` to keep the old behaviour.
- Polynomial fitting functions (`np.polyfit`) exist, but they're outside this sheet; build the
  design matrix yourself.
- PyTorch: `torch.linalg.lstsq(A, B)` returns a named tuple with `.solution`; its default
  driver on CPU (`gelsy`) handles rank-deficient `A`.

## Builds on

016, 031, 032.

## Your task

Implement in `problem.py`, with **no Python loops** (`np.polyfit` and `np.linalg.inv` are banned):

1. `fit_line(x, y)` - least-squares slope and intercept, as two Python floats.
2. `fit_plane(points)` - for an `(n, 3)` array of `(x, y, z)` points, the `(a, b, c)` of the
   best plane `z ≈ a·x + b·y + c`, as a float64 array of shape `(3,)`.
3. `lstsq_multi(A, B)` - fit every column of `B` `(n, k)`; return `X` `(d, k)` and the residual
   sum of squares per column, shape `(k,)` (always filled, even when `lstsq`'s own is empty).
4. `min_norm_solution(A, b)` - for a wide `A` (fewer rows than columns), the exact solution with
   the smallest norm.

## Hints

<details>
<summary>Hint 1</summary>

`fit_line`: `A = np.column_stack([x, np.ones_like(x)])`, then
`np.linalg.lstsq(A, y, rcond=None)[0]`.

</details>

<details>
<summary>Hint 2</summary>

`fit_plane`: the design matrix has columns `x`, `y`, `1`; the target is `z`.

</details>

<details>
<summary>Hint 3</summary>

`lstsq_multi`: `rss = ((A @ X - B) ** 2).sum(axis=0)`.

</details>

## Self-check

1. Why does the design matrix for a line need a column of ones?
2. When is the residuals output of `lstsq` empty?
3. Which solution does `lstsq` return when there are infinitely many?
4. Why avoid solving `AᵀA x = Aᵀb` directly?

<details>
<summary>Answers</summary>

1. It multiplies the intercept `c`; without it the model is `y = m·x`, a line through the origin.
2. When `A` is rank-deficient, or has no more rows than columns.
3. The one with the smallest Euclidean norm (the pseudo-inverse solution).
4. `AᵀA` has the square of `A`'s condition number, so twice as many digits can be lost.

</details>

## Go deeper

- [numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
