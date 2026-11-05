# 032 · solve vs inv, condition number

**Category:** Linear algebra and numerics · **Difficulty:** Medium

## What it is

To solve `A x = b` you could compute `inv(A) @ b`. `np.linalg.solve(A, b)` gets `x` directly
instead, without forming the inverse. The **condition number** `np.linalg.cond(A)` tells you
how sensitive the answer is to tiny errors: roughly how many digits you can lose.

## Why it exists

Linear systems are everywhere: fitting, interpolation, physics steps, Newton's method. `solve`
uses an LU factorization: it's about as fast as one matrix product, more accurate than going
through `inv`, and it handles many right-hand sides at once. The condition number warns you
when no method can give an accurate answer, because the problem itself is nearly singular.

## How it works

```python
A = np.array([[3.0, 1.0],
              [1.0, 2.0]])
b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)      # [2., 3.]
np.allclose(A @ x, b)          # True
```

**Many right-hand sides** at once: make `B` a matrix whose **columns** are the different `b`s.
`np.linalg.solve(A, B)` with `B` of shape `(n, k)` returns `X` of shape `(n, k)`.

**Why not inv?** On the 10×10 Hilbert matrix `H[i, j] = 1 / (i + j + 1)` (very
ill-conditioned, `cond ≈ 1.6e13`) with true answer all ones:

| method | relative residual `‖Ax - b‖ / ‖b‖` | max error in `x` |
|---|---|---|
| `solve(H, b)` | about `1e-16` | about `2e-5` |
| `inv(H) @ b` | about `3e-5` | about `5e-3` |

`solve` gives an `x` that fits the equations to machine precision. `inv` is worse on both
counts, and it costs more.

**Condition number.** With float64 you have about 16 significant digits; a system with
`cond(A) ≈ 10^k` can lose about `k` of them in `x`.

```
cond = 1e2   -> ~14 good digits in x
cond = 1e13  -> ~3 good digits
cond ≈ 1e16+ -> numerically singular: the answer can be garbage
```

An exactly singular matrix makes `solve` raise `np.linalg.LinAlgError` (a subclass of
`ValueError`). A *nearly* singular one gives no error at all, just a bad answer. Checking `cond`
first catches that.

**Relative residual** `‖A x − b‖ / ‖b‖` checks a solution after the fact. Small residual with a
huge `cond` means "x fits the equations, but x itself may still be far from the true answer".

**Batched systems and the NumPy 2 shape rule.** `solve` broadcasts over leading axes:
`A` `(N, n, n)` with `B` `(N, n, k)`. But in NumPy 2, `b` is treated as a vector **only if it is
1-D**. A stack of vectors `b` of shape `(N, n)` is read as a **matrix**:

```python
A = rng.random((4, 3, 3)); b = rng.random((4, 3))
np.linalg.solve(A, b)                  # error: (4, 3) is read as a (4, 3) matrix, 4 != 3
np.linalg.solve(A, b[..., None])[..., 0]   # correct: (4, 3, 1) columns -> (4, 3)
```

When `N == n` the wrong call **runs** and returns a meaningless `(n, n, n)` array.

## Gotchas

- Don't write `inv(A) @ b`. Use `solve`; it's faster and more accurate.
- `solve` only works for square, non-singular `A`. Non-square systems need least squares (033).
- No error doesn't mean a good answer: check `cond` or the residual for ill-conditioned problems.
- Batched vectors need the trailing `[..., None]` in NumPy 2 (NumPy 1.x guessed from the shapes).
- PyTorch: `torch.linalg.solve` follows the same batching idea; `torch.linalg.cond` exists too.

## Builds on

015, 031.

## Your task

Implement in `problem.py`, with **no Python loops** (`np.linalg.inv` is banned everywhere):

1. `solve_many(A, B)` - solve `A X = B` for an `(n, n)` matrix and `(n, k)` right-hand sides.
2. `batched_solve(A, b)` - `A` `(N, n, n)`, `b` `(N, n)` -> `x` `(N, n)` with `A[i] @ x[i] = b[i]`.
3. `solve_checked(A, b, max_cond)` - solve, but raise `ValueError` if `cond(A) > max_cond`
   (including singular matrices).
4. `relative_residual(A, x, b)` - `‖A x − b‖ / ‖b‖` (2-norm), as a Python float.

## Hints

<details>
<summary>Hint 1</summary>

`solve_many` is a single `np.linalg.solve` call. For `batched_solve`, add a trailing axis to `b`
and remove it from the result.

</details>

<details>
<summary>Hint 2</summary>

`np.linalg.cond(A)` returns a huge number (or inf) for singular matrices, so check it before
calling `solve`. `np.linalg.norm` gives the 2-norm of a vector.

</details>

## Self-check

1. Why prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b`?
2. What does a condition number of `1e10` mean for float64 results?
3. What goes wrong with `np.linalg.solve(A, b)` for `A` `(5, 5, 5)` and `b` `(5, 5)` in NumPy 2?
4. Can a solution have a tiny residual and still be inaccurate?

<details>
<summary>Answers</summary>

1. It skips forming the inverse: less work and smaller rounding errors.
2. You can lose about 10 of the ~16 digits; expect about 6 correct digits in `x`.
3. `b` is read as one `(5, 5)` matrix shared by all 5 systems, so you get shape `(5, 5, 5)`
   instead of one solution vector per system.
4. Yes, if `A` is ill-conditioned: many very different `x` fit the equations almost equally well.

</details>

## Go deeper

- [numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
