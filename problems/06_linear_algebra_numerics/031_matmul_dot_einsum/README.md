# 031 · matmul, dot, @ and einsum ★

**Category:** Linear algebra and numerics · **Difficulty:** Medium

## What it is

- `a @ b` (= `np.matmul`) - matrix multiplication, with **batching** over leading axes.
- `np.dot` - the older function; same as `@` for 1-D and 2-D, but different for 3-D and up.
- `a * b` - element-wise product, not matrix multiplication.
- `np.einsum("ij,jk->ik", a, b)` - write any sum of products with index letters.

## Why it exists

Linear layers, projections, attention scores, covariance matrices and distances are all sums of
products. `@` covers the common cases with fast BLAS code. `einsum` covers everything else
(traces, bilinear forms, batched dot products) in one readable line, without transposes and
reshapes.

## How it works

**Shapes for `@`:** the last two axes are multiplied as matrices; any axes before them are
batch axes and broadcast (012).

```
(n, k) @ (k, m)          -> (n, m)
(B, n, k) @ (B, k, m)    -> (B, n, m)
(B, n, k) @ (k, m)       -> (B, n, m)      the (k, m) matrix is shared by every batch item
(n, k) @ (k,)            -> (n,)           a 1-D right operand is treated as a column vector
```

**Batched matrix-vector product.** With `A` of shape `(B, n, m)` and one vector per item
`x` of shape `(B, m)`, `A @ x` fails: the `(B, m)` array is read as **one** matrix. Turn the
vectors into columns and back:

```python
(A @ x[..., None])[..., 0]           # (B, n, m) @ (B, m, 1) -> (B, n, 1) -> (B, n)
np.einsum("bnm,bm->bn", A, x)        # same thing
```

**np.dot on 3-D arrays is not batched:**

```python
A = np.ones((2, 3, 4)); B = np.ones((2, 4, 5))
(A @ B).shape        # (2, 3, 5)      pairs batch 0 with batch 0, 1 with 1
np.dot(A, B).shape   # (2, 3, 2, 5)   every A batch times every B batch
```

Use `@` for anything with a batch axis.

**einsum** in three rules: each input gets one letter per axis; a letter that appears in the
inputs but **not** after `->` is summed over; the letters after `->` give the output's axes.

| expression | meaning |
|---|---|
| `"ij,jk->ik"` | matrix product |
| `"bij,bjk->bik"` | batched matrix product |
| `"ii->"` | trace (sum of the diagonal) |
| `"bii->b"` | trace of every matrix in a batch |
| `"bi,ij,bj->b"` | `x[b] @ W @ y[b]` for every `b` (a bilinear form) |
| `"bi,bi->b"` | row-wise dot products |

**Attention scores** `Q K^T / sqrt(d)`: `Q` is `(B, T, d)`, `K` is `(B, S, d)`:

```
Q @ K.swapaxes(-1, -2)     (B, T, d) @ (B, d, S) -> (B, T, S)
```

`K.T` would reverse **all** three axes, giving `(d, S, B)`. Use `swapaxes(-1, -2)` (or
`np.matrix_transpose` in NumPy 2) to transpose only the last two.

## Gotchas

- `*` is element-wise. `A * B` on two `(n, n)` matrices runs fine and is wrong.
- `.T` on a 3-D array reverses all axes; it doesn't transpose each matrix.
- `np.dot` and `@` agree only up to 2-D.
- Integer matrices multiply in integer arithmetic (and can overflow); there's no BLAS speed-up
  for ints.
- PyTorch: `@`, `torch.matmul` and `torch.einsum` follow the same rules. `torch.dot` only accepts
  1-D tensors, unlike `np.dot`.

## Builds on

012, 014, 015.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `batched_matvec(A, x)` - `A` `(B, n, m)`, `x` `(B, m)` -> `(B, n)` with `out[b] = A[b] @ x[b]`.
2. `bilinear(x, W, y)` - `x` `(B, n)`, `W` `(n, m)`, `y` `(B, m)` -> `(B,)` with
   `out[b] = x[b] @ W @ y[b]`.
3. `batch_trace(A)` - `(B, n, n)` -> `(B,)`, the trace of each matrix (`np.trace` is banned).
4. `attention_scores(Q, K)` - `(B, T, d)` and `(B, S, d)` -> `(B, T, S)`, `Q K^T / sqrt(d)`.

## Hints

<details>
<summary>Hint 1</summary>

`batched_matvec`: add a trailing axis to `x`, multiply, drop it again. Or one `einsum`.

</details>

<details>
<summary>Hint 2</summary>

`bilinear` and `batch_trace` are each one `einsum`; read their strings off the table.

</details>

<details>
<summary>Hint 3</summary>

`attention_scores`: `Q @ K.swapaxes(-1, -2) / np.sqrt(Q.shape[-1])`.

</details>

## Self-check

1. What shape is `(B, n, k) @ (k, m)`?
2. Why does `A @ x` fail for `A` `(B, n, m)` and `x` `(B, m)` when `B != m`?
3. What does `np.einsum("ij->ji", a)` do? And `"ij->"`?
4. What does `.T` do to an array of shape `(B, T, d)`?

<details>
<summary>Answers</summary>

1. `(B, n, m)`: the matrix broadcasts over the batch.
2. `x` is read as a single `(B, m)` matrix, so `@` needs `m == B`. It's not "one vector per item".
3. A transpose; the sum of all elements.
4. It reverses all axes: `(d, T, B)`.

</details>

## Go deeper

- [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
