# 034 · Norms, SVD and eigh (batched)

**Category:** Linear algebra and numerics · **Difficulty:** Hard

## What it is

- `np.linalg.norm` - the size of a vector or matrix (Euclidean, L1, max, Frobenius, ...).
- `np.linalg.svd` - the singular value decomposition `A = U · diag(s) · Vh`, for any matrix.
- `np.linalg.eigh` - eigenvalues and eigenvectors of a **symmetric** matrix.

All three work on **stacks** of matrices: the last two axes are the matrix, the rest is a batch.

## Why it exists

Norms measure distances and errors. The SVD is the Swiss army knife of numerics: rank,
compression, pseudo-inverse, condition number, PCA. `eigh` handles symmetric matrices
(covariances, Gram matrices, graph Laplacians) faster and more reliably than the general `eig`.

## How it works

**Vector norms along an axis:**

```python
X = np.array([[3.0, 4.0],
              [1.0, -1.0]])
np.linalg.norm(X, axis=1)              # [5.    1.414]   L2 per row
np.linalg.norm(X, ord=1, axis=1)       # [7. 2.]        sum of |x|
np.linalg.norm(X, ord=np.inf, axis=1)  # [4. 1.]        max |x|
np.linalg.norm(X)                      # 5.196          no axis: Frobenius norm of the matrix
```

**SVD.** For `A` of shape `(m, n)` with `full_matrices=False`:

```
A (m, n)  =  U (m, r)  @  diag(s) (r, r)  @  Vh (r, n)        r = min(m, n)

s is sorted from largest to smallest; columns of U and rows of Vh are orthonormal.
```

Keeping only the first `k` singular values gives the **best rank-k approximation** of `A`
(Eckart-Young theorem):

```python
U, s, Vh = np.linalg.svd(A, full_matrices=False)
A_k = (U[:, :k] * s[:k]) @ Vh[:k, :]      # U * s scales column j of U by s[j]
```

On a batch `(B, m, n)`, use `...` and keep the batch axis: `U[..., :, :k] * s[..., None, :k]`.

**eigh** for symmetric `S` of shape `(n, n)`:

```python
w, V = np.linalg.eigh(S)    # w: eigenvalues ASCENDING, V: eigenvectors as COLUMNS
S @ V[:, j]  ≈  w[j] * V[:, j]
```

```
w = [0.5, 1.0, 4.0]            V = [ v0 | v1 | v2 ]    <- the largest is the LAST column
```

**Matrix functions** of a symmetric matrix work on the eigenvalues: `S = V diag(w) Vᵀ`, so
`sqrt(S) = V diag(sqrt(w)) Vᵀ`. For a positive semi-definite matrix all `w ≥ 0`, but rounding
can make a zero eigenvalue come out as `-1e-17`, so clip at 0 before `sqrt`.

## Gotchas

- `eigh` returns eigenvalues in **ascending** order, eigenvectors in **columns**. `V[0]` is a row,
  not an eigenvector.
- Eigenvectors and singular vectors are only defined up to **sign** (`-v` works as well as `v`).
  Don't compare them with `assert_allclose` directly.
- `eigh` only reads one triangle of the matrix and trusts that it's symmetric; a non-symmetric
  input gives silently wrong answers. Use `eig` for general matrices (complex results possible).
- `svd` returns `Vh` (V transposed), not `V`.
- `norm(A)` on a matrix is Frobenius; `norm(A, 2)` is the largest singular value.
- PyTorch: `torch.linalg.svd`, `torch.linalg.eigh` (same ascending order) and
  `torch.linalg.vector_norm` / `matrix_norm`.

## Builds on

018, 031, 032.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `row_norms(X, ord)` - the vector norm of each row of a 2-D array for `ord` in `1`, `2`, `np.inf`.
2. `low_rank_approx(A, k)` - the best rank-`k` approximation of `A` `(..., m, n)`, batched.
3. `top_eigenpair(S)` - for symmetric `S` `(B, n, n)`: the largest eigenvalue `(B,)` and its
   unit eigenvector `(B, n)` for every matrix.
4. `psd_sqrt(S)` - the symmetric square root of positive semi-definite `S` `(..., n, n)`:
   `R @ R ≈ S`. Clip tiny negative eigenvalues to 0.

## Hints

<details>
<summary>Hint 1</summary>

`low_rank_approx`: `U[..., :k] * s[..., None, :k]` scales the kept columns, then `@ Vh[..., :k, :]`.

</details>

<details>
<summary>Hint 2</summary>

`top_eigenpair`: `w, V = np.linalg.eigh(S)`; the answer is `w[:, -1]` and `V[:, :, -1]`
(the last column of each matrix).

</details>

<details>
<summary>Hint 3</summary>

`psd_sqrt`: `(V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ V.swapaxes(-1, -2)`.

</details>

## Self-check

1. In which order does `eigh` return eigenvalues, and where are the eigenvectors?
2. Why can't you compare eigenvectors from two methods element by element?
3. What does `np.linalg.norm(A)` compute for a 2-D `A` without `ord` or `axis`?
4. Why clip eigenvalues at 0 before taking a square root?

<details>
<summary>Answers</summary>

1. Ascending; eigenvectors are the columns of `V`.
2. Each one is only defined up to sign (and, for repeated eigenvalues, up to a rotation).
3. The Frobenius norm: the square root of the sum of all squared entries.
4. Rounding can make a true zero eigenvalue slightly negative, and `sqrt` of a negative gives NaN.

</details>

## Go deeper

- [Linear algebra (numpy.linalg)](https://numpy.org/doc/stable/reference/routines.linalg.html)
