# 011 · take_along_axis and put_along_axis

**Category:** Indexing · **Difficulty:** Medium

## What it is

`np.take_along_axis(x, idx, axis)` reads, for every position along the other axes, the
elements of `x` named by `idx` along one axis. `np.put_along_axis(x, idx, values, axis)` writes
the same way. They are the partners of index-returning functions like `argmax`, `argsort` and
`argpartition`.

## Why it exists

Functions like `np.argsort(x, axis=1)` give you **indices** per row. Turning those indices back
into values needs the "row index + column index" trick from 010 (`x[np.arange(n)[:, None], idx]`),
which gets ugly for 3-D arrays or another axis. `take_along_axis` does it for any shape and any
axis, in one readable call.

## How it works

The rule: `idx` has the **same number of dimensions** as `x`. Along `axis` it can have any
length; along the other axes it matches `x` (or is 1 and broadcasts).

```python
x = np.array([[10, 30, 20],
              [60, 40, 50]])
order = np.argsort(x, axis=1)       # ascending order per row
```

```
order:          np.take_along_axis(x, order, axis=1):
0 2 1           10 20 30
1 2 0           40 50 60
```

For 2-D and `axis=1`, `take_along_axis(x, idx, 1)[i, j]` is `x[i, idx[i, j]]`.

**Reductions drop the axis**, so add it back before using the indices:

```python
best = np.argmax(x, axis=1)                  # shape (2,)    -> [1, 0]
best = np.expand_dims(best, axis=1)          # shape (2, 1)  (or argmax(..., keepdims=True))
np.take_along_axis(x, best, axis=1)          # [[30], [60]]  shape (2, 1)
```

```
x:            idx (2, 1):   result (2, 1):
10 [30] 20      1              30
[60] 40 50      0              60
```

**Writing** works the same way, in place:

```python
mask = np.zeros(x.shape, dtype=bool)
np.put_along_axis(mask, best, True, axis=1)
```

```
False  True False
 True False False
```

**Top-k per row**: sort the indices, keep the last `k` columns, reverse them for descending
order, then `take_along_axis`. (`np.argsort` is covered properly in 021; here you only need
"indices that would sort each row".)

## Gotchas

- `idx` must have the same `ndim` as `x`. Passing the `(n,)` output of `argmax` straight in
  raises an error; expand it to `(n, 1)` first.
- `put_along_axis` modifies its first argument **in place** and returns `None`.
- `np.argsort(-x)` for "descending" breaks on unsigned ints (`-x` wraps around, 002). Reverse an
  ascending argsort instead.
- PyTorch's equivalents are `torch.gather(t, dim, idx)` and `Tensor.scatter_`.
  Same idea, different argument order.

## Builds on

007, 010, and the one-line idea "argsort gives the indices that sort each row" from 021.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `topk_last(x, k)` - the `k` largest values along the last axis, in descending order, and
   their indices. Works for any number of dimensions.
2. `pick_along(x, idx, axis)` - `idx` has `x`'s shape without `axis`; return the elements of `x`
   chosen by `idx` along `axis` (same shape as `idx`).
3. `keep_topk(x, k)` - a copy of a 2-D array where everything except the `k` largest entries of
   each row is set to 0.
4. `argmax_mask(x, axis)` - a boolean array of `x`'s shape, True at the (first) maximum along `axis`.

## Hints

<details>
<summary>Hint 1</summary>

`topk_last`: `np.argsort(x, axis=-1)[..., ::-1][..., :k]` gives the indices; `take_along_axis`
gives the values.

</details>

<details>
<summary>Hint 2</summary>

`pick_along`: `np.expand_dims(idx, axis)`, then `take_along_axis`, then `np.squeeze(..., axis)`.

</details>

<details>
<summary>Hint 3</summary>

`keep_topk` and `argmax_mask`: start from `np.zeros_like(...)` and `put_along_axis` into it.

</details>

## Self-check

1. Why must `idx` have the same number of dimensions as `x`?
2. What does `np.put_along_axis` return?
3. What is `np.take_along_axis(x, np.argsort(x, axis=1), axis=1)` the same as?
4. Why is `np.argsort(-x)` risky for descending order?

<details>
<summary>Answers</summary>

1. The other axes are matched position by position (with broadcasting); NumPy won't guess where a
   missing axis belongs.
2. `None`; it writes into the array you pass.
3. `np.sort(x, axis=1)`.
4. For unsigned ints, `-x` wraps around (and for ties it may reorder differently).

</details>

## Go deeper

- [numpy.take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
