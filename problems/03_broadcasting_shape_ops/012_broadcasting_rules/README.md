# 012 · Broadcasting rules ★

**Category:** Broadcasting and shape ops · **Difficulty:** Medium

## What it is

**Broadcasting** is how NumPy combines arrays of different shapes in one element-wise operation.
Axes of length 1 (and missing leading axes) are stretched to match the other array, without
copying any data. It's the same idea as in PyTorch; this problem makes you apply the rule
precisely, by hand.

## Why it exists

Without broadcasting, `x - x.mean(axis=0)` would need you to tile the mean into a full matrix
first. Broadcasting does the "stretch" virtually (stride 0, see 005), so code stays short and no
memory is wasted. Outer products, pairwise distances and per-channel scaling all come from it.

## How it works

The rule, comparing shapes **from the right**:

1. If one shape is shorter, pad it with 1s on the **left**.
2. Two lengths are compatible if they are **equal** or one of them is **1**.
3. The result length is the larger one. Anything else is an error.

```
a:       (8, 1, 6, 1)
b:          (7, 1, 5)     -> pad -> (1, 7, 1, 5)
result:  (8, 7, 6, 5)

a:       (3, 4)
b:          (3,)          -> (1, 3): 4 vs 3, neither is 1 -> ValueError
```

**Outer sum.** A column `(3, 1)` plus a row `(1, 4)` gives a `(3, 4)` grid:

```python
a = np.array([0, 10, 20])
b = np.array([1, 2, 3, 4])
a[:, None] + b[None, :]
```

```
          b:  1   2   3   4
a:  0  ->     1   2   3   4
   10  ->    11  12  13  14
   20  ->    21  22  23  24
```

**Pairwise differences.** For points `x` of shape `(n, d)`:

```
x[:, None, :]   (n, 1, d)
x[None, :, :]   (1, n, d)
difference      (n, n, d)     diff[i, j] = x[i] - x[j]
```

Square, sum over the last axis, and you have all pairwise squared distances.

`None` (= `np.newaxis`) inserts a length-1 axis; 015 covers it in detail.

## Gotchas

- Broadcasting can **silently** succeed with the wrong meaning: `(n,)` against `(n, n)` lines up
  with the **columns**, not the rows (013).
- `(n,)` and `(n, 1)` are different. `v - v[:, None]` is an `(n, n)` matrix, not zeros.
- The result of a broadcast operation is a real, full-size array. `(10000, 1, 512) - (1, 10000, 512)`
  allocates 10000 × 10000 × 512 floats (about 400 GB). For distances on big inputs, use
  `|x|² + |y|² - 2 x·y` instead (040).
- PyTorch uses exactly the same rules, so shapes that broadcast in one broadcast in the other.

## Builds on

001, 005, 006. Uses `None` indexing, explained fully in 015.

## Your task

Implement in `problem.py`:

1. `broadcast_shape(shape1, shape2)` - the broadcast result shape, by applying the rule yourself.
   Raise `ValueError` if incompatible. (`np.broadcast_shapes` and `np.broadcast` are banned.)
2. `outer_sum(a, b)` - `out[i, j] = a[i] + b[j]` for 1-D inputs.
3. `pairwise_diff(x)` - `out[i, j] = x[i] - x[j]` for `x` of shape `(n, d)`; result `(n, n, d)`.
4. `pairwise_sq_dist(x, y)` - squared Euclidean distances, `(n, d)` and `(m, d)` -> `(n, m)`.

Tasks 2-4: no Python loops; `np.add.outer`, `np.subtract.outer` and `np.outer` are banned.

## Hints

<details>
<summary>Hint 1</summary>

For `broadcast_shape`, pad the shorter tuple with 1s on the left, then walk both tuples together
with `zip`.

</details>

<details>
<summary>Hint 2</summary>

`outer_sum` is `a[:, None] + b[None, :]` (or just `a[:, None] + b`).

</details>

<details>
<summary>Hint 3</summary>

`pairwise_sq_dist`: `diff = x[:, None, :] - y[None, :, :]` has shape `(n, m, d)`; square it and
sum over the last axis.

</details>

## Self-check

1. What is the broadcast shape of `(5, 1, 4)` and `(3, 1)`?
2. Do `(3, 4)` and `(4, 3)` broadcast?
3. Why does `x[:, None, :] - x[None, :, :]` give shape `(n, n, d)`?
4. Why can broadcasting run out of memory even though it "doesn't copy"?

<details>
<summary>Answers</summary>

1. `(5, 3, 4)`.
2. No: from the right, 4 vs 3 doesn't match and neither is 1.
3. `(n, 1, d)` and `(1, n, d)`: each length-1 axis stretches to the other's length.
4. The *inputs* aren't copied, but the *output* is a new array with the full broadcast shape.

</details>

## Go deeper

- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
