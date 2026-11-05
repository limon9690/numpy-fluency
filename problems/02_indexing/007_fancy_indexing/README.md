# 007 · Fancy indexing ★

**Category:** Indexing · **Difficulty:** Medium

## What it is

**Fancy indexing** (also called integer-array or advanced indexing) means indexing with an array
or list of integers: `a[[3, 0, 3]]`. You get the elements at those positions, in that order,
repeats allowed. With one index array per axis, NumPy pairs them up element by element.
The result is always a **copy**.

## Why it exists

It's NumPy's "gather": pick rows by ID, reorder data by a permutation, look up values in a
table, pick the score of the correct class for every sample. Without it you'd write a Python
loop over indices.

## How it works

One index array on axis 0 picks rows:

```python
a = np.array([[10, 11, 12],
              [20, 21, 22],
              [30, 31, 32]])
a[[2, 0, 2]]
```

```
30 31 32
10 11 12
30 31 32
```

The **result has the shape of the index array**. A 2-D index into a 1-D table gives a 2-D result:

```python
table = np.array([0, 10, 20, 30])
table[np.array([[1, 1], [3, 0]])]
```

```
10 10
30  0
```

That's exactly a **lookup table**: `lut[img]` replaces every pixel value `v` by `lut[v]`.

**One index array per axis are paired**, not crossed:

```python
rows = np.array([0, 1, 2])
cols = np.array([2, 0, 1])
a[rows, cols]          # [a[0,2], a[1,0], a[2,1]] = [12, 20, 31]
```

```
 10 11 [12]
[20] 21 22
 30 [31] 32
```

This is how you pick one entry per row, e.g. `probs[np.arange(n), labels]`.

**Fancy assignment** writes into those positions:

```python
z = np.zeros((3, 4))
z[np.arange(3), [1, 3, 0]] = 1     # one 1 per row: a one-hot matrix
```

```
0 1 0 0
0 0 0 1
1 0 0 0
```

## Gotchas

- `a[rows, cols]` pairs indices. For the full block of rows × cols you need `np.ix_` (010).
- Fancy indexing returns a **copy**; `a[[0, 1]][0] = 5` does not change `a` (004).
- With repeated indices in an assignment, `a[[0, 0]] += 1` adds **once**, not twice
  (the last write wins). Use `np.add.at` for that (025).
- A Python **tuple** is not an index array: `a[(0, 1)]` means `a[0, 1]`. Use a list or array.
- Index arrays must be integers (or booleans, see 008). Float arrays raise `IndexError`.
- PyTorch behaves the same way here (`t[idx]`, `t[rows, cols]`), and also copies.

## Builds on

006.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `one_hot(labels, num_classes)` - `(n,)` int labels -> `(n, num_classes)` float32 matrix with a
   single 1 per row.
2. `pick_per_row(a, cols)` - `out[i] = a[i, cols[i]]`, shape `(n,)`.
3. `apply_lut(img, lut)` - replace every value `v` of a uint8 array (any shape) by `lut[v]`;
   `lut` is a `(256,)` uint8 table.
4. `inverse_permutation(p)` - the array `inv` with `inv[p[i]] = i` (`np.argsort` is banned).

## Hints

<details>
<summary>Hint 1</summary>

`one_hot` and `inverse_permutation` are both "make an empty array, then fancy-assign into it".
For `one_hot`, the row indices are `np.arange(n)`.

</details>

<details>
<summary>Hint 2</summary>

`apply_lut` is one indexing expression; the result automatically gets the shape of `img`.

</details>

<details>
<summary>Hint 3</summary>

`inverse_permutation`: `inv = np.empty_like(p)`, then `inv[p] = np.arange(len(p))`.

</details>

## Self-check

1. What shape is `a[idx]` if `a` has shape `(10, 3)` and `idx` has shape `(4, 5)`?
2. What is the difference between `a[[0, 1], [0, 1]]` and `a[0:2, 0:2]`?
3. After `a = np.zeros(3); a[[1, 1, 1]] += 1`, what is `a[1]`?
4. Is `a[[0, 1, 2]]` a view or a copy?

<details>
<summary>Answers</summary>

1. `(4, 5, 3)`: the index shape replaces axis 0.
2. The first picks 2 elements (the diagonal), shape `(2,)`. The second is a `(2, 2)` block view.
3. `1.0`: each `+= 1` reads the same old value, so the writes don't add up.
4. A copy, even though it selects every row.

</details>

## Go deeper

- [Integer array indexing](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing)
