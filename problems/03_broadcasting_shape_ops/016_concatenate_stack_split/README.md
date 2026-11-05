# 016 · concatenate, stack, split

**Category:** Broadcasting and shape ops · **Difficulty:** Easy

## What it is

- `np.concatenate(arrays, axis)` - join arrays along an **existing** axis.
- `np.stack(arrays, axis)` - join arrays along a **new** axis.
- `np.split(a, indices_or_sections, axis)` - cut an array into pieces (the inverse).

`hstack`, `vstack`, `column_stack` and `dstack` are shortcuts for common cases.

## Why it exists

Data arrives in pieces: frames of a video, per-file results, mini-batches, separate R, G and B
planes. You glue them into one array to process them together, then cut the result apart again.

## How it works

Two `(2, 3)` arrays `a` (all 1s) and `b` (all 2s):

```
concatenate axis=0 -> (4, 3)     concatenate axis=1 -> (2, 6)
1 1 1                            1 1 1 2 2 2
1 1 1                            1 1 1 2 2 2
2 2 2
2 2 2

stack axis=0  -> (2, 2, 3)     stack axis=-1 -> (2, 3, 2)
```

The rule: **concatenate** needs all shapes equal except along `axis`; **stack** needs all shapes
exactly equal and adds an axis of length `len(arrays)`.

Joining three `(H, W)` planes into an RGB image:

```python
np.stack([r, g, b], axis=-1)                 # (H, W, 3)
np.concatenate([r, g, b], axis=-1)           # (H, 3W): wrong, no new axis
```

**Interleaving** is a stack followed by a reshape:

```python
a = np.array([1, 2, 3]); b = np.array([10, 20, 30])
np.stack([a, b], axis=1)                     # (3, 2): pairs side by side
np.stack([a, b], axis=1).reshape(-1)         # [1, 10, 2, 20, 3, 30]
```

**split** takes either a number of equal sections or a list of cut points:

```python
x = np.arange(10)
np.split(x, 2)              # [0..4], [5..9]
np.split(x, [3, 6])         # [0 1 2], [3 4 5], [6 7 8 9]
np.split(x, 3)              # ValueError: 10 doesn't divide into 3 equal parts
np.array_split(x, 3)        # sizes 4, 3, 3 (spread evenly)
```

The pieces from `split` are **views** of `x`. Concatenate and stack always **copy**.

## Gotchas

- Stacking with `np.ones` or `np.zeros` silently promotes: `np.ones((n, 1))` is float64, so adding
  a bias column to float32 data gives float64. Pass `dtype=X.dtype`.
- Calling `np.concatenate` inside a loop to grow an array is quadratic: each call copies everything.
  Collect pieces in a list and join once at the end.
- `np.split(x, 3)` demands equal sizes; for "batches of size k with a smaller last one" pass cut
  points: `np.arange(k, n, k)`.
- PyTorch: `torch.cat` = `np.concatenate`, `torch.stack` = `np.stack`, but `torch.split(t, k)`
  takes a **size** per chunk, while `np.split(x, k)` takes a **number** of chunks.

## Builds on

014, 015.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `add_bias_column(X)` - put a column of ones in front of the `(n, d)` matrix `X`; result
   `(n, d + 1)` with **the same dtype** as `X`.
2. `split_batches(x, batch_size)` - a list of consecutive pieces of `x` along axis 0, each of
   length `batch_size` except possibly the last. The pieces are views.
3. `interleave(a, b)` - two arrays of the same shape `(n, ...)` -> `(2n, ...)` with rows
   `a[0], b[0], a[1], b[1], ...`.
4. `join_channels(r, g, b)` - three `(H, W)` planes -> one `(H, W, 3)` image.

## Hints

<details>
<summary>Hint 1</summary>

`add_bias_column`: `np.ones((n, 1), dtype=X.dtype)` and `np.concatenate(..., axis=1)`.

</details>

<details>
<summary>Hint 2</summary>

`split_batches`: `np.split(x, np.arange(batch_size, len(x), batch_size))`.
`interleave`: stack on a new axis 1, then reshape to `(2n, ...)` with `-1` and `a.shape[1:]`.

</details>

## Self-check

1. What is the difference between `np.concatenate` and `np.stack`?
2. What shape does `np.stack([a, b, c])` have for three `(4, 5)` arrays? And with `axis=-1`?
3. Why is growing an array with `np.concatenate` in a loop slow?
4. Are the pieces returned by `np.split` views or copies?

<details>
<summary>Answers</summary>

1. `concatenate` joins along an existing axis; `stack` creates a new one.
2. `(3, 4, 5)`; with `axis=-1`, `(4, 5, 3)`.
3. Each call allocates a new array and copies all the data so far: O(n²) work overall.
4. Views of the original array.

</details>

## Go deeper

- [Joining arrays](https://numpy.org/doc/stable/reference/routines.array-manipulation.html#joining-arrays)
