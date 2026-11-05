# 025 · np.add.at and scatter-style updates

**Category:** Reductions and vectorized ops · **Difficulty:** Hard

## What it is

`np.add.at(a, idx, values)` adds `values` into `a` at the positions `idx`, **correctly
handling repeated indices**: if position 2 appears three times, it receives all three additions.
Every binary ufunc has `.at` (`np.maximum.at`, `np.multiply.at`, ...). This is a "scatter"
operation, the opposite of the "gather" `a[idx]`.

## Why it exists

The obvious code `a[idx] += values` silently drops updates when `idx` has repeats. Scatter with
repeats is common: counting points per pixel, summing gradients for an embedding table where
the same word appears many times, accumulating votes, building a 2-D histogram.

## How it works

**The trap:**

```python
a = np.zeros(4)
idx = np.array([0, 2, 2, 2])
a[idx] += 1
a                        # [1., 0., 1., 0.]   <- index 2 got +1 once, not 3 times!
```

`a[idx] += 1` runs as three separate steps: `tmp = a[idx]` (gather a copy), `tmp += 1`,
`a[idx] = tmp` (scatter). The three writes to `a[2]` all write the same value, so the last one
wins and the others are lost.

```
idx:      0   2   2   2
tmp:      0   0   0   0      read a[idx]
tmp+1:    1   1   1   1
write:  a[0]=1, a[2]=1, a[2]=1, a[2]=1   -> a[2] == 1
```

**The fix:**

```python
a = np.zeros(4)
np.add.at(a, idx, 1)
a                        # [1., 0., 3., 0.]
```

`.at` is **unbuffered**: each update reads the current value, so repeats add up. It works in
place and returns `None`.

**Multi-dimensional targets.** `idx` can be a tuple of index arrays (one per axis), just like
fancy indexing (007):

```python
counts = np.zeros((2, 3), dtype=np.int64)
np.add.at(counts, (np.array([0, 0, 1]), np.array([2, 2, 0])), 1)
```

```
0 0 2
1 0 0
```

Or index only axis 0 and add whole rows: `np.add.at(table, ids, rows)` with `rows` of shape
`(len(ids), D)` adds each row into `table[ids[k]]`. That's exactly the gradient of an embedding
lookup `table[ids]`.

**Other ufuncs:** `np.maximum.at(best, idx, values)` keeps the largest value per slot. Start
`best` at `-inf` so empty slots are visible.

**Alternatives:** for 1-D sums, `np.bincount(idx, weights=values, minlength=n)` (022) does the
same and is often faster. For 2-D, flatten the coordinates with `np.ravel_multi_index`.

## Gotchas

- `a[idx] += v` and `a[idx] = a[idx] + v` both lose repeated updates. No error, no warning.
- `np.add.at` modifies `a` in place and returns `None`. Copy first if you need the original.
- `.at` used to be very slow; it got much faster in NumPy 1.25+, but `bincount` is still the
  quickest for simple 1-D sums.
- PyTorch: `index_add_`, `scatter_add_` and `index_put_(..., accumulate=True)` do the same job.
  Plain `t[idx] += v` has the same lost-update problem as NumPy.

## Builds on

007, 019, 022.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `scatter_add(target, idx, values)` - a copy of the 1-D `target` with each `values[k]` added at
   `idx[k]`; repeated indices add up.
2. `pixel_counts(ys, xs, h, w)` - an `(h, w)` int64 array counting how many points fall on each pixel.
3. `scatter_max(n, idx, values)` - an `(n,)` float64 array with the largest value sent to each
   slot, `-inf` for slots that got nothing.
4. `embedding_grad(indices, grad_out, vocab_size)` - `indices` `(B, T)`, `grad_out` `(B, T, D)`
   -> `(vocab_size, D)` with `grad_out[b, t]` summed into row `indices[b, t]`.

## Hints

<details>
<summary>Hint 1</summary>

`scatter_add`: `out = target.copy()`, then `np.add.at(out, idx, values)`.

</details>

<details>
<summary>Hint 2</summary>

`pixel_counts`: `np.add.at(counts, (ys, xs), 1)`. `scatter_max`: start from
`np.full(n, -np.inf)` and use `np.maximum.at`.

</details>

<details>
<summary>Hint 3</summary>

`embedding_grad`: flatten `indices` to `(B*T,)` and `grad_out` to `(B*T, D)`, then one `np.add.at`
on a zero `(vocab_size, D)` array.

</details>

## Self-check

1. After `a = np.zeros(3); a[[1, 1]] += 5`, what is `a[1]`? And with `np.add.at(a, [1, 1], 5)` on a fresh array?
2. Why does buffering cause lost updates?
3. What does `np.add.at` return?
4. How would you do a 1-D scatter-add with `bincount` instead?

<details>
<summary>Answers</summary>

1. `5.0`; with `np.add.at`, `10.0`.
2. All values are read first, updated in a temporary, then written back; repeated positions
   are overwritten with the same updated value instead of being added twice.
3. `None`; it works in place.
4. `target + np.bincount(idx, weights=values, minlength=len(target))`.

</details>

## Go deeper

- [numpy.ufunc.at](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
