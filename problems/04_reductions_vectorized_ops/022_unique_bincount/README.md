# 022 · unique and bincount ★

**Category:** Reductions and vectorized ops · **Difficulty:** Medium

## What it is

- `np.unique(x)` returns the sorted distinct values. With `return_counts=True` it also counts
  them; with `return_inverse=True` it tells you, for every element, **which** distinct value it
  is (an integer code). With `axis=0` it works on whole rows.
- `np.bincount(ints)` counts how often each non-negative integer 0, 1, 2, ... occurs. With
  `weights=` it **sums** weights per integer instead.

## Why it exists

Counting and grouping are everywhere: class frequencies, label encoding (`"cat" -> 0`), duplicate
rows, per-group sums and means. `unique` handles arbitrary values; `bincount` is the fast path
when your keys are already small integers (class IDs, pixel values, bucket numbers).

## How it works

```python
labels = np.array([30, 10, 30, 20, 30])
values, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
```

```
values:   [10, 20, 30]        sorted distinct values
inverse:  [ 2,  0,  2,  1,  2]   labels == values[inverse]
counts:   [ 1,  1,  3]
```

`inverse` is a **label encoding**: every value replaced by its index in `values`.

**Unique rows** (`axis=0`) treat each row as one item:

```python
a = np.array([[1, 2],
              [3, 4],
              [1, 2]])
rows, counts = np.unique(a, axis=0, return_counts=True)
```

```
rows:     counts:
1 2       2
3 4       1
```

**bincount** - position `k` of the result is the count of `k`:

```python
np.bincount([0, 2, 2, 3, 2])             # [1, 0, 3, 1]
np.bincount([0, 2, 2], minlength=5)      # [1, 0, 2, 0, 0]
np.bincount([0, 2, 2], weights=[1.0, 0.5, 4.0])   # [1.0, 0.0, 4.5]
```

**Group mean** in two lines: sums with weights, counts without, then divide:

```
groups: [0, 1, 0, 2]     values: [10, 5, 20, 7]
sums   = bincount(groups, weights=values) = [30, 5, 7]
counts = bincount(groups)                 = [ 2, 1, 1]
mean   = sums / counts                    = [15, 5, 7]
```

**Most frequent** small int: `np.bincount(x).argmax()` (ties go to the smallest value).

## Gotchas

- `bincount` needs **non-negative ints**. Negative values raise; floats raise.
- `bincount` makes an array of length `max(x) + 1`: `np.bincount([10**9])` tries to allocate
  a billion counters. Use `unique` for large or sparse keys.
- `minlength` sets the minimum length, so groups that never appear still get a slot.
- `return_inverse` shape: in NumPy 2 the codes have the **same shape as the input** (NumPy 1.x
  returned them flat). With `axis=0` they are 1-D, one code per row.
- `np.unique` sorts, so it's O(n log n) and the values come out sorted, not in first-seen order.
- PyTorch: `torch.unique` has `return_inverse`/`return_counts`; `torch.bincount` exists too.

## Builds on

007, 018, 021.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `encode_labels(labels)` - `(classes, codes)`: sorted distinct values and an int code for every
   element, with `codes` the **same shape** as `labels` and `classes[codes] == labels`.
2. `unique_rows_counts(a)` - the distinct rows of a 2-D array (sorted) and how often each occurs.
3. `most_frequent(x)` - the most common value of a 1-D array of non-negative ints (smallest on
   ties), as a Python int. `np.unique` is banned here.
4. `group_mean(groups, values, n_groups)` - the mean of `values` for each group id
   `0..n_groups-1`; empty groups give NaN, without warnings. `np.unique` is banned here.

## Hints

<details>
<summary>Hint 1</summary>

`encode_labels`: `np.unique(labels, return_inverse=True)`. Reshape the codes to `labels.shape`
if you want to be safe across NumPy versions.

</details>

<details>
<summary>Hint 2</summary>

`group_mean`: `np.bincount(groups, weights=values, minlength=n_groups)` for sums, and the same
without weights for counts.

</details>

<details>
<summary>Hint 3</summary>

Divide only where the count is positive: `np.divide(sums, counts, out=np.full(n_groups, np.nan), where=counts > 0)` (019).

</details>

## Self-check

1. What does `np.bincount([1, 1, 3])` return?
2. What does `values[inverse]` give back?
3. When is `np.unique` a better choice than `np.bincount`?
4. Why pass `minlength` to `bincount`?

<details>
<summary>Answers</summary>

1. `[0, 2, 0, 1]`.
2. The original array.
3. When the values are negative, floats, huge or sparse integers, or whole rows.
4. So every possible id gets a slot, even ids that don't occur (otherwise the result is only
   `max + 1` long).

</details>

## Go deeper

- [numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html)
