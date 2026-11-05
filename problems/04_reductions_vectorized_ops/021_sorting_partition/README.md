# 021 · sort, stable argsort, lexsort, partition ★

**Category:** Reductions and vectorized ops · **Difficulty:** Medium

## What it is

- `np.sort(x)` returns sorted values; `np.argsort(x)` returns the **indices** that would sort `x`.
- `kind="stable"` keeps equal elements in their original order.
- `np.lexsort(keys)` sorts by several keys (like sorting a spreadsheet by two columns).
- `np.partition(x, k)` only guarantees that the `k`-th element is in its sorted place, with
  smaller ones before it and larger ones after. It's faster than a full sort.

## Why it exists

You rarely want sorted values alone. You want to **reorder something else** by them: sort table
rows by a column, sort samples by score, reorder labels with their data. That's argsort.
Stability matters when you sort by one thing after another. Partition is for "the 10 smallest
of a million" without paying for a full sort.

## How it works

**argsort and reordering:**

```python
scores = np.array([30, 10, 20])
names  = np.array([0, 1, 2])            # any data aligned with scores
order = np.argsort(scores)              # [1, 2, 0]
scores[order]                           # [10, 20, 30]
names[order]                            # [1, 2, 0]   same reordering
```

**Stability.** The default sort (`kind="quicksort"`) is **not** stable: equal keys can come out
in any order. With `kind="stable"`, ties keep their input order:

```
key:    2  1  2  1          stable argsort -> [1, 3, 0, 2]   (1 before 3, 0 before 2)
index:  0  1  2  3          default argsort -> may give [3, 1, 2, 0]
```

**lexsort** sorts by the **last** key first, then the earlier ones break ties:

```python
dept = np.array([2, 1, 2, 1])
age  = np.array([30, 40, 20, 25])
np.lexsort((age, dept))                 # by dept, then by age: [3, 1, 2, 0]
```

```
row  dept  age            sorted:  row 3: dept 1, age 25
 0    2    30                      row 1: dept 1, age 40
 1    1    40                      row 2: dept 2, age 20
 2    2    20                      row 0: dept 2, age 30
 3    1    25
```

**partition:**

```python
x = np.array([7, 2, 9, 1, 5])
np.partition(x, 2)          # e.g. [1, 2, 5, 7, 9] or [2, 1, 5, 9, 7]: only x[2] is guaranteed
np.partition(x, 2)[:3]      # the 3 smallest, in no particular order
```

Full sort: O(n log n). Partition: O(n) on average.

**Ranks** ("what position would each element have after sorting?") are the inverse permutation of
argsort: `ranks[order] = np.arange(n)` (007), or `np.argsort(order)`.

## Gotchas

- `x.sort()` sorts **in place** and returns `None`; `np.sort(x)` returns a sorted copy.
- `np.lexsort`'s **last** key is the primary one. It's backwards from what you'd guess.
- For descending order, `np.argsort(-x)` breaks for unsigned ints (wraparound, 002) and reverses
  the order of ties. Use `np.argsort(x, kind="stable")[::-1]` if you just need descending, and
  think about what ties should do.
- NaN sorts to the end.
- PyTorch: `torch.sort` returns `(values, indices)` together; `stable=True` is the stable option.
  `torch.topk` is the partition-style tool.

## Builds on

007, 011.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `sort_rows_by_column(data, col)` - the rows of a 2-D array sorted by column `col`, ascending,
   **stable** (rows with equal keys keep their order).
2. `order_by_two_keys(primary, secondary)` - the indices that sort by `primary`, then by
   `secondary` for ties (then by original position).
3. `k_smallest(x, k)` - the `k` smallest values of a 1-D array, in any order. `np.sort` and
   `np.argsort` are banned: use partition.
4. `rank(x)` - for each element, its position in the stable ascending sort (0 = smallest).

## Hints

<details>
<summary>Hint 1</summary>

`sort_rows_by_column`: `data[np.argsort(data[:, col], kind="stable")]`.

</details>

<details>
<summary>Hint 2</summary>

`order_by_two_keys`: `np.lexsort((secondary, primary))`. (lexsort is always stable.)

</details>

<details>
<summary>Hint 3</summary>

`rank`: `order = np.argsort(x, kind="stable")`, then invert the permutation.

</details>

## Self-check

1. What does `np.argsort([30, 10, 20])` return?
2. Why does stability matter when sorting table rows?
3. In `np.lexsort((a, b))`, which key decides first?
4. When is `np.partition` better than `np.sort`?

<details>
<summary>Answers</summary>

1. `[1, 2, 0]`.
2. Rows with equal keys may otherwise be shuffled, losing an earlier ordering (e.g. by date).
3. `b`, the last one.
4. When you need only the k smallest (or a median-like element), not the full order: it's O(n).

</details>

## Go deeper

- [Sorting, searching, and counting](https://numpy.org/doc/stable/reference/routines.sort.html)
