# 024 · Set operations: isin, intersect1d, setdiff1d, union1d

**Category:** Reductions and vectorized ops · **Difficulty:** Easy

## What it is

NumPy has vectorized versions of Python's set operations, working on arrays:

- `np.isin(x, test)` - for every element of `x`, is it in `test`? (a mask, same shape as `x`)
- `np.intersect1d(a, b)` - values in both
- `np.union1d(a, b)` - values in either
- `np.setdiff1d(a, b)` - values in `a` but not in `b`
- `np.setxor1d(a, b)` - values in exactly one of them

All except `isin` return **sorted, unique, 1-D** arrays.

## Why it exists

"Which user IDs are new since yesterday?", "Keep only words in the vocabulary", "Which samples
appear in both the train and test split?" (a data leak check). With Python sets you'd convert
back and forth and loop; these work directly on arrays and are much faster for big ones.

## How it works

```python
a = np.array([5, 1, 5, 3])
b = np.array([3, 5, 7])

np.intersect1d(a, b)     # [3 5]
np.union1d(a, b)         # [1 3 5 7]
np.setdiff1d(a, b)       # [1]
np.setxor1d(a, b)        # [1 7]
```

Duplicates disappear and results are sorted, just like `np.unique` (022).

**isin keeps the shape** of `x`, so it works as a mask (008):

```python
x = np.array([[1, 2],
              [3, 4]])
np.isin(x, [2, 3])
```

```
False  True
 True False
```

```python
np.where(np.isin(x, [2, 3]), x, -1)      # replace everything else with -1
```

```
-1  2
 3 -1
```

**Where were the common values?** `return_indices=True` also gives the position of each common
value's **first** occurrence in `a` and in `b`:

```python
common, ia, ib = np.intersect1d(a, b, return_indices=True)
# common = [3, 5], ia = [3, 0], ib = [0, 1]   -> a[ia] == b[ib] == common
```

**Jaccard similarity** of two sets is `|A ∩ B| / |A ∪ B|`: 1 for identical sets, 0 for
disjoint ones.

## Gotchas

- Results are sorted and unique: you lose the original order and multiplicity. Use `isin`
  with a mask if you need to keep the order of `x`.
- `x in arr` (Python's `in`) asks "is `x` equal to **any** element", for one value only. For
  many values, use `np.isin`.
- `np.in1d` was the old name for a flat `isin`; it's gone in recent NumPy 2 versions. Use `np.isin`.
- Set operations flatten their inputs. They don't work on rows; for that, use `np.unique(..., axis=0)`.
- PyTorch has `torch.isin`, but no `intersect1d`/`union1d`.

## Builds on

008, 022.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `new_ids(old, current)` - the sorted unique IDs in `current` that are not in `old`.
2. `keep_known(x, vocab, unknown)` - a copy of `x` (any shape) where every value not in `vocab`
   is replaced by `unknown`.
3. `common_with_positions(a, b)` - `(common, ia, ib)`: the sorted values in both 1-D arrays and
   the index of each value's first occurrence in `a` and in `b`.
4. `jaccard(a, b)` - `|A ∩ B| / |A ∪ B|` of the distinct values, as a Python float
   (1.0 if both are empty).

## Hints

<details>
<summary>Hint 1</summary>

`new_ids` is one call. `keep_known` is `np.where` with an `np.isin` mask.

</details>

<details>
<summary>Hint 2</summary>

`common_with_positions`: `np.intersect1d(a, b, return_indices=True)`. `jaccard`: the sizes of
`intersect1d` and `union1d`.

</details>

## Self-check

1. What does `np.setdiff1d([3, 1, 3, 9], [1])` return?
2. What shape does `np.isin(x, test)` have?
3. Why does `np.intersect1d` lose the original order of `a`?
4. How would you check that no sample ID is in both the train and test splits?

<details>
<summary>Answers</summary>

1. `[3, 9]`: sorted and without duplicates.
2. The shape of `x`.
3. It sorts both arrays to find matches (the same way `np.unique` works).
4. `np.intersect1d(train_ids, test_ids).size == 0` (or `not np.isin(test_ids, train_ids).any()`).

</details>

## Go deeper

- [Set routines](https://numpy.org/doc/stable/reference/routines.set.html)
