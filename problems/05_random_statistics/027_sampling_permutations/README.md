# 027 · Sampling: choice, permutation, shuffle

**Category:** Random and statistics · **Difficulty:** Easy

## What it is

Three Generator methods for picking and reordering:

- `rng.permutation(x)` - a shuffled **copy** (or, for an int `n`, a random order of `0..n-1`).
- `rng.shuffle(x)` - shuffles `x` **in place** and returns `None`.
- `rng.choice(a, size, replace, p)` - draws elements, with or without replacement, optionally
  with probabilities `p`.

## Why it exists

Train/test splits, shuffling a dataset every epoch, picking a random subset to inspect,
sampling classes in proportion to weights (or to rebalance them). Doing these with index
arrays (007) keeps related arrays aligned, which is the part that usually goes wrong.

## How it works

```python
rng = np.random.default_rng(0)
rng.permutation(5)                 # e.g. [4 2 0 3 1]: each of 0..4 exactly once
rng.permutation(np.array([10, 20, 30]))   # shuffled copy, input unchanged

a = np.arange(5)
rng.shuffle(a)                     # a itself is now shuffled; returns None
```

**Keeping arrays aligned.** Shuffle **indices** once, then index every array with them:

```
perm:  [2, 0, 1]
x:     [x0, x1, x2]   ->  x[perm] = [x2, x0, x1]
y:     [y0, y1, y2]   ->  y[perm] = [y2, y0, y1]    pairs stay together
```

Shuffling `x` and `y` with two separate calls gives two **different** orders and scrambles the
labels.

**Train/test split** from one permutation:

```python
perm = rng.permutation(n)
n_test = round(0.2 * n)
test, train = perm[:n_test], perm[n_test:]     # disjoint, together cover everything
```

**choice:**

```python
rng.choice(10, size=3, replace=False)          # 3 distinct ints from 0..9
rng.choice(np.array([5, 7, 9]), size=6)        # with replacement (the default)
rng.choice(3, size=1000, p=[0.7, 0.2, 0.1])    # weighted
```

`p` must be non-negative and sum to 1. Raw weights (counts, scores) need normalizing first:
`p = w / w.sum()`.

## Gotchas

- `rng.shuffle(x)` returns `None`. `x = rng.shuffle(x)` sets `x` to `None`.
- `rng.shuffle` and `rng.permutation` on a 2-D array reorder **rows only** (axis 0). Each row
  keeps its contents. `rng.permuted(x, axis=1)` shuffles inside each row independently.
- `replace=True` is the default for `choice`. Asking for more distinct samples than exist
  (`replace=False`, `size > len(a)`) raises `ValueError`.
- `p` that sums to 0.999 because of rounding raises an error. Normalize with a division.
- PyTorch: `torch.randperm(n)` is `rng.permutation(n)`; `torch.multinomial` does weighted sampling.

## Builds on

007, 026.

## Your task

Implement in `problem.py`. Use the Generator you're given, follow the recipe in each docstring
(tests compare against the same calls), and use **no Python loops**:

1. `split_indices(n, test_frac, rng)` - `(train, test)` index arrays from one permutation.
2. `shuffle_together(x, y, rng)` - shuffled copies of `x` and `y` in the **same** order; inputs
   unchanged.
3. `sample_distinct(population, k, rng)` - `k` different elements of a 1-D array.
4. `weighted_sample(classes, weights, n, rng)` - `n` draws from `classes` with probability
   proportional to the non-negative `weights` (not necessarily summing to 1).

## Hints

<details>
<summary>Hint 1</summary>

Tasks 1 and 2 both start with `perm = rng.permutation(n)`.

</details>

<details>
<summary>Hint 2</summary>

Tasks 3 and 4 are each one `rng.choice` call; task 4 needs `p=weights / weights.sum()`.

</details>

## Self-check

1. What is the difference between `rng.shuffle(x)` and `rng.permutation(x)`?
2. Why must you shuffle features and labels with one shared permutation?
3. What does `rng.choice(5, size=5, replace=False)` return?
4. What happens if `p` doesn't sum to 1?

<details>
<summary>Answers</summary>

1. `shuffle` reorders `x` in place and returns `None`; `permutation` returns a shuffled copy.
2. Two separate shuffles give two different orders, so each feature row ends up with the wrong label.
3. A random permutation of `0..4`.
4. `ValueError`: probabilities must sum to 1 (normalize raw weights first).

</details>

## Go deeper

- [Generator.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html)
