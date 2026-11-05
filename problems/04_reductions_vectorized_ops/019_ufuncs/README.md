# 019 · ufuncs: out=, where=, reduce, accumulate, outer

**Category:** Reductions and vectorized ops · **Difficulty:** Medium

## What it is

A **ufunc** ("universal function") is a NumPy function that works element by element:
`np.add`, `np.multiply`, `np.maximum`, `np.log`, `np.gcd` and dozens more. Every ufunc
accepts `out=` (write the result into an existing array) and `where=` (only compute some
elements). Binary ufuncs also have methods: `.reduce`, `.accumulate` and `.outer`.

## Why it exists

`out=` avoids allocating a new array for every step, which matters for large data (040).
`where=` lets you skip invalid inputs (log of 0, divide by 0) instead of computing garbage and
cleaning up afterwards. The methods turn **any** binary operation into a reduction, a running
total or an all-pairs table, not just `+` and `*`.

## How it works

**out= and where=:**

```python
x = np.array([1.0, 0.0, -1.0, np.e])
out = np.full(x.shape, -np.inf)          # the value for skipped elements
np.log(x, out=out, where=x > 0)          # [0., -inf, -inf, 1.]   no warnings
```

`where=False` positions are **not touched**: they keep whatever `out` held before. That's why
you prefill `out`.

**reduce** applies the operation between neighbours until one value is left (per axis):

```python
np.add.reduce([1, 2, 3, 4])        # 10        same as np.sum
np.maximum.reduce([3, 9, 2])       # 9         same as np.max
np.gcd.reduce([12, 18, 30])        # 6         no np.gcd_all exists - reduce gives it to you
```

**accumulate** keeps every intermediate result:

```python
np.add.accumulate([1, 2, 3, 4])        # [ 1  3  6 10]   = np.cumsum
np.maximum.accumulate([1, 3, 2, 5, 4]) # [1 3 3 5 5]     running maximum
```

**outer** applies the operation to every pair:

```python
np.subtract.outer([10, 20], [1, 2, 3])
```

```
 9  8  7
19 18 17
```

`f.outer(a, b)` has shape `a.shape + b.shape`. It's the same as `f(a[:, None], b[None, :])`
for 1-D inputs (012).

All methods take `axis=` like normal reductions.

## Gotchas

- `where=` without `out=` leaves the skipped elements **uninitialized** (random leftovers in
  memory). NumPy 2 now warns about this. Always pass `out=` with `where=`.
- `out=` must have the right shape and a dtype the result can be cast to (`same_kind`):
  `np.add(int_a, 0.5, out=int_a)` raises (002).
- `np.log(0)` gives `-inf` with a `RuntimeWarning`, `np.log(-1)` gives `nan`. `where=` avoids both.
- `np.maximum` (element-wise, 2 arrays) and `np.max` (a reduction) are different functions.
- PyTorch has no ufunc methods. Use `torch.cummax` or `torch.cumsum` for accumulate, and
  broadcasting for outer.

## Builds on

009, 012, 018.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `safe_log(x, fill)` - natural log where `x > 0`, `fill` elsewhere; float64; no warnings.
2. `running_max(x, axis)` - the running maximum along `axis`.
3. `gcd_all(x)` - the greatest common divisor of all elements of an int array, as a Python int.
4. `pairs_within(a, b, tol)` - an `(len(a), len(b))` boolean table: `|a[i] - b[j]| <= tol`.

Use the ufunc tools from this card (`where=`/`out=`, `.accumulate`, `.reduce`, `.outer`).

## Hints

<details>
<summary>Hint 1</summary>

`safe_log`: `out = np.full(x.shape, fill, dtype=np.float64)`, then
`np.log(x, out=out, where=x > 0)`.

</details>

<details>
<summary>Hint 2</summary>

`running_max` is `np.maximum.accumulate(x, axis=axis)`; `gcd_all` is `np.gcd.reduce` on the
flattened array.

</details>

## Self-check

1. What happens to the elements where `where=` is False?
2. What is the difference between `np.add.reduce` and `np.add.accumulate`?
3. What shape does `np.multiply.outer(a, b)` have for `a` of shape `(2, 3)` and `b` of shape `(4,)`?
4. Why would you use `out=` at all?

<details>
<summary>Answers</summary>

1. They aren't written: they keep the value already in `out` (or garbage if there is no `out`).
2. `reduce` returns only the final result; `accumulate` returns every intermediate step.
3. `(2, 3, 4)`.
4. To reuse memory instead of allocating a new array each time, which saves memory and time on
   big arrays.

</details>

## Go deeper

- [Universal functions (ufunc)](https://numpy.org/doc/stable/reference/ufuncs.html)
