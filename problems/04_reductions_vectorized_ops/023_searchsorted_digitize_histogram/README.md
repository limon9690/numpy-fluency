# 023 · searchsorted, digitize, histogram, interp

**Category:** Reductions and vectorized ops · **Difficulty:** Medium

## What it is

Four tools that work with **sorted** positions:

- `np.searchsorted(a, v)` - where would each `v` go in the sorted array `a` to keep it sorted?
- `np.digitize(x, bins)` - which bin does each `x` fall into?
- `np.histogram(x, bins)` - how many `x` fall into each bin?
- `np.interp(x_new, x, y)` - linear interpolation of a 1-D function at new points.

## Why it exists

They replace loops of `if/elif` over thresholds: turning ages into age groups, latency values
into buckets, counting events in time windows, resampling a sensor signal onto a new time grid.
All of them use binary search, so they're O(log n) per query.

## How it works

**searchsorted** with `side`:

```python
a = np.array([10, 20, 20, 30])
np.searchsorted(a, 20)                 # 1   (left: before the existing 20s)
np.searchsorted(a, 20, side="right")   # 3   (right: after them)
np.searchsorted(a, [5, 25, 99])        # [0, 3, 4]
```

```
index:   0    1    2    3
a:      10   20   20   30

insert 20, side="left"   -> slot 1  (before the 20s)
insert 20, side="right"  -> slot 3  (after the 20s)
insert 5 -> 0     insert 25 -> 3     insert 99 -> 4 (past the end)
```

**Counting in a range** `[lo, hi)` of a sorted array is two searches:
`searchsorted(a, hi) - searchsorted(a, lo)`. It works for many ranges at once.

**Bins.** With edges `[0, 10, 20]`, `searchsorted(edges, x, side="right") - 1` gives the bin
`i` with `edges[i] <= x < edges[i+1]`:

```
x:      -5   0   9.9  10   25
bin:    -1   0   0    1    2      (-1 = below the first edge, 2 = at or past the last edge)
```

`np.digitize(x, edges)` returns the same thing plus 1.

**histogram:**

```python
counts, edges = np.histogram([0.1, 0.4, 0.5, 1.0], bins=2, range=(0, 1))
counts      # [2, 2]
edges       # [0. , 0.5, 1. ]
```

Every bin is half-open `[a, b)` **except the last one**, which is closed `[a, b]`. That's why
`1.0` counted. Values outside `range` are dropped.

**interp:**

```python
t = np.array([0.0, 1.0, 2.0]);  y = np.array([0.0, 10.0, 0.0])
np.interp([0.5, 1.5, 3.0], t, y)       # [5., 5., 0.]   (3.0 is clamped to the last value)
```

## Gotchas

- `searchsorted` and `interp` assume the input is **sorted increasing**. They don't check, and
  give silently wrong answers otherwise.
- `side="left"` vs `"right"` only matters for values equal to an edge, which is exactly where
  bugs hide.
- The last histogram bin includes its right edge; all others don't.
- `np.interp` clamps outside the data range instead of extrapolating (use `left=`/`right=` to
  set other values).
- PyTorch: `torch.searchsorted` and `torch.bucketize` exist; `torch.histc` is like
  `np.histogram` but returns only counts.

## Builds on

018, 021.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `bucketize(x, edges)` - the bin index `i` with `edges[i] <= x < edges[i+1]` for each `x`;
   `-1` below `edges[0]`, `len(edges) - 1` at or above `edges[-1]`.
2. `count_in_windows(times, starts, ends)` - for each window `[starts[k], ends[k])`, how many of
   the sorted `times` fall inside.
3. `histogram_counts(x, n_bins, lo, hi)` - the same `(counts, edges)` as
   `np.histogram(x, n_bins, range=(lo, hi))`, built yourself (`np.histogram` is banned).
4. `resample_linear(t, y, t_new)` - linear interpolation of `y(t)` at `t_new`, clamped at the
   ends; raise `ValueError` if `t` is not strictly increasing.

## Hints

<details>
<summary>Hint 1</summary>

`bucketize`: `np.searchsorted(edges, x, side="right") - 1`.

</details>

<details>
<summary>Hint 2</summary>

`histogram_counts`: `edges = np.linspace(lo, hi, n_bins + 1)`; bin numbers with `bucketize`;
put values equal to `hi` into the last bin; drop the rest that are out of range; `np.bincount(..., minlength=n_bins)`.

</details>

## Self-check

1. What does `np.searchsorted([1, 3, 5], 3)` return? And with `side="right"`?
2. In `np.histogram`, which bin gets a value exactly equal to the upper end of the range?
3. How do you count the elements of a sorted array in `[lo, hi)` without a mask?
4. What does `np.interp` do for a point beyond the last `x`?

<details>
<summary>Answers</summary>

1. `1`; with `side="right"`, `2`.
2. The last bin: it's closed on the right.
3. `np.searchsorted(a, hi) - np.searchsorted(a, lo)`.
4. It returns the last `y` value (clamping), unless you pass `right=`.

</details>

## Go deeper

- [numpy.searchsorted](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html)
