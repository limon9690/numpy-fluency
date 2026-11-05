# 020 · cumsum, diff and running statistics

**Category:** Reductions and vectorized ops · **Difficulty:** Easy

## What it is

`np.cumsum` gives running totals; `np.diff` gives the differences between neighbours. They are
inverses of each other (almost exactly). Together they answer "running" questions in one
vectorized step: moving averages, running means, offsets of variable-length segments, and the
places where a signal changes.

## Why it exists

A moving average written as a loop is slow; with cumsum it's two subtractions. Segment offsets
("where does sample `i` start in the packed array?") are a cumsum. Event detection ("when did
the sensor switch on?") is a diff. These patterns show up in time series, audio and data
pipelines all the time.

## How it works

```python
x = np.array([3, 1, 4, 1, 5])
np.cumsum(x)                 # [ 3  4  8  9 14]
np.diff(x)                   # [-2  3 -3  4]       length n - 1
np.diff(np.cumsum(x))        # [1 4 1 5]           x without its first element
```

**Moving average with cumsum.** Put a 0 in front of the running total: `c = [0, 3, 4, 8, 9, 14]`.
The sum of any window `x[i:i+w]` is `c[i+w] - c[i]`:

```
c:          0   3   4   8   9  14
window 3:   c[3]-c[0]=8   c[4]-c[1]=6   c[5]-c[2]=10
average:    2.67          2.0           3.33
```

So all window sums are `c[w:] - c[:-w]`, one subtraction for the whole array.

**Running mean:** `np.cumsum(x) / np.arange(1, n + 1)` = mean of the first 1, 2, 3, ... elements.

**Offsets from lengths.** Three segments of lengths `[2, 3, 1]` packed into one array of 6:

```
lengths:  [2, 3, 1]
offsets:  [0, 2, 5, 6]     segment k is packed[offsets[k]:offsets[k+1]]
```

`offsets = np.concatenate([[0], np.cumsum(lengths)])`. This layout (one flat array + offsets) is
how ragged data is stored efficiently, e.g. variable-length sentences.

**Edges of a boolean signal.** Convert to int first, then diff: `+1` = switched on, `-1` = off.

```
signal:   F  F  T  T  F  T
as int:   0  0  1  1  0  1
diff:        0  1  0 -1  1      rising edges at indices 2 and 5 (diff index + 1)
```

`np.diff(x, prepend=x[0])` keeps the length at `n` if you need it.

## Gotchas

- `np.diff` on **uint8** wraps: `np.diff(np.array([5, 3], np.uint8))` is `[254]`. Cast to a
  signed type first.
- `np.diff` on **bool** gives XOR (`True` wherever the value changed), so you can't tell on from
  off. Cast to int8 first.
- `cumsum` on float32 adds one by one and drifts: summing a million `0.1`s gives about
  `100958` with `cumsum`, but `100000.01` with `sum` (which adds in pairs). Use float64 for long
  running totals.
- The cumsum moving average is fast but can lose precision for huge, long signals (same reason).
- `cumsum` of small ints promotes like `sum`: int8 gives int64, uint8 gives uint64.
- PyTorch: `torch.cumsum(t, dim)` needs `dim`; `torch.diff` exists too.

## Builds on

016, 018, 019.

## Your task

Implement in `problem.py`, with **no Python loops** (and without `np.convolve` or
`sliding_window_view`):

1. `moving_average(x, w)` - means of every window of length `w` of a 1-D array; float64, length
   `n - w + 1`.
2. `running_mean(x)` - `out[i]` = mean of `x[:i+1]`; float64.
3. `lengths_to_offsets(lengths)` - the int64 offsets array of length `len(lengths) + 1`,
   starting at 0.
4. `rising_edges(signal)` - the indices `i` where a 1-D bool signal goes from False (at `i-1`)
   to True (at `i`).

## Hints

<details>
<summary>Hint 1</summary>

`moving_average`: `c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])`, then
`(c[w:] - c[:-w]) / w`.

</details>

<details>
<summary>Hint 2</summary>

`rising_edges`: `d = np.diff(signal.astype(np.int8))`, then `np.flatnonzero(d == 1) + 1`.

</details>

## Self-check

1. What length does `np.diff` return for an input of length `n`?
2. Why does the cumsum moving average start with a 0 in front?
3. Why must you convert a boolean signal to int before `np.diff`?
4. Why can `np.cumsum` on float32 give a different total than `np.sum`?

<details>
<summary>Answers</summary>

1. `n - 1`.
2. So that the first window's sum `c[w] - c[0]` works like all the others.
3. `diff` on bools gives "changed / not changed" (XOR); you lose the direction of the change.
4. `cumsum` adds strictly left to right, so rounding error grows; `sum` uses pairwise
   summation, which keeps the error much smaller.

</details>

## Go deeper

- [numpy.cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
