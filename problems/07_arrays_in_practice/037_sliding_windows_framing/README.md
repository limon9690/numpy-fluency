# 037 · Sliding windows and framing (sliding_window_view)

**Category:** Arrays in practice · **Difficulty:** Medium

## What it is

`np.lib.stride_tricks.sliding_window_view(x, window_shape)` returns **every window** of an
array as a new axis, without copying anything. A 1-D signal of length `n` with windows of
length `w` becomes an `(n - w + 1, w)` view. It works in 2-D too (image patches).

## Why it exists

Many operations look at a neighbourhood: moving max/median, audio frames for a spectrogram,
image patches for a filter. Writing them as loops is slow; computing them with tricks like
cumsum (020) only works for sums. A window view turns any of them into "reduce over the last
axis", using everything you already know.

## How it works

```python
from numpy.lib.stride_tricks import sliding_window_view
x = np.arange(6)                       # [0 1 2 3 4 5]
w = sliding_window_view(x, 3)          # shape (4, 3)
```

```
0 1 2
1 2 3
2 3 4
3 4 5
```

No data is copied: the view reuses `x`'s memory with strides `(8, 8)`. Moving one row down
and moving one column right both step **one element** in memory (005):

```
memory of x:  [0 | 1 | 2 | 3 | 4 | 5]
row 0:         ^^^^^^^^^
row 1:             ^^^^^^^^^
row 2:                 ^^^^^^^^^
```

Now any reduction over the last axis is a moving statistic: `w.max(axis=-1)` is a moving max,
`np.median(w, axis=-1)` a moving median.

**Hop size (framing).** Audio is cut into frames of length `L` that start every `hop` samples.
That's every `hop`-th window: `sliding_window_view(x, L)[::hop]`. Frames that would run past the
end are dropped, which gives `1 + (n - L) // hop` frames.

**2-D patches.** `sliding_window_view(img, (k, k))` on an `(H, W)` image has shape
`(H - k + 1, W - k + 1, k, k)`: one `k × k` patch for every position. `.mean(axis=(-2, -1))`
is a box blur ("valid" mode: no padding, so the output is smaller).

```
image (4, 4), k = 2:   patch at (0, 0)   patch at (0, 1)   ...
 0  1  2  3            0 1               1 2
 4  5  6  7            4 5               5 6
 8  9 10 11
12 13 14 15            9 patches in a (3, 3) grid
```

## Gotchas

- The result is **read-only**. Many windows share the same memory, so writing one element
  would change several windows. Call `.copy()` if you really need to write.
- The view is free, but anything that **materializes** it is not: `w + 0`, `w.copy()`,
  `np.ascontiguousarray(w)` allocate `w` times the memory of `x`. A `(3, 3)` patch view of a
  1000 × 1000 image is 9 × 8 MB if copied. Reductions like `w.mean(-1)` are fine (040).
- Don't use the older `as_strided` for this: a wrong stride there reads random memory.
  `sliding_window_view` can't go out of bounds.
- `window_shape` larger than the array raises `ValueError`.
- PyTorch: `Tensor.unfold(dim, size, step)` is the equivalent, and it takes the hop (`step`)
  directly.

## Builds on

005, 018, 020.

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `frames(signal, frame_len, hop)` - the `(n_frames, frame_len)` read-only view of a 1-D
   signal: frames starting every `hop` samples, incomplete ones dropped.
2. `moving_max(x, w)` - the max of every length-`w` window of a 1-D array, length `n - w + 1`.
3. `patches_2d(img, k)` - the `(H - k + 1, W - k + 1, k, k)` view of all `k × k` patches of a
   2-D image.
4. `box_blur_valid(img, k)` - the mean of every `k × k` patch, float64 `(H - k + 1, W - k + 1)`.

## Hints

<details>
<summary>Hint 1</summary>

`frames`: `sliding_window_view(signal, frame_len)[::hop]`. Slicing a view gives a view.

</details>

<details>
<summary>Hint 2</summary>

`patches_2d`: `sliding_window_view(img, (k, k))`. `box_blur_valid`: the mean of that over the
last two axes.

</details>

## Self-check

1. What shape does `sliding_window_view(np.arange(10), 4)` have?
2. Why is the window view read-only?
3. How many frames do you get from 100 samples with frame length 20 and hop 10?
4. Why can `np.ascontiguousarray(windows)` run out of memory when the view itself didn't?

<details>
<summary>Answers</summary>

1. `(7, 4)`.
2. Overlapping windows share memory; writing to one element would silently change other windows.
3. `1 + (100 - 20) // 10 = 9`.
4. A contiguous copy stores every window separately: `w` times the original data.

</details>

## Go deeper

- [sliding_window_view](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html)
