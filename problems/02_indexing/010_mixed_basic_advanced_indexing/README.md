# 010 · Mixing basic and advanced indexing, np.ix_

**Category:** Indexing · **Difficulty:** Hard

## What it is

You can put slices and index arrays in the same index, like `video[:, :, :, [0, 2]]`. NumPy then
has to decide **where the new axes go** in the result. The rule is simple once you know it, but it
surprises almost everyone the first time. `np.ix_` is the tool for the other common wish:
picking a full **block** of rows × columns.

## Why it exists

Real arrays have several axes (batch, time, height, width, channel). You often want "these
channels, all pixels" or "for each sample, these items, all features". Mixing slices and index
arrays does that in one step, without loops.

## How it works

Take `x` with shape `(5, 6, 7, 3)`. Integers and index arrays are both "advanced" here. All the
advanced indices broadcast together into one shape (call it `B`). Then:

**Rule 1: advanced indices next to each other** -> `B` goes where they were.

```python
x[:, [0, 1], :, :]        # (5, 2, 7, 3)
x[:, [0, 1], [2, 3], :]   # (5, 2, 3)     the two arrays are paired -> one axis of 2
```

**Rule 2: advanced indices separated by a slice** -> `B` goes **first**, because NumPy can't
tell where else to put it.

```python
x[[0, 1], :, :, [0, 1]]   # (2, 6, 7)
x[0, :, :, [0, 1]]        # (2, 6, 7)     <- the surprise! You expected (6, 7, 2).
```

In the last line, the integer `0` counts as advanced, and it is separated from `[0, 1]` by
slices, so the "2" jumps to the front. Two fixes:

```python
x[0][:, :, [0, 1]]        # (6, 7, 2)   index in two steps (x[0] is a cheap view)
x[0, ..., [0, 1]]         # still (2, 6, 7): the Ellipsis is a slice too
```

**Paired vs crossed.** Index arrays are always paired element by element (007).
For the block of rows × columns, use `np.ix_`, which reshapes them so they broadcast into a grid:

```python
a = np.arange(12).reshape(3, 4)
a[[0, 2], [1, 3]]            # [1, 11]       pairs (0,1) and (2,3)
a[np.ix_([0, 2], [1, 3])]    # 2x2 block
```

```
a:                  a[np.ix_([0, 2], [1, 3])]:
 0  1  2  3            1  3
 4  5  6  7            9 11
 8  9 10 11
```

`np.ix_([0, 2], [1, 3])` returns index arrays of shapes `(2, 1)` and `(1, 2)`, so they broadcast
to `(2, 2)`. It also accepts boolean masks.

**Broadcasting index arrays yourself.** For a batch `x` of shape `(B, N, D)` and `idx` of shape
`(B, K)`: `x[np.arange(B)[:, None], idx]` has index shapes `(B, 1)` and `(B, K)`, which broadcast
to `(B, K)`, followed by the untouched `D` axis: result `(B, K, D)`.

## Gotchas

- An integer index next to an index array, separated by a slice, moves axes to the front (Rule 2).
- `a[rows][:, cols]` also gives a block, but makes a temporary copy first; `np.ix_` does it in
  one step and also works for **assignment**: `a[np.ix_(r, c)] = 0`.
- PyTorch only partly matches. Two index arrays separated by a slice behave the same
  (`x[[0, 1], :, :, [0, 1]]` is `(2, 6, 7)` in both). But PyTorch treats a plain integer as basic
  indexing, so `x[0, :, :, [0, 1]]` is `(6, 7, 2)` in PyTorch and `(2, 6, 7)` in NumPy.
  Watch for this when porting code between the two.
- Mixed indexing always returns a copy, because it contains advanced indexing.

## Builds on

006, 007, 008. (The batch trick comes back in the next problem, `take_along_axis`.)

## Your task

Implement in `problem.py`, with **no Python loops**:

1. `submatrix(a, rows, cols)` - the block `rows × cols` of a 2-D array; `rows` and `cols` may be
   int arrays or boolean masks.
2. `channel_pixels(img, ys, xs)` - for an `(H, W, C)` image and point coordinates, return the
   `(k, C)` pixel values.
3. `select_frame_channels(video, t, channels)` - from `(T, H, W, C)`, frame `t` and the listed
   channels, as `(H, W, len(channels))`.
4. `batch_gather(x, idx)` - `x` is `(B, N, D)`, `idx` is `(B, K)`; return `(B, K, D)` with
   `out[b, k] = x[b, idx[b, k]]`. (`np.take_along_axis` and `np.take` are banned here.)

## Hints

<details>
<summary>Hint 1</summary>

`submatrix` is one line with `np.ix_`. `channel_pixels` is one line with two paired index arrays
and no trailing index.

</details>

<details>
<summary>Hint 2</summary>

`select_frame_channels`: `video[t, :, :, channels]` has the wrong shape (Rule 2). Index in two
steps, or move the axis back with `np.moveaxis`.

</details>

<details>
<summary>Hint 3</summary>

`batch_gather`: you need a row index of shape `(B, 1)` next to `idx` of shape `(B, K)`.

</details>

## Self-check

1. What shape is `x[:, [0, 1], :, [0, 1]]` for `x` of shape `(5, 6, 7, 3)`?
2. Why does `x[0, :, :, [0, 1]]` put the length-2 axis first?
3. What is the difference between `a[[0, 2], [1, 3]]` and `a[np.ix_([0, 2], [1, 3])]`?
4. What shapes does `np.ix_` give to its outputs for inputs of length 2 and 3?

<details>
<summary>Answers</summary>

1. `(2, 5, 7)`: the two advanced indices are separated by a slice, so their shape `(2,)` goes first.
2. The integer `0` is an advanced index too; it and `[0, 1]` are separated by slices.
3. The first gives 2 paired elements, shape `(2,)`; the second gives the `(2, 2)` block.
4. `(2, 1)` and `(1, 3)`.

</details>

## Go deeper

- [Combining advanced and basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing)
