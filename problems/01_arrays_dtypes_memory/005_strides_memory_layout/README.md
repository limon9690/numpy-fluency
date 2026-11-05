# 005 · Strides and memory layout

**Category:** Arrays, dtypes, memory · **Difficulty:** Medium

## What it is

Memory is one long line of bytes, but arrays have rows, columns and more. **Strides** connect the
two: `a.strides[k]` is how many bytes you jump to move one step along axis `k`. Shape + strides +
a start address fully describe where every element lives.

## Why it exists

Strides are what make views free (004). A transpose doesn't move any data: it swaps the strides.
A slice with step 2 doubles a stride. `broadcast_to` uses a stride of **0** to repeat a row
without copying it. Layout also matters for speed and for libraries (C code, PyTorch, OpenCV)
that require "C-contiguous" memory.

## How it works

A `(2, 3)` int32 array (itemsize 4 bytes) in **C order** (row-major, NumPy's default):

```
logical          memory (bytes 0..23)
 0 1 2           [ 0 | 1 | 2 | 3 | 4 | 5 ]
 3 4 5             row 0       row 1

strides = (12, 4): next row = +12 bytes, next column = +4 bytes
```

In **F order** (column-major, Fortran/MATLAB style) columns are stored one after another:

```
memory: [ 0 | 3 | 1 | 4 | 2 | 5 ]     strides = (4, 8)
```

The general rule for C order: the last axis has stride `itemsize`, and each earlier axis has
stride = (next axis's stride) × (next axis's length). F order is the same from the other end.

Address of an element: `start + sum(index[k] * strides[k])`. For `a[1, 2]` in C order:
`1*12 + 2*4 = 20` bytes from the start.

Views just change these numbers:

```python
a = np.arange(6, dtype=np.int32).reshape(2, 3)
a.strides          # (12, 4)
a.T.strides        # (4, 12)    same memory, axes swapped
a[:, ::2].strides  # (12, 8)    skip every other column
a[::-1].strides    # (-12, 4)   negative stride: walk backwards
a.T.flags.c_contiguous   # False
```

**Contiguous** means the elements fill one solid block in the order the flag names.
`np.ascontiguousarray(x)` returns `x` itself if it already is C-contiguous, otherwise a C-ordered copy.

**Zero strides.** `np.broadcast_to(row, (1000, 3))` has strides `(0, 4)`: every "row" is the same
3 numbers in memory. It costs nothing, and it is **read-only**, because writing one element
would change all 1000 rows.

## Gotchas

- `.T` and `np.moveaxis` don't change memory, so the result is usually not C-contiguous. Many
  C extensions and image libraries need a contiguous buffer: call `np.ascontiguousarray` first.
- Writing to a `broadcast_to` result raises `ValueError: assignment destination is read-only`.
  Call `.copy()` if you need a writable array.
- `a.strides` is in **bytes**; PyTorch's `t.stride()` is in **elements**.
- An array can be both C- and F-contiguous (any 1-D contiguous array, or shapes with 1s).

## Builds on

002, 004.

## Your task

Implement in `problem.py`:

1. `strides_for(shape, itemsize, order)` - the strides (bytes) of a fresh array with this
   shape, element size and order (`"C"` or `"F"`). Compute them; don't allocate an array.
2. `element_offset(a, index)` - the byte distance from `a[0, ..., 0]` to `a[index]`, using `a.strides`.
   Works for any view, including transposed and reversed ones.
3. `ensure_c_contiguous(a)` - a C-contiguous array equal to `a`; no copy if `a` already is one.
4. `repeat_rows_view(row, n)` - a read-only `(n, len(row))` array where every row is `row`,
   without copying any data.

## Hints

<details>
<summary>Hint 1</summary>

For C order, walk the shape from the last axis to the first, carrying a running product that
starts at `itemsize`. For F order, walk from first to last.

</details>

<details>
<summary>Hint 2</summary>

`element_offset` is one line: `sum(i * s for i, s in zip(index, a.strides))`. Negative strides
give negative offsets, and that's correct.

</details>

<details>
<summary>Hint 3</summary>

`np.ascontiguousarray` and `np.broadcast_to` do exactly what tasks 3 and 4 ask.

</details>

## Self-check

1. A C-ordered float64 array has shape `(4, 5)`. What are its strides?
2. What happens to the strides when you transpose an array? Is any data moved?
3. Why is the output of `np.broadcast_to` read-only?
4. What does a negative stride mean?

<details>
<summary>Answers</summary>

1. `(40, 8)`: one row is 5 elements × 8 bytes.
2. They are reversed (swapped for 2-D). No data moves; only the view's metadata changes.
3. Many logical elements share one memory location (stride 0), so one write would change them all.
4. Moving forward along that axis moves backward in memory, e.g. `a[::-1]`.

</details>

## Go deeper

- [Internal memory layout of an ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)
