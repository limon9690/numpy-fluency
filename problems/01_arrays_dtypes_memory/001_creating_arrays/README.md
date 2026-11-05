# 001 · Creating arrays

**Category:** Arrays, dtypes, memory · **Difficulty:** Easy

## What it is

Every NumPy program starts by making arrays. `np.array` turns Python lists into an array and
guesses a dtype. `np.zeros`, `np.ones` and `np.full` make an array of a given shape and fill it.
`np.arange` and `np.linspace` make ranges of numbers. The `*_like` functions copy the shape
(and the dtype!) of an existing array.

## Why it exists

An array is a block of memory with one fixed dtype and one fixed shape. You almost always know
the shape before you know the values, so you allocate first and fill later (an output buffer,
an image canvas, a grid of sample points). Choosing the dtype at creation time decides how much
memory you use and what values fit.

## How it works

```python
np.array([1, 2, 3])          # int64   (all ints)
np.array([1, 2.5, 3])        # float64 (one float makes everything float)
np.array([[1, 2], [3, 4]], dtype=np.float32)   # shape (2, 2), float32

np.zeros((2, 3))             # float64 by default
np.full((2, 3), 7, dtype=np.uint8)
```

`np.full((2, 3), 7, dtype=np.uint8)` gives:

```
7 7 7
7 7 7
```

**arange vs linspace.** `arange(start, stop, step)` keeps adding `step` until it reaches `stop`.
With float steps, rounding error decides whether the last point sneaks in:

```python
np.arange(1, 1.3, 0.1)       # [1.0, 1.1, 1.2, 1.3]   <- 4 points, 1.3 is "excluded"!
np.arange(0, 1, 0.1)         # 10 points, as expected
```

`0.1` is not exactly representable, so `1 + 3*0.1` lands just *below* 1.3 and passes the
`< stop` test. `linspace(start, stop, num)` avoids this: you give the **count**, not the step,
and both endpoints are exact.

```python
np.linspace(0, 1, 5)                  # [0.  0.25 0.5  0.75 1.  ]
np.linspace(0, 1, 5, retstep=True)    # (array, 0.25)
```

Rule of thumb: `arange` for integers, `linspace` for floats.

**`*_like`.** `np.zeros_like(a)` = `np.zeros(a.shape, dtype=a.dtype)`. Same dtype is often what
you want, and sometimes a trap (see Gotchas).

## Gotchas

- `np.full_like(int_array, 0.5)` is all **zeros**: the fill value is cast to the int dtype.
  Pass `dtype=` when the result should be float.
- Ragged lists (`[[1, 2], [3]]`) raise `ValueError` in NumPy 2. Older versions made a weird
  `object` array instead.
- Default dtypes: NumPy uses **float64** for `zeros`/`ones`/`linspace`; PyTorch uses float32.
  Integer lists become int64 in both.
- `np.empty` does not zero memory. The values are whatever was there before.
- `arange` with a float step can give one more or one fewer point than you expect.

## Builds on

Nothing. This is the first problem.

## Your task

Implement four functions in `problem.py`:

1. `safe_arange(start, stop, step)` - the points `start, start + step, start + 2*step, ...`
   going toward `stop` (excluded), but with the count fixed as `round((stop - start) / step)`
   so float error can't add or drop a point. Return float64. Empty range -> empty array.
2. `linspace_with_step(start, stop, n)` - `n >= 2` evenly spaced float64 points from `start` to
   `stop`, both included, plus the spacing as a Python `float`.
3. `blank_like(a, fill)` - a new float64 array with the shape of `a`, filled with `fill`
   (even when `a` is an int array).
4. `as_float32_matrix(rows)` - turn a list of lists into a 2-D float32 array. Raise
   `ValueError` if the rows are ragged or the result isn't 2-D.

## Hints

<details>
<summary>Hint 1</summary>

For `safe_arange`, compute the count `n` first, then build `n` points from it. `linspace` takes
an `endpoint=False` argument, or you can scale `np.arange(n)`.

</details>

<details>
<summary>Hint 2</summary>

`np.full(a.shape, fill, dtype=np.float64)` or `np.full_like(a, fill, dtype=np.float64)`.
For the matrix, `np.array(rows, dtype=np.float32)` already raises on ragged input; then check
`.ndim`.

</details>

## Self-check

1. Why can `np.arange(1, 1.3, 0.1)` return 4 points?
2. What dtype does `np.zeros((3,))` have? And `torch.zeros(3)`?
3. What does `np.zeros_like(np.array([1, 2]))` hold after you assign `0.7` to an element?
4. When would you pick `np.empty` over `np.zeros`?

<details>
<summary>Answers</summary>

1. `1 + 3*0.1` is slightly less than `1.3` in floating point, so it passes the `< stop` test.
2. NumPy: float64. PyTorch: float32.
3. `0`: the array is int64 (copied from the input), so `0.7` is truncated.
4. When every element will be overwritten anyway, e.g. an output buffer filled by `out=`.

</details>

## Go deeper

- [Array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html)
