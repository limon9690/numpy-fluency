# 002 · dtypes, casting and overflow

**Category:** Arrays, dtypes, memory · **Difficulty:** Easy

## What it is

Every array has one **dtype**: the type and size of each element (`uint8`, `int64`,
`float32`, ...). Fixed-size integers can overflow, float-to-int conversion has rules, and when
two dtypes meet in one operation NumPy picks a result dtype ("type promotion").

## Why it exists

Fixed-size elements are what make NumPy fast and compact: one million `uint8` pixels take 1 MB,
not 28 MB of Python ints. The price is that you must think about ranges. An 8-bit image can
only hold 0-255, and NumPy will not stop you from leaving that range.

## How it works

**Overflow wraps around** (modulo 2⁸ for uint8). No error, no warning, for arrays:

```python
img = np.array([250, 100], dtype=np.uint8)
img + 10        # array([  4, 110], dtype=uint8)   <- 260 wrapped to 4
```

The fix: do the math in a wider dtype, clip, and cast back:

```
uint8 [250, 100] --astype(int16)--> [250, 100] --+10--> [260, 110] --clip 0..255--> [255, 110] --astype(uint8)
```

**Float to int truncates toward zero**: `np.array([1.7, -1.7]).astype(np.int64)` is `[1, -1]`.
Round first if you want the nearest integer.

**`np.round` rounds half to even** ("banker's rounding"):

```python
np.round([0.5, 1.5, 2.5, -0.5])    # [ 0.,  2.,  2., -0.]
```

This avoids a bias when you round many values, but it's not school rounding.
School rounding (half away from zero) is `sign(x) * floor(|x| + 0.5)`.

**Ranges:** `np.iinfo(np.int8)` has `.min == -128` and `.max == 127`. `np.finfo` does the same
for floats.

**Promotion in NumPy 2** (NEP 50). Python numbers are "weak": they adapt to the array.
NumPy scalars and arrays are "strong": they keep their own dtype.

| expression | result dtype |
|---|---|
| `float32_array * 0.5` | float32 (Python float adapts) |
| `float32_array * np.float64(0.5)` | **float64** (NumPy scalar is strong) |
| `uint8_array + 300` | `OverflowError` (300 doesn't fit uint8) |
| `int8_array + np.int64(1)` | int64 |

So a `np.float64` coming out of `x.mean()` can silently turn your float32 data into float64.

## Gotchas

- **In-place ops can't change dtype.** `a = np.array([1, 2]); a += 0.5` raises
  `UFuncTypeError` (a `TypeError`), because the float result can't be stored in an int array.
  `a = a + 0.5` works: it makes a new float array.
- `astype` always makes a copy (unless you pass `copy=False` and nothing changes).
- uint8 subtraction wraps too: `np.uint8(5) - np.uint8(10)` is `251`.
- PyTorch also wraps on uint8 overflow, but its promotion rules differ in details. Don't assume
  the same result dtype on both sides of the bridge (see 039).
- `.astype(np.uint8)` on floats outside 0-255 gives garbage (undefined behaviour), not clipping.

## Builds on

001.

## Your task

Implement in `problem.py`:

1. `brighten(img, amount)` - add `amount` (may be negative) to a uint8 array, **saturating** at 0
   and 255 instead of wrapping. Return uint8; don't modify the input.
2. `round_half_up(x)` - round floats to the nearest integer, halves away from zero
   (`2.5 -> 3`, `-2.5 -> -3`). Return float64.
3. `to_int_safe(x, dtype)` - round with `np.round` (half to even), then cast to the integer
   `dtype`. Raise `ValueError` for NaN/inf and `OverflowError` if a rounded value doesn't fit.
4. `scale_keep_dtype(a, factor)` - multiply a float array by `factor` and keep `a`'s dtype,
   even when `factor` is a NumPy float64 scalar.

## Hints

<details>
<summary>Hint 1</summary>

For `brighten`, go to `np.int16` (or wider) first: `img.astype(np.int16) + amount` can't wrap.
Then `np.clip(..., 0, 255)` and `.astype(np.uint8)`.

</details>

<details>
<summary>Hint 2</summary>

For `to_int_safe`: `np.isfinite(x).all()`, then `info = np.iinfo(dtype)` and compare the rounded
values with `info.min` and `info.max` **before** casting.

</details>

<details>
<summary>Hint 3</summary>

For `scale_keep_dtype`: `(a * factor).astype(a.dtype)`, or `np.multiply(a, factor, dtype=a.dtype)`.

</details>

## Self-check

1. What is `np.array([200], dtype=np.uint8) + np.array([100], dtype=np.uint8)`?
2. What does `np.round(3.5)` give? And `np.round(4.5)`?
3. Why does `a += 0.5` fail on an int array while `a = a + 0.5` works?
4. In NumPy 2, what is the dtype of `np.ones(3, np.float32) + 1.0`?

<details>
<summary>Answers</summary>

1. `array([44], dtype=uint8)`: 300 wraps around to 300 - 256.
2. `4.0` and `4.0`: halves go to the even neighbour.
3. In-place ops must store the result in the existing int memory; a float can't be cast there
   under the default `same_kind` rule. `a + 0.5` allocates a new float64 array.
4. float32: the Python float `1.0` is weak and adapts to the array.

</details>

## Go deeper

- [Data type promotion in NumPy](https://numpy.org/doc/stable/reference/arrays.promotion.html)
