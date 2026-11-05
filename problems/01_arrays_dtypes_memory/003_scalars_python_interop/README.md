# 003 · NumPy scalars and Python interop

**Category:** Arrays, dtypes, memory · **Difficulty:** Easy

## What it is

When you pull a single value out of an array, you don't get a Python `int` or `float`. You get a
**NumPy scalar** such as `np.int64` or `np.float32`. There are also **0-d arrays**: arrays with
shape `()`. `.item()` and `.tolist()` convert NumPy values back into plain Python objects.

## Why it exists

NumPy scalars keep the dtype, so `x[0] + x[1]` follows the same rules as the array (uint8 still
wraps, float32 stays float32). But the rest of the Python world (`json`, many libraries, config
files, logging) only understands built-in types. At that boundary you have to convert.

## How it works

```python
a = np.array([1.5, 2.5], dtype=np.float32)

a[0]              # np.float32(1.5)     NumPy scalar, shape ()
a.sum()           # np.float32(4.0)     reductions also return NumPy scalars
a[0].item()       # 1.5                 Python float
a.tolist()        # [1.5, 2.5]          list of Python floats
np.array(7)       # array(7)            a 0-d array, ndim == 0, shape ()
np.array(7).item()  # 7
```

`json` breaks on most NumPy types:

```python
json.dumps({"n": np.int64(3)})        # TypeError: Object of type int64 is not JSON serializable
json.dumps({"n": np.int64(3).item()}) # '{"n": 3}'
json.dumps(np.zeros(2).tolist())      # '[0.0, 0.0]'
```

`.tolist()` converts recursively, so a `(2, 3)` array becomes a list of 2 lists of 3 Python numbers.
`.item()` works on any array with exactly one element, whatever its shape.

To store an array as JSON and get it back **exactly**, you need three things: the values, the
shape and the dtype. The values alone lose the dtype (`uint8` comes back as `int64`) and,
for empty arrays, the shape.

```
array (2, 2) uint8          JSON
  1  2      ->   {"dtype": "uint8", "shape": [2, 2], "data": [1, 2, 3, 4]}
  3  4
```

## Gotchas

- `np.float64` is a subclass of Python `float`, so `json.dumps(np.float64(1.0))` happens to work.
  `np.float32`, `np.int64` and `np.bool_` are not subclasses and fail. Don't rely on luck: convert.
- `float(np.array([1.0]))` raises a `TypeError` in recent NumPy 2 (only 0-d arrays convert
  this way). Use `.item()` for size-1 arrays.
- 0-d arrays are mutable (`z[()] = 5`), NumPy scalars are not.
- `type(a[0]) is float` is `False`. If code checks `isinstance(v, int)`, `np.int64` fails it.
- PyTorch has the same split: `t[0]` is a 0-d tensor and `t[0].item()` is the Python number.

## Builds on

001, 002.

## Your task

Implement in `problem.py`:

1. `to_jsonable(obj)` - recursively convert a structure of dicts, lists, tuples, NumPy arrays and
   NumPy scalars into built-in Python types (arrays -> nested lists, tuples -> lists), so that
   `json.dumps` works. Everything else is returned unchanged.
2. `as_python_number(x)` - a Python scalar from a NumPy scalar, a Python number, or any array with
   exactly one element. Raise `ValueError` if the array has more or fewer than one element.
3. `array_to_json(a)` - a JSON string that stores `a`'s dtype, shape and values.
4. `array_from_json(s)` - the exact inverse: same dtype, same shape, same values.

## Hints

<details>
<summary>Hint 1</summary>

In `to_jsonable`, check types in order: `dict`, `list`/`tuple`, `np.ndarray`, `np.generic`
(the base class of every NumPy scalar). `.tolist()` and `.item()` do the real work.

</details>

<details>
<summary>Hint 2</summary>

For the JSON pair, store `str(a.dtype)`, `list(a.shape)` and `a.ravel().tolist()`. On the way
back, `np.array(data, dtype=...)` then `.reshape(shape)`. Flattening the data makes empty and
0-d arrays work too.

</details>

## Self-check

1. What is `type(np.arange(3)[0])`?
2. Why does `json.dumps(np.float32(1.0))` fail but `json.dumps(np.float64(1.0))` work?
3. What is the shape of `np.array(5)`, and how do you get the `5` out?
4. Why does a JSON round trip of an array need the dtype and the shape?

<details>
<summary>Answers</summary>

1. `numpy.int64`, a NumPy scalar, not a Python `int`.
2. `np.float64` subclasses Python `float`, which `json` knows; `np.float32` doesn't.
3. `()`. Use `.item()` (or `z[()]` to get a NumPy scalar).
4. JSON numbers have no dtype (uint8 would come back as int64), and an empty array's data `[]`
   says nothing about its shape.

</details>

## Go deeper

- [Scalars](https://numpy.org/doc/stable/reference/arrays.scalars.html)
