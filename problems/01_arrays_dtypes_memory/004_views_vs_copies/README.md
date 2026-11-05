# 004 · Views vs copies ★

**Category:** Arrays, dtypes, memory · **Difficulty:** Medium

## What it is

A **view** is a new array object that looks at the **same memory** as another array. A **copy**
has its own memory. Writing through a view changes the original; writing to a copy doesn't.
Many NumPy operations return views, and knowing which ones is the key to avoiding silent bugs.

## Why it exists

Views make slicing, transposing and reshaping free: no data is copied, only a new "window"
(shape, strides, offset) is made. On a 1 GB array that's the difference between instant and
slow-and-out-of-memory. The price: two names can now change the same numbers.

## How it works

```python
a = np.arange(6)          # owns its memory
v = a[1:4]                # view: same memory, different window
c = a[1:4].copy()         # copy: new memory

v[0] = 99
a                         # array([ 0, 99,  2,  3,  4,  5])   <- changed through v
c                         # array([1, 2, 3])                  <- not affected
```

```
memory of a:   [ 0 | 99 |  2 |  3 |  4 |  5 ]
v (a[1:4]):         ^^^^^^^^^^^^^^
c:             [ 1 | 2 | 3 ]   (somewhere else)
```

How to tell:

- `v.base is a`: a view remembers the array it came from (`a.base` is `None` for an owner).
- `np.shares_memory(x, y)`: do any elements actually overlap? This is the reliable test.

| returns a **view** | returns a **copy** |
|---|---|
| basic slicing `a[1:4]`, `a[:, 0]`, `a[::2]` | fancy indexing `a[[0, 2]]` |
| `a.T`, `a.reshape(...)` (usually), `a.view()` | boolean masks `a[a > 0]` |
| `np.asarray(a)` (returns `a` itself) | `a.copy()`, `np.array(a)`, `a.astype(...)` |

**In-place vs rebinding.** Inside a function, `a -= m` writes into the caller's memory.
`a = a - m` makes a new array and only rebinds the local name; the caller sees nothing.

**The chained-indexing trap.** Fancy and boolean indexing return copies (more in 007 and 008),
so a second index on top writes into a temporary that is thrown away:

```python
rows = [0, 2]
a[rows][a[rows] < 0] = 0     # does NOTHING to a: a[rows] is a copy
a[rows] = np.where(a[rows] < 0, 0, a[rows])   # works: one assignment into a
```

A single `a[...] = value` is always an in-place write. The trap is only when you index twice.

## Gotchas

- `a[::2]` and `a[1::2]` use the same buffer but never touch the same element:
  `np.shares_memory` is `False` for them (`np.may_share_memory` says `True`; it only checks bounds).
- `reshape` returns a copy when the memory layout makes a view impossible (e.g. after `.T`).
  Don't write through a reshape and assume the original changed.
- PyTorch is similar: slicing gives views, `.clone()` is NumPy's `.copy()`. But `torch.reshape`
  vs `view` makes the view/copy choice more explicit than NumPy does.
- A small view keeps the **whole** base array alive. `big[:10].copy()` lets the big array be freed.

## Builds on

001. Uses one idea each from 007 (fancy indexing makes a copy) and 008 (boolean masks).

## Your task

Implement in `problem.py`:

1. `memory_relation(a, b)` - return `"same"` if `a` and `b` are the same object, `"shared"` if
   they overlap in memory, else `"independent"`.
2. `first_column_copy(a)` - column 0 of a 2-D array as a 1-D array that is safe to modify.
3. `standardize_inplace(a)` - subtract the mean and divide by the (population) std, **in place**,
   return `None`. Must work when `a` is a view. Constant input -> all zeros. Non-float -> `TypeError`.
4. `zero_small_in_rows(a, rows, threshold)` - in place: in the listed rows only, set values below
   `threshold` to 0. Return `None`.

## Hints

<details>
<summary>Hint 1</summary>

`np.shares_memory(a, b)` answers "shared". Check `a is b` first.

</details>

<details>
<summary>Hint 2</summary>

In `standardize_inplace`, use only in-place operations on `a` (`a -= ...`, `a /= ...`, or
`a[...] = ...`). Any plain `a = ...` disconnects you from the caller's memory.

</details>

<details>
<summary>Hint 3</summary>

For `zero_small_in_rows`, read the rows into a temporary, fix it, then write it back with one
assignment: `a[rows] = fixed`.

</details>

## Self-check

1. Does `a[:, 0]` return a view or a copy? And `a[:, [0]]`?
2. Why does `a[[0, 2]][0] = 5` leave `a` unchanged?
3. Inside a function, what is the difference between `x -= 1` and `x = x - 1` for the caller?
4. Why can a tiny slice keep a huge array in memory?

<details>
<summary>Answers</summary>

1. `a[:, 0]` is a view (basic indexing). `a[:, [0]]` is a copy (a list is fancy indexing).
2. `a[[0, 2]]` makes a new array; the assignment changes that temporary, which is then discarded.
3. `x -= 1` changes the caller's array in place. `x = x - 1` builds a new array and only the
   local name points to it.
4. The view's `.base` holds a reference to the whole original buffer.

</details>

## Go deeper

- [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html)
