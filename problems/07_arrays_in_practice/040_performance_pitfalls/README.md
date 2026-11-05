# 040 · Performance: vectorized vs loops, fake vectorization, memory blow-ups

**Category:** Arrays in practice · **Difficulty:** Medium

## What it is

NumPy is fast when the looping happens **in C**, inside one call over a whole array. It's slow
when Python touches every element. Memory is the other half: every intermediate result in an
expression is a full new array, so a one-line formula can use many times the memory of its input.

## Why it exists

The same computation can take 1 second or 100, and 50 MB or 50 GB, depending only on how you
write it. You'll meet three traps over and over:

1. **Python loops** over elements (obvious) and **fake vectorization** (not obvious).
2. **Temporaries**: big intermediate arrays you never named.
3. **Wrong dtype or shape**: float64 copies of uint8 data, broadcasts that explode.

## How it works

**Fake vectorization.** `np.vectorize(f)`, `np.frompyfunc` and `np.apply_along_axis` *look*
vectorized but call your Python function once per element (or row). They're convenience
wrappers around a loop, with loop speed. Real vectorization means expressing the work with
ufuncs, reductions, indexing and searching (`np.searchsorted`, 023).

Example: map keys to values with a lookup table.

```
loop / np.vectorize:   for every key, search the table in Python        -> slow
vectorized:            sort the table once, np.searchsorted all keys     -> one C call
```

**Temporaries.** `y = (x - x.mean()) / x.std()` on a 400 MB float32 array allocates
`x - x.mean()` (400 MB), the result (400 MB), and `x.std()` internally makes another full
temporary. In-place operations reuse memory:

```python
x -= x.mean()                  # no new array
x /= np.sqrt(np.dot(flat, flat) / x.size)   # flat = x.ravel(): a view for contiguous x
```

**dtype blow-ups.** `imgs.astype(np.float64)` on uint8 images is **8×** the memory. Many
reductions take `dtype=` and accumulate in a wide type **without** converting the whole array:
`imgs.mean(axis=(0, 1, 2), dtype=np.float64)`.

**Broadcast blow-ups.** Pairwise squared distances via `X[:, None, :] - Y[None, :, :]` build an
`(n, m, d)` array (012). For `n = m = 10 000`, `d = 512` that's 400 GB. Expand the square instead:

```
‖x - y‖² = ‖x‖² + ‖y‖² - 2 x·y

D = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T      only (n, m) arrays
```

Rounding can make tiny distances slightly negative; clip at 0.

**Measuring.** Time with `%timeit` (IPython) or `time.perf_counter`, memory with `tracemalloc`
(NumPy reports its allocations to it). Don't trust intuition: measure before and after.

## Gotchas

- `np.vectorize` is not vectorization. Neither is `apply_along_axis`.
- A list comprehension that builds an array (`np.array([f(v) for v in x])`) is a Python loop.
- `x = x - m` allocates; `x -= m` doesn't (but it changes the caller's array, 004).
- `ravel()`, slicing and `reshape` are free only when they return views (004, 014).
- The fast distance formula trades a little accuracy for memory: fine for most uses, but compare
  tiny distances with care (035).
- PyTorch has the same issues, plus GPU memory is smaller; `torch.cdist` exists for distances.

## Builds on

004, 012, 018, 023, 031.

## Your task

Implement in `problem.py`, with **no Python loops**. The tests measure peak memory with
`tracemalloc`; they never time anything.

1. `pairwise_sq_dist_fast(X, Y)` - `(n, d)` and `(m, d)` -> `(n, m)` squared distances, float64,
   clipped at 0, **without** any `(n, m, d)` intermediate.
2. `standardize_inplace_f32(x)` - standardize a C-contiguous float32 array in place (mean 0,
   population std 1), using almost no extra memory. Return `None`.
3. `lookup(keys, table_keys, table_values, default)` - the value for each key (`default` if
   missing). `np.vectorize`, `np.frompyfunc` and `np.apply_along_axis` are banned.
4. `channel_means_uint8(imgs)` - per-channel means of a uint8 `(N, H, W, C)` batch as float64
   `(C,)`, without making a float copy of the batch.

## Hints

<details>
<summary>Hint 1</summary>

`pairwise_sq_dist_fast`: the formula in the card, then `np.maximum(D, 0)`.

</details>

<details>
<summary>Hint 2</summary>

`standardize_inplace_f32`: `x -= x.mean()`, then `flat = x.ravel()` (a view) and
`var = np.dot(flat, flat) / x.size`, then `x /= np.sqrt(var)`. `x.std()` would allocate a full
temporary.

</details>

<details>
<summary>Hint 3</summary>

`lookup`: `order = np.argsort(table_keys)`, `pos = np.searchsorted(table_keys, keys, sorter=order)`,
clip `pos`, check which keys really match, then `np.where`.

</details>

## Self-check

1. Why is `np.vectorize` slow?
2. How much memory does `X[:, None] - Y[None]` need for `n = m = 1000`, `d = 100`, float64?
3. Why can `imgs.mean(dtype=np.float64)` use far less memory than `imgs.astype(np.float64).mean()`?
4. What's the memory difference between `x = x * 2` and `x *= 2`?

<details>
<summary>Answers</summary>

1. It calls the Python function once per element; it's a loop with a NumPy interface.
2. `1000 · 1000 · 100 · 8` bytes = 800 MB.
3. The reduction converts small chunks as it goes; `astype` builds a full float64 copy (8× uint8) first.
4. `x * 2` allocates a new array of the same size; `x *= 2` writes into the existing one.

</details>

## Go deeper

- [numpy.vectorize](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html) (see its note: "essentially a for loop")
