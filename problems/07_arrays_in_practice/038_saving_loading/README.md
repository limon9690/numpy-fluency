# 038 · Saving and loading: .npy, .npz, memmap, loadtxt

**Category:** Arrays in practice · **Difficulty:** Easy

## What it is

- `np.save` / `np.load` - one array in a `.npy` file, with its dtype and shape.
- `np.savez` / `np.savez_compressed` - several named arrays in one `.npz` file (a zip).
- `np.memmap` - treat a file on disk as an array, reading only the parts you touch.
- `np.loadtxt` - read numbers from a text file (CSV and similar).

## Why it exists

You save intermediate results, features and model outputs to avoid recomputing them. `.npy`
is exact and fast: no rounding to text, and dtype and shape come back automatically. When data
is bigger than RAM, a memory map lets you work on slices of it. Text files are for data that
comes from (or goes to) other tools.

## How it works

**One array:**

```python
np.save(path, a)            # writes path (adds ".npy" if missing)
b = np.load(path)           # same dtype, same shape, same values
```

**Several arrays:**

```python
np.savez_compressed("data.npz", images=X, labels=y)
with np.load("data.npz") as f:          # f is a lazy NpzFile, like a dict
    print(f.files)                      # ['images', 'labels']
    X2 = f["images"]                    # read from the zip when you ask for it
```

The `NpzFile` keeps the file open. Use `with`, and copy what you need into a normal dict
(`{k: f[k] for k in f.files}`) before the block ends.

**Memory maps.** A raw binary file of float32 values in C order with shape `(1000000, 64)` is
256 MB. A memmap reads only the pages you index:

```python
m = np.memmap(path, dtype=np.float32, mode="r", shape=(1_000_000, 64))
col_mean = m[:, 3].mean()      # touches column 3 only (still strided reads, but no 256 MB copy)
```

```
file on disk:   [ row 0 (64 floats) | row 1 | row 2 | ... ]
m[:, 3]          ^ 4th value of every row, read on demand
```

Modes: `"r"` read-only, `"r+"` read and write in place, `"w+"` create a new file,
`"c"` copy-on-write. A raw file has **no header**: you must know dtype and shape yourself.
`np.load(path, mmap_mode="r")` memory-maps a `.npy` file and reads its header for you.

**Text files:**

```
x,y,label            <- header line
1.5,2.0,0
3.0,4.5,1
```

```python
data = np.loadtxt(path, delimiter=",", skiprows=1)        # float64, shape (2, 3)
```

With a single data row, `loadtxt` returns a **1-D** array of shape `(3,)`. Pass `ndmin=2` to
always get a 2-D table.

## Gotchas

- **`allow_pickle`.** Object arrays (lists of Python objects) are saved with pickle. `np.load`
  refuses them by default (`allow_pickle=False`), because loading a pickle can run arbitrary
  code. Only allow it for files you created yourself.
- `np.save("x", a)` writes `x.npy`, and `np.savez("x", ...)` writes `x.npz`; the extension is
  added for you.
- Text loses precision unless you write enough digits, and is much slower and bigger than `.npy`.
- Writing through a memmap in `"r+"` mode changes the file. Call `m.flush()` to be sure it's
  written.
- PyTorch's `torch.save` uses pickle; `.npy` files are a safer exchange format between the two
  (load with `np.load`, then `torch.from_numpy`, 039).

## Builds on

001, 002, 005.

## Your task

Implement in `problem.py` (tests write only into a temporary folder):

1. `save_arrays(path, arrays)` - save a dict of name -> array into one compressed `.npz` file.
2. `load_arrays(path)` - load an `.npz` file into a plain `dict` of name -> ndarray, and close
   the file.
3. `memmap_column_mean(path, shape, dtype, col)` - the float64 mean of column `col` of a raw
   binary file holding a C-ordered array of `shape` and `dtype`, opened read-only as a memmap.
4. `load_csv(path)` - a comma-separated file with one header line -> `(names, data)`: the list of
   column names and a 2-D float64 array (2-D even for a single data row).

## Hints

<details>
<summary>Hint 1</summary>

`save_arrays`: `np.savez_compressed(path, **arrays)`. `load_arrays`:
`with np.load(path) as f: return {k: f[k] for k in f.files}`.

</details>

<details>
<summary>Hint 2</summary>

`memmap_column_mean`: `np.memmap(path, dtype=dtype, mode="r", shape=shape)`, then
`m[:, col].mean(dtype=np.float64)`.

</details>

<details>
<summary>Hint 3</summary>

`load_csv`: read the first line yourself for the names (`open(path).readline().strip().split(",")`),
then `np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)`.

</details>

## Self-check

1. What does a `.npy` file store besides the values?
2. Why does `np.load` refuse object arrays by default?
3. What do you need to know to open a raw binary file as a memmap?
4. What does `np.loadtxt` return for a file with only one data row, and how do you fix it?

<details>
<summary>Answers</summary>

1. The dtype, the shape and the memory order (a small header).
2. They're stored with pickle, and unpickling can run arbitrary code.
3. The dtype, the shape and the order (and an offset if there's a header).
4. A 1-D array; pass `ndmin=2`.

</details>

## Go deeper

- [Input and output](https://numpy.org/doc/stable/reference/routines.io.html)
