# 026 · Generator API and seeding ★

**Category:** Random and statistics · **Difficulty:** Easy

## What it is

Modern NumPy random numbers come from a **Generator** object: `rng = np.random.default_rng(seed)`.
You call methods on it: `rng.random()`, `rng.normal()`, `rng.integers()`, `rng.choice()`.
The old style, `np.random.seed(0)` plus `np.random.rand(...)`, uses one hidden **global** state
and is now called "legacy".

## Why it exists

Research code must be **reproducible**: same seed, same numbers, same result. The global state
makes that fragile: any library that calls `np.random.*` changes the numbers your code gets.
A Generator is a plain object you create and pass around, so each part of your program has
its own stream, and nothing else can touch it.

## How it works

```python
rng = np.random.default_rng(42)
rng.random(3)                 # 3 floats in [0, 1)
rng.normal(0, 1, size=(2, 2)) # mean 0, std 1
rng.integers(0, 10, size=5)   # ints in [0, 10)   (high is excluded)

np.random.default_rng(42).random(3)    # the same 3 numbers again
```

A Generator is a **stream**: each call moves it forward.

```
rng = default_rng(42)
rng.random()  -> 0.773...    rng.random() -> 0.438...    rng.random() -> 0.858...
                 ^ draw 1                    ^ draw 2                    ^ draw 3
```

So to reproduce a result you need the same seed **and** the same sequence of calls.

**Pass the generator in**, don't create it inside helpers:

```python
def augment(x, rng):          # caller decides the seed
    return x + rng.normal(0, 0.1, size=x.shape)
```

**Independent streams.** For several workers (or several experiments), don't use seeds
`0, 1, 2, ...` by hand. Spawn children from one seed; they're guaranteed not to overlap:

```python
children = np.random.SeedSequence(42).spawn(3)
streams = [np.random.default_rng(s) for s in children]
# or: np.random.default_rng(42).spawn(3)
```

```
SeedSequence(42)
 ├── child 0 -> Generator for worker 0
 ├── child 1 -> Generator for worker 1
 └── child 2 -> Generator for worker 2
```

**Legacy vs new:**

| legacy (global) | Generator |
|---|---|
| `np.random.seed(0)` | `rng = np.random.default_rng(0)` |
| `np.random.rand(3)` | `rng.random(3)` |
| `np.random.randn(3)` | `rng.standard_normal(3)` |
| `np.random.randint(0, 10)` | `rng.integers(0, 10)` |

The same seed gives **different numbers** in the two systems (different algorithms).

## Gotchas

- `default_rng()` with no seed is seeded from the OS: different every run. Fine for production
  randomness, wrong for experiments.
- Re-creating `default_rng(seed)` inside a loop gives the **same** numbers every iteration.
- `np.random.seed` does not affect Generators, and `default_rng` does not touch the global state.
- PyTorch has its own generator: `torch.manual_seed` does not seed NumPy, and NumPy seeds don't
  seed PyTorch. DataLoader workers need their own NumPy seeds too.
- `rng.integers(low, high)` excludes `high`; pass `endpoint=True` to include it.

## Builds on

001, 007.

## Your task

Implement in `problem.py`. Follow the exact recipe in each docstring (the tests compare against
the same calls), and never use the legacy global functions:

1. `noisy_signal(n, noise_std, seed)` - `sin` on `n` points of `[0, 2π]` plus Gaussian noise.
2. `dropout_mask(shape, p, seed)` - a boolean keep-mask: `rng.random(shape) >= p`.
3. `spawn_streams(seed, n)` - `n` independent Generators spawned from `seed`.
4. `bootstrap_means(x, n_boot, rng)` - means of `n_boot` resamples (with replacement) of `x`,
   using the Generator you're given. No Python loops.

## Hints

<details>
<summary>Hint 1</summary>

Create the Generator once at the top of the function: `rng = np.random.default_rng(seed)`.

</details>

<details>
<summary>Hint 2</summary>

`bootstrap_means`: draw all indices at once with
`rng.integers(0, len(x), size=(n_boot, len(x)))`, then `x[idx].mean(axis=1)`.

</details>

## Self-check

1. Why is `np.random.seed(0)` fragile in a big program?
2. You call `rng.random()` twice on one Generator. Are the two numbers equal?
3. How do you get 8 non-overlapping streams for 8 worker processes?
4. Does `torch.manual_seed(0)` make `np.random.default_rng()` reproducible?

<details>
<summary>Answers</summary>

1. Any code, including libraries, that draws from the global state shifts the numbers your code
   sees afterwards.
2. No; the stream moves forward on every call.
3. `np.random.SeedSequence(seed).spawn(8)` (or `rng.spawn(8)`), one Generator per child.
4. No; PyTorch and NumPy have separate random states.

</details>

## Go deeper

- [Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
