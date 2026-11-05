# 035 · Float precision: isclose, cancellation, logsumexp

**Category:** Linear algebra and numerics · **Difficulty:** Medium

## What it is

Floats store about 16 significant decimal digits (float64) or 7 (float32). Most results are
slightly rounded, so you compare floats with a **tolerance** (`np.isclose`, `np.allclose`), not
`==`. Some formulas lose far more precision than that, and a few classic rewrites keep them
accurate: subtracting the max (logsumexp), `log1p`, and avoiding the subtraction of nearly equal
numbers.

## Why it exists

`0.1 + 0.2 == 0.3` is `False`. Tests that compare floats with `==` break for no visible reason.
Worse, some computations quietly return 0, inf or garbage: probabilities that underflow, a
quadratic root that comes out as exactly 0. Knowing the few standard tricks prevents most of these.

## How it works

**Tolerances.** `np.isclose(a, b, rtol=1e-5, atol=1e-8)` checks
`|a - b| <= atol + rtol * |b|`:

```
rtol: "same to about 5 significant digits"   (relative to b, so it's not symmetric)
atol: needed near zero, where any relative tolerance becomes tiny
```

`np.isclose(np.nan, np.nan)` is False unless you pass `equal_nan=True`.

**Overflow and underflow: logsumexp.** Probabilities are often kept as logs. Adding them needs
`log(sum(exp(x)))`, but `exp(1000)` is inf and `exp(-1000)` is 0. Pull the max out first:

```
log(sum(exp(x))) = m + log(sum(exp(x - m)))      with m = max(x)

x = [1000, 1000]:  naive -> inf          stable -> 1000 + log(2) = 1000.693
x = [-1000, -1000]: naive -> -inf        stable -> -999.307
```

After the shift the largest term is `exp(0) = 1`, so nothing overflows and the sum is at least 1.
A slice that is all `-inf` (all probabilities 0) must give `-inf`, which needs a little care
because `-inf - (-inf)` is NaN.

**log1p.** For tiny `t`, `1 + t` rounds to exactly `1`, so `log(1 + t)` returns 0.
`np.log1p(t)` computes it without forming `1 + t`:

```
t = exp(-40) = 4.25e-18
np.log(1 + t)  -> 0.0          (all information lost)
np.log1p(t)    -> 4.25e-18     (correct)
```

**softplus** `log(1 + exp(x))` combines both problems: `exp(x)` overflows for large `x`, and
the result vanishes for very negative `x`. A stable form is
`max(x, 0) + log1p(exp(-|x|))`, which is what `np.logaddexp(0, x)` computes.

**Catastrophic cancellation.** Subtracting two nearly equal numbers keeps only their few
**differing** digits. The textbook quadratic formula for `x² - (1e8 + 1e-9)x + 0.1 = 0`
(roots `1e8` and `1e-9`):

```
(-b - sqrt(b² - 4ac)) / 2a  ->  (1e8 - 1e8) / 2 = 0.0     100% error for the small root
```

The fix: compute the root **without** subtraction first, then get the other one from the
product of the roots, `r1 · r2 = c / a`:

```
q  = -(b + sign(b) · sqrt(b² - 4ac)) / 2      (b and sign(b)·sqrt have the same sign: no cancellation)
r1 = q / a,   r2 = c / q
```

The same issue hits `var = mean(x²) - mean(x)²` for data far from zero: use the two-pass
`mean((x - mean(x))²)`, which is what `np.var` does.

## Gotchas

- Never `==` floats that came out of arithmetic. Use `np.isclose`/`np.allclose`, and pass `atol`
  when the expected values can be 0.
- `np.isclose(a, b)` is not symmetric: the tolerance scales with `|b|`.
- float32 has ~7 digits: `np.float32(16777216) + 1` is still `16777216`.
- `np.sign(0)` is 0, which breaks `sign(b)` tricks when `b == 0`. Use `np.where(b >= 0, 1, -1)`.
- PyTorch has `torch.logsumexp`, `torch.log1p`, `torch.nn.functional.softplus` and
  `torch.isclose`/`torch.allclose` with the same meaning.

## Builds on

002, 018, 030.

## Your task

Implement in `problem.py`, with **no Python loops** and no warnings:

1. `logsumexp(x, axis)` - stable `log(sum(exp(x)))` along `axis`; a slice of all `-inf` gives `-inf`.
2. `softplus(x)` - stable `log(1 + exp(x))`, accurate for `x = 1000` and for `x = -40`.
3. `quadratic_roots(a, b, c)` - both real roots of `a·x² + b·x + c = 0` (arrays broadcast),
   accurate even when one root is tiny; return `(smaller, larger)`.
4. `count_mismatches(a, b, rtol, atol)` - how many elements are **not** close (NaN equals NaN),
   as a Python int.

## Hints

<details>
<summary>Hint 1</summary>

`logsumexp`: `m = x.max(axis=axis, keepdims=True)`, replace non-finite `m` by 0, compute
`log(sum(exp(x - m)))` inside `np.errstate(divide="ignore")`, add `m` back and squeeze the axis.

</details>

<details>
<summary>Hint 2</summary>

`softplus`: `np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))`.

</details>

<details>
<summary>Hint 3</summary>

`quadratic_roots`: `s = np.where(b >= 0, 1.0, -1.0)`, `q = -0.5 * (b + s * np.sqrt(b*b - 4*a*c))`,
roots `q / a` and `c / q`, then `np.minimum` / `np.maximum`.

</details>

## Self-check

1. Why does subtracting the max make logsumexp safe from overflow?
2. Why does `np.log(1 + 1e-18)` return 0?
3. What makes `(-b - sqrt(b² - 4ac)) / 2a` inaccurate for one of the roots?
4. Why is `np.isclose(1e-10, 0)` True with the default tolerances?

<details>
<summary>Answers</summary>

1. Every shifted exponent is `<= 0`, so each `exp` is at most 1; the largest term is exactly 1.
2. `1 + 1e-18` rounds to `1.0` in float64 (about 16 digits), and `log(1) = 0`.
3. When `b² ≫ 4ac`, `sqrt(b² - 4ac) ≈ |b|`, and subtracting two nearly equal numbers cancels
   almost all the digits.
4. The default `atol=1e-8` is larger than the difference `1e-10`.

</details>

## Go deeper

- [numpy.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
