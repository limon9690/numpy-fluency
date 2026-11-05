# numpy-fluency

A NeetCode-style problem sheet for learning NumPy itself, well enough to use it confidently in
data processing, research code, images, and next to PyTorch.
Every problem = short theory card + stub to fill in + automatic tests + hidden solution.

## Who this is for

- The learner knows Python and is working through a separate PyTorch sheet, so tensor basics
  (broadcasting, reshaping, einsum) are familiar. Here the focus is **what is specific to NumPy**:
  dtypes and memory, copy semantics, NumPy-only indexing rules, the vectorization toolbox,
  `Generator`, `linalg`, images as arrays, I/O, and the NumPy ↔ PyTorch boundary.
- No classic ML from scratch (no k-means, regression models, etc.). The goal is NumPy fluency.
- Learner prefers plain, simple English, concept-first explanations ("why does this exist?"),
  worked examples, and visuals. When showing arrays in explanations, write them as readable
  2D grids, never inline `[[...]]`.

## Environment

- Python 3.11+, **NumPy 2.x** (type promotion changed in 2.0; cards and tests assume 2.x), pytest.
- PyTorch is **optional**. Only 039 uses it: its tests start with `pytest.importorskip("torch")`.
- Tests never use the network. File I/O (038) writes only into pytest's `tmp_path`.
  Images (036) are synthetic arrays; don't depend on PIL or OpenCV.

## Repository layout

```
numpy-fluency/
  CLAUDE.md          # this file (design + conventions)
  README.md          # how to use the repo (already written)
  TRACKER.md         # the sheet; all 40 rows already exist
  requirements.txt
  pytest.ini
  conftest.py        # `sol`, `ban` and `no_loops` fixtures (already written)
  scripts/
    check_batch.py   # automatic review of problems
    reset.sh         # restore stubs for a fresh start
  problems/
    01_arrays_dtypes_memory/
      001_creating_arrays/
        README.md        # theory card
        problem.py       # stub
        test_problem.py  # tests
  solutions/
    01_arrays_dtypes_memory/
      001_creating_arrays.py
```

- **Folder names are fixed**: use exactly the paths linked in TRACKER.md.
- Category folders: `01_arrays_dtypes_memory`, `02_indexing`, `03_broadcasting_shape_ops`,
  `04_reductions_vectorized_ops`, `05_random_statistics`, `06_linear_algebra_numerics`,
  `07_arrays_in_practice`.
- Tests reach code only through the `sol` fixture, which loads `problem.py`, or the matching
  file in `solutions/` when `SOLUTIONS=1`.

## Problem folder contents

### README.md (theory card), in this order
1. **What it is** - 2-4 sentences.
2. **Why it exists** - the problem it solves; when you'd reach for it in real work.
3. **How it works** - a small worked example with real numbers and shapes. Add a figure where it
   helps (ASCII or Mermaid): memory layout, strides, broadcasting grids, axis directions.
4. **Gotchas** - mistakes people actually make. Where NumPy differs from PyTorch
   (`axis` vs `dim`, copy semantics, dtype defaults, in-place behaviour), say so here.
5. **Builds on** - problem numbers this one uses.
6. **Your task** - exactly what to implement, shapes and dtypes, banned shortcuts.
7. **Hints** - 2-3 progressive hints, each in its own `<details>` block.
8. **Self-check** - 3-4 short concept questions, answers in one `<details>` block.
9. **Go deeper** (optional) - 1-2 links to the official NumPy docs.

Card title format: `# NNN · Title` with ` ★` at the end for Core, and a line
`**Category:** ... · **Difficulty:** ...` right under it.
Keep cards short: readable in about 5 minutes. No marketing tone, no filler.

### problem.py
- Functions with full signatures, type hints (`np.ndarray`) and a docstring with the spec,
  shapes, dtypes and a tiny example. Body: `raise NotImplementedError`.
- The docstring alone must be enough to re-solve the problem from a blank file.
- 1-4 functions per problem; 20-60 minutes of work. A problem may go a little beyond its title
  to reach that size, as long as it stays on topic.
- If a card needs a one-line idea from a later problem, say so in "Builds on"; don't renumber.

### test_problem.py
- Fast (whole file < 5 seconds), deterministic (fixed seeds; use `np.random.default_rng(0)`).
- Compare against NumPy's own function whenever one exists (`np.testing.assert_allclose`,
  `assert_array_equal`). Check **dtype and shape**, not just values; dtypes matter a lot in NumPy.
- Test behaviour from the spec, not one particular implementation. Any correct solution must pass.
- Where it matters, test memory semantics: `np.shares_memory`, whether the input was modified,
  `flags.writeable`, `flags.c_contiguous`.
- Failure messages say what went wrong in plain words.
- Banned shortcuts: `ban(np, "unique")` works for module functions. ndarray methods can't be
  monkeypatched, so don't try to ban those.
- "No Python loops" rules: use `no_loops(sol.func_name)`. It fails if the function body has
  `for`/`while` loops or comprehensions.
- Never assert on wall-clock timing (it's flaky). 040 tests behaviour and memory, not speed.

### Solutions
- Clean, idiomatic NumPy, the way a strong engineer would write it.
- Short comments on the non-obvious lines explaining *why*, not *what*.

## Difficulty and Core marking

- Difficulty: Easy / Medium / Hard. ★ = **Core**: must be re-solvable from a blank file (12 in total).

## Rules for every build session

- **Never modify an existing `problem.py`**: the learner's work lives there.
- **Don't edit shared files**: README.md, CLAUDE.md, TRACKER.md, conftest.py, pytest.ini,
  requirements.txt, scripts/. TRACKER.md already has every row. Batches may run in parallel.
  Put any helper a problem needs inside its own test file.
- Don't change the curriculum, numbering or folder names. If something seems wrong, say so in
  the PR description.
- Follow "Git style" below exactly.
- Work unattended: don't stop to ask questions. Make the most reasonable choice, and list every
  such choice in the PR description.
- Don't modify `scripts/check_batch.py` to make a problem pass; fix the problem instead.
  If a check is genuinely wrong, say so in the PR description.
- At the end: run `python scripts/check_batch.py NNN-NNN` on your batch's range. Fix every ERROR;
  fix warnings or explain them. The checker runs the `SOLUTIONS=1` tests and checks that stubs
  fail cleanly. Then push your branch (see "Git style"). **Don't open a pull request.**

## Git style

Commits should read like the learner wrote them by hand.

- Before your first commit, in this repo:
  `git config user.name "Iqbal Hossain Limon"` and
  `git config user.email "70581375+limon9690@users.noreply.github.com"`.
- Branch name: `batch-1` or `batch-2`. If the environment forces another name, that's fine.
- Commit messages: one short line, lowercase, imperative, no period, under 50 characters.
  Good: `add problems 001-005`, `add indexing problems`, `fix dtype test in 002`.
- No message body, no bullet lists, no emoji, no trailers (`Co-Authored-By`, session links),
  no "Generated with" lines, no mention of AI or tools.
- A few commits per batch (e.g. one per category). Not dozens.
- **Don't open a pull request** (the learner opens it himself). End your final message with:
  1. a PR title in the same style as commits, e.g. `add problems 001-017`;
  2. a PR description of 2-3 plain sentences: what was added and the checker's summary line;
  3. separately, for the learner only: the list of choices you made and anything to look at.

## Out of scope (don't add)

Classic ML algorithms, pandas, SciPy, structured and masked arrays, datetime64, string arrays,
polynomials, FFT, `np.matrix`, Numba.

## Curriculum (40 problems)

### 01 Arrays, dtypes, memory
001 Creating arrays (array, zeros/ones/full, arange vs linspace float pitfalls, *_like) - Easy
002 dtypes, casting and overflow (uint8 wraparound, in-place cast errors, round-half-even, NumPy 2 promotion) - Easy
003 NumPy scalars and Python interop (.item, .tolist, 0-d arrays, JSON) - Easy
004 Views vs copies (base, shares_memory, the mutation trap) ★ - Medium
005 Strides and memory layout (C vs F order, contiguity, read-only broadcast_to) - Medium

### 02 Indexing
006 Slicing and basic indexing - Easy
007 Fancy (integer-array) indexing ★ - Medium
008 Boolean masking ★ - Medium
009 where, nonzero, argwhere - Easy
010 Mixing basic and advanced indexing, np.ix_ - Hard
011 take_along_axis and put_along_axis - Medium

### 03 Broadcasting and shape ops
012 Broadcasting rules (outer sum, pairwise differences) ★ - Medium
013 Row and column scaling and normalization ★ - Easy
014 reshape, ravel vs flatten, transpose, moveaxis - Easy
015 newaxis, expand_dims, squeeze - Easy
016 concatenate, stack, split - Easy
017 tile, repeat, pad, meshgrid - Medium

### 04 Reductions and vectorized ops
018 Reductions with axis and keepdims ★ - Easy
019 ufuncs: out=, where=, reduce, accumulate, outer - Medium
020 cumsum, diff and running statistics - Easy
021 sort, stable argsort, lexsort, partition ★ - Medium
022 unique (counts, inverse) and bincount ★ - Medium
023 searchsorted, digitize, histogram, interp - Medium
024 Set operations: isin, intersect1d, setdiff1d, union1d - Easy
025 np.add.at and scatter-style updates - Hard

### 05 Random and statistics
026 Generator API and seeding (default_rng, legacy np.random, spawning streams) ★ - Easy
027 Sampling: choice, permutation, shuffle - Easy
028 Percentiles, clip and winsorize - Medium
029 var and std with ddof, cov, corrcoef - Medium
030 NaN and inf: isfinite, nan* functions, errstate - Medium

### 06 Linear algebra and numerics
031 matmul, dot, @ and einsum ★ - Medium
032 solve vs inv, condition number - Medium
033 Least squares with lstsq - Medium
034 Norms, SVD and eigh (batched) - Hard
035 Float precision: isclose, cancellation, logsumexp - Medium

### 07 Arrays in practice
036 Images as arrays: HWC uint8 vs float, channels, grayscale ★ - Medium
037 Sliding windows and framing (sliding_window_view) - Medium
038 Saving and loading: .npy, .npz, memmap, loadtxt - Easy
039 NumPy and PyTorch bridge (shared memory, dtype, layout) ★ - Easy
040 Performance: vectorized vs loops, fake vectorization, memory blow-ups - Medium

## Build plan

Two sessions; they can run at the same time.

- Batch 1: categories 01-03 (001-017)
- Batch 2: categories 04-07 (018-040)
