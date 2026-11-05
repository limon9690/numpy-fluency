# numpy-fluency

A problem sheet for learning NumPy properly, in the style of NeetCode.
40 small problems on what makes NumPy NumPy: dtypes and memory, copy semantics, indexing rules,
the vectorization toolbox, random numbers, linear algebra, images as arrays, I/O, and the
boundary with PyTorch.

Every problem has:

- a short **theory card** (`README.md`): what it is, why it exists, a worked example,
  gotchas, your task, hints, and self-check questions,
- a **stub** (`problem.py`) that you fill in,
- **automatic tests** (`test_problem.py`) that tell you when you're done,
- a hidden **reference solution** in `solutions/` (look only after you've tried).

The full curriculum is in [`CLAUDE.md`](CLAUDE.md). Progress lives in [`TRACKER.md`](TRACKER.md).

## Setup

Python 3.11+ and NumPy 2.x. PyTorch is optional (only problem 039 uses it).

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

For problem 039, also install a CPU-only PyTorch (skip this if you already have torch):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## How to work on a problem

1. Open the problem's `README.md` and read the theory card (about 5 minutes).
2. Implement the functions in `problem.py`. The docstrings give the exact spec.
3. Run the tests for that problem only:

   ```bash
   pytest problems/01_arrays_dtypes_memory/001_creating_arrays
   ```

   Add `-x` to stop at the first failure, `-q` for short output.
4. When everything is green, answer the **self-check** questions out loud before opening the answers.
5. Compare with the reference solution in `solutions/<category>/<problem>.py`.
6. Update your row in `TRACKER.md`.

A fresh stub fails every test with `NotImplementedError`. That's expected.

## Running tests

```bash
pytest
```

runs every problem against your `problem.py` files.

```bash
SOLUTIONS=1 pytest
```

runs the same tests against the reference solutions (they should all pass). You can also do
this for one problem, e.g. to check that a test is fair:

```bash
SOLUTIONS=1 pytest problems/02_indexing/010_mixed_basic_advanced_indexing
```

If `pytest` fails at `import numpy` even though they're installed, your
`pytest` command comes from a different Python. Use `python -m pytest ...` instead.

## Starting a problem over

```bash
scripts/reset.sh 012        # one problem
scripts/reset.sh 001-008    # a range
scripts/reset.sh all        # everything
```

This puts `problem.py` back to the original stub. Your old attempts stay in git history.

Tests are offline and take a few seconds.

## Review rule (spaced repetition)

Solving a problem once is not fluency. After you solve a problem:

| when | what |
|---|---|
| after **3 days** | re-solve it |
| after **2 weeks** | re-solve it again |
| after **6 weeks** | re-solve it a last time |

Write the dates in `TRACKER.md` (`Last solved`, `Next review`).

**★ Core problems** (12 of them) are re-solved **from a blank file**: rename or
empty your `problem.py`, write the functions again using only the docstring spec (from
memory or from the README's "Your task"), and run the tests. These are the skills you
should be able to write in an interview or on a whiteboard without looking anything up.

If a re-solve fails, reset that problem's schedule to "3 days" again.
Each card's **Builds on** line tells you which earlier problems to review when you get stuck.

## Repository layout

```
CLAUDE.md          design notes and the full curriculum
README.md          this file
TRACKER.md         the sheet: one row per problem
conftest.py        test helpers: loads your code (or the solution), `ban`, `no_loops`
pytest.ini
requirements.txt
scripts/           check_batch.py (automatic review), reset.sh (fresh start)
problems/
  01_arrays_dtypes_memory/
    001_creating_arrays/
      README.md        theory card
      problem.py       your work
      test_problem.py  tests
solutions/
  01_arrays_dtypes_memory/
    001_creating_arrays.py
```

## Categories

| # | Category | Problems |
|---|---|---|
| 01 | Arrays, dtypes, memory | 001-005 |
| 02 | Indexing | 006-011 |
| 03 | Broadcasting and shape ops | 012-017 |
| 04 | Reductions and vectorized ops | 018-025 |
| 05 | Random and statistics | 026-030 |
| 06 | Linear algebra and numerics | 031-035 |
| 07 | Arrays in practice | 036-040 |
