"""Shared pytest setup.

Tests never import `problem.py` directly. They use the `sol` fixture, which loads
the code under test by file path:

* normally: the `problem.py` next to the test file (your work), or
* with the environment variable `SOLUTIONS=1`: the matching reference file in
  `solutions/<category>/<problem_folder>.py`.

Examples:
    pytest problems/01_arrays_dtypes_memory/001_creating_arrays
    SOLUTIONS=1 pytest
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import random
import textwrap
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent


def _use_solutions() -> bool:
    return os.environ.get("SOLUTIONS", "").strip() not in ("", "0", "false", "False")


def _target_file(test_file: Path) -> Path:
    problem_dir = test_file.resolve().parent
    if _use_solutions():
        category = problem_dir.parent.name
        return ROOT / "solutions" / category / f"{problem_dir.name}.py"
    return problem_dir / "problem.py"


def _load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Could not find the file under test: {path}")
    name = f"under_test_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sol(request) -> ModuleType:
    """The module under test: your problem.py, or the reference solution if SOLUTIONS=1."""
    return _load_module(_target_file(Path(str(request.path))))


@pytest.fixture(autouse=True)
def _fixed_seeds():
    """Every test starts from the same random state, so results are deterministic."""
    random.seed(0)
    np.random.seed(0)
    yield


@pytest.fixture
def ban(monkeypatch):
    """Replace a module function with one that fails loudly, to enforce "don't use X" rules.

    Usage inside a test:  ban(np, "unique")  or  ban(np.linalg, "inv")
    (ndarray methods such as `x.sort()` can't be replaced this way.)
    """

    def _ban(owner, name: str) -> None:
        label = f"{getattr(owner, '__name__', owner)}.{name}"

        def _banned(*args, **kwargs):
            raise AssertionError(f"`{label}` is banned in this problem. Build it yourself.")

        monkeypatch.setattr(owner, name, _banned)

    return _ban


_LOOP_NODES = (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@pytest.fixture
def no_loops():
    """Fail if a function's body contains a Python loop or comprehension.

    Usage inside a test:  no_loops(sol.pairwise_diff)
    The point of many NumPy problems is to let NumPy do the looping in C.
    """

    def _check(func) -> None:
        if isinstance(inspect.unwrap(func), type):
            return
        try:
            source = textwrap.dedent(inspect.getsource(func))
        except (OSError, TypeError):
            return
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, _LOOP_NODES):
                kind = type(node).__name__.replace("Comp", " comprehension").replace("GeneratorExp", "generator expression")
                raise AssertionError(
                    f"`{func.__name__}` uses a Python {kind.lower()} (line {node.lineno} of the function). "
                    "Use array operations instead."
                )

    return _check
