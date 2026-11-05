"""Automatic review of problems: structure, consistency and tests.

Run from the repo root:

    python scripts/check_batch.py                # every problem
    python scripts/check_batch.py 032-051        # one batch
    python scripts/check_batch.py 012            # one problem
    python scripts/check_batch.py --no-tests     # structure only, no pytest runs

What it checks for every problem folder:

  Files      README.md, problem.py, test_problem.py and the solution file exist;
             folder name is NNN_snake_case inside the right category folder.
  Card       Sections present and in the order CLAUDE.md asks for; 2-3 hints, each in
             its own non-empty <details>; 3-4 self-check questions with one answer each;
             "Builds on" only points to problems that exist; no inline [[...]] matrices;
             card short enough to read in about 5 minutes.
  Metadata   Title, CLAUDE.md curriculum and TRACKER.md agree on Core ★ and difficulty;
             the tracker row exists and its link works.
  Code       problem.py and the solution define the same public functions/classes with the
             same arguments; every public function in problem.py has a docstring;
             1-4 functions; tests only reach code through the `sol` fixture.
  Tests      With SOLUTIONS=1 every test passes and each test file runs in < 5 s.
             Untouched stubs fail only with NotImplementedError (stubs you already solved
             are skipped, they are your work).

Errors make the script exit with code 1. Warnings are things a person should look at.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"
SOLUTIONS = ROOT / "solutions"

SECTIONS = [
    "What it is",
    "Why it exists",
    "How it works",
    "Gotchas",
    "Builds on",
    "Your task",
    "Hints",
    "Self-check",
]
OPTIONAL_SECTIONS = ["Go deeper"]
CARD_WORD_LIMIT = 900          # about 5 minutes of reading
TEST_FILE_TIME_LIMIT = 5.0     # seconds


@dataclass
class Report:
    number: str
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ---------------------------------------------------------------- inputs


def parse_curriculum() -> dict[str, dict]:
    """Read problem number -> {core, difficulty} from CLAUDE.md."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"^(\d{3}) (.+?) - (Easy|Medium|Hard)\s*$", text, re.M):
        out[m.group(1)] = {"core": "★" in m.group(2), "difficulty": m.group(3)}
    return out


def parse_tracker() -> dict[str, dict]:
    path = ROOT / "TRACKER.md"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and re.fullmatch(r"\d{3}", cells[0]):
            link = re.search(r"\]\(([^)]+)\)", cells[1])
            out[cells[0]] = {
                "link": link.group(1) if link else None,
                "difficulty": cells[3],
                "core": "★" in cells[4],
            }
    return out


def problem_dirs(selection: set[str] | None) -> list[Path]:
    dirs = sorted(p for p in PROBLEMS.glob("*/*") if p.is_dir() and re.match(r"\d{3}_", p.name))
    if selection is not None:
        dirs = [d for d in dirs if d.name[:3] in selection]
    return dirs


def parse_selection(args: list[str]) -> set[str] | None:
    if not args:
        return None
    chosen = set()
    for arg in args:
        if "-" in arg:
            a, b = arg.split("-")
            chosen |= {f"{i:03d}" for i in range(int(a), int(b) + 1)}
        else:
            chosen.add(f"{int(arg):03d}")
    return chosen


# ---------------------------------------------------------------- card checks


def split_sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"^## +(.+?)\s*$", text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def details_blocks(body: str) -> list[tuple[str, str]]:
    """Return (summary, content) for every <details> block."""
    blocks = []
    for m in re.finditer(r"<details>(.*?)</details>", body, re.S):
        inner = m.group(1)
        s = re.search(r"<summary>(.*?)</summary>", inner, re.S)
        summary = s.group(1).strip() if s else ""
        content = re.sub(r"<summary>.*?</summary>", "", inner, flags=re.S).strip()
        blocks.append((summary, content))
    return blocks


def numbered_items(body: str) -> int:
    return len(re.findall(r"^\d+\.\s", body, re.M))


def check_card(r: Report, readme: Path, all_numbers: set[str], curriculum: dict) -> None:
    text = readme.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text else ""
    if not first.startswith(f"# {r.number}"):
        r.err(f"card title should start with '# {r.number}', found: {first!r}")

    sections = split_sections(text)
    names = [n for n, _ in sections]
    found = []
    for want in SECTIONS:
        idx = next((i for i, n in enumerate(names) if n.lower().startswith(want.lower())), None)
        if idx is None:
            r.err(f"card is missing the section '{want}'")
        else:
            found.append(idx)
    if found != sorted(found):
        r.err("card sections are not in the order CLAUDE.md asks for")
    extra = [n for n in names if not any(n.lower().startswith(s.lower()) for s in SECTIONS + OPTIONAL_SECTIONS)]
    if extra:
        r.warn(f"card has sections CLAUDE.md doesn't mention: {extra}")
    body = {n.lower(): b for n, b in sections}

    def get(prefix: str) -> str:
        return next((b for n, b in body.items() if n.startswith(prefix.lower())), "")

    # Hints: 2-3 separate, non-empty <details>
    hints = details_blocks(get("Hints"))
    if not 2 <= len(hints) <= 3:
        r.err(f"expected 2-3 hints in separate <details> blocks, found {len(hints)}")
    for i, (_, content) in enumerate(hints, 1):
        if not content:
            r.err(f"hint {i} is empty")

    # Self-check: 3-4 questions, answers inside <details>, one per question
    sc = get("Self-check")
    sc_details = details_blocks(sc)
    questions = numbered_items(re.sub(r"<details>.*?</details>", "", sc, flags=re.S))
    if not 3 <= questions <= 4:
        r.err(f"expected 3-4 self-check questions, found {questions}")
    if len(sc_details) != 1:
        r.err(f"expected one <details> block with the self-check answers, found {len(sc_details)}")
    elif not sc_details[0][1]:
        r.err("self-check answers block is empty")
    else:
        answers = numbered_items(sc_details[0][1])
        if answers != questions:
            r.err(f"{questions} self-check questions but {answers} answers")

    # Builds on: referenced problems exist; earlier is normal, later is flagged
    refs = re.findall(r"\b(\d{3})\b", get("Builds on"))
    for ref in refs:
        if ref not in curriculum:
            r.err(f"'Builds on' points to {ref}, which isn't in the curriculum")
        elif ref > r.number:
            r.warn(f"'Builds on' points ahead to {ref} (fine if the card says so)")
        elif ref not in all_numbers:
            r.warn(f"'Builds on' points to {ref}, which isn't built yet")

    # Inline nested-list matrices outside code blocks are against the style rule
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    prose = re.sub(r"`[^`]*`", "", prose)
    if "[[" in prose:
        r.warn("card shows a matrix as inline [[...]] - use a 2D grid")

    words = len(re.findall(r"\w+", prose))
    if words > CARD_WORD_LIMIT:
        r.warn(f"card is long ({words} words outside code); aim for about 5 minutes of reading")

    # Title star must match the curriculum
    meta = curriculum.get(r.number)
    if meta and ("★" in first) != meta["core"]:
        r.err(f"card title ★ doesn't match CLAUDE.md (Core = {meta['core']})")
    if meta and meta["difficulty"].lower() not in text.splitlines()[2].lower() if len(text.splitlines()) > 2 else False:
        r.warn(f"card doesn't state difficulty {meta['difficulty']} near the top")


# ---------------------------------------------------------------- code checks


def public_api(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    api = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            if isinstance(node, ast.FunctionDef):
                api[node.name] = [a.arg for a in node.args.args + node.args.kwonlyargs]
            else:
                api[node.name] = ["<class>"]
    return api


def is_untouched_stub(path: Path) -> bool:
    """True if every public function body is just docstring + raise NotImplementedError."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            funcs.append(node)
    if not funcs:
        return False
    for f in funcs:
        body = [s for s in f.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
        if not (len(body) == 1 and isinstance(body[0], ast.Raise)):
            return False
        exc = body[0].exc
        name = getattr(exc, "id", None) or getattr(getattr(exc, "func", None), "id", None)
        if name != "NotImplementedError":
            return False
    return True


def check_code(r: Report, folder: Path, solution: Path) -> None:
    problem = folder / "problem.py"
    stub_api = public_api(problem)
    sol_api = public_api(solution)
    funcs = [n for n, a in stub_api.items() if a != ["<class>"]]
    if not stub_api:
        r.err("problem.py defines nothing to implement")
    # Given config dataclasses (e.g. capstone `Config`) aren't work for the learner.
    stub_tree = ast.parse(problem.read_text(encoding="utf-8"))
    given = {
        n.name for n in stub_tree.body
        if isinstance(n, ast.ClassDef)
        and any("dataclass" in ast.unparse(d) for d in n.decorator_list)
    }
    todo = [n for n in stub_api if n not in given]
    if len(todo) > 4:
        r.warn(f"problem.py has {len(todo)} public functions/classes to write (CLAUDE.md says 1-4)")
    for name in stub_api:
        if name not in sol_api:
            r.err(f"solution is missing `{name}`")
        elif stub_api[name] != sol_api[name]:
            r.err(f"`{name}` has arguments {stub_api[name]} in problem.py but {sol_api[name]} in the solution")
    for name in sol_api:
        if name not in stub_api:
            r.warn(f"solution defines `{name}`, which problem.py doesn't")

    tree = ast.parse(problem.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            if not ast.get_docstring(node):
                r.err(f"`{node.name}` in problem.py has no docstring")

    test_src = (folder / "test_problem.py").read_text(encoding="utf-8")
    if re.search(r"^\s*(from|import)\s+problem\b", test_src, re.M):
        r.err("test_problem.py imports problem directly instead of using the `sol` fixture")
    if "sol" not in test_src:
        r.err("test_problem.py never uses the `sol` fixture")
    if re.search(r"download\s*=\s*True|urlopen|requests\.get|(?<!\w)weights\s*=\s*(?!None\b)['\"A-Z]", test_src):
        r.err("test_problem.py looks like it uses the network (downloads or pretrained weights)")


# ---------------------------------------------------------------- tests


def run_pytest(folder: Path, solutions: bool) -> tuple[int, str, float]:
    env = dict(os.environ)
    if solutions:
        env["SOLUTIONS"] = "1"
    else:
        env.pop("SOLUTIONS", None)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rN", "--tb=line", str(folder)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr, time.perf_counter() - start


def check_tests(r: Report, folder: Path) -> None:
    code, out, secs = run_pytest(folder, solutions=True)
    if code != 0:
        tail = "\n      ".join(out.strip().splitlines()[-6:])
        r.err(f"tests fail against the solution:\n      {tail}")
    elif secs > TEST_FILE_TIME_LIMIT + 3:  # allow for pytest start-up time
        r.warn(f"tests took {secs:.1f} s (limit is about {TEST_FILE_TIME_LIMIT:.0f} s)")

    if is_untouched_stub(folder / "problem.py"):
        code, out, _ = run_pytest(folder, solutions=False)
        failures = [l for l in out.splitlines() if re.search(r"(Error|Exception|assert)", l) and ".py:" in l]
        bad = [l for l in failures if "NotImplementedError" not in l]
        if code == 0:
            r.err("tests pass on the untouched stub - they don't test anything")
        elif bad:
            r.err("stub fails with something other than NotImplementedError:\n      " + "\n      ".join(bad[:3]))


# ---------------------------------------------------------------- main


def main() -> int:
    args = sys.argv[1:]
    skip_tests = "--no-tests" in args
    args = [a for a in args if a != "--no-tests"]
    selection = parse_selection(args)
    curriculum = parse_curriculum()
    tracker = parse_tracker()
    dirs = problem_dirs(selection)
    if not dirs:
        print("No problem folders matched.")
        return 1
    built = {d.name[:3] for d in PROBLEMS.glob("*/*") if d.is_dir()}

    reports = []
    for folder in dirs:
        r = Report(folder.name[:3], folder)
        reports.append(r)
        print(f"checking {folder.relative_to(ROOT)} ...", end="\r", flush=True)

        if not re.fullmatch(r"\d{3}_[a-z0-9_]+", folder.name):
            r.err(f"folder name '{folder.name}' isn't NNN_snake_case")
        if not re.fullmatch(r"\d{2}_[a-z0-9_]+", folder.parent.name):
            r.err(f"category folder '{folder.parent.name}' isn't NN_snake_case")
        if r.number not in curriculum:
            r.err("this number isn't in the CLAUDE.md curriculum")

        solution = SOLUTIONS / folder.parent.name / f"{folder.name}.py"
        missing = [p for p in (folder / "README.md", folder / "problem.py", folder / "test_problem.py", solution) if not p.exists()]
        for p in missing:
            r.err(f"missing file: {p.relative_to(ROOT)}")
        if missing:
            continue

        check_card(r, folder / "README.md", built, curriculum)
        check_code(r, folder, solution)

        row = tracker.get(r.number)
        meta = curriculum.get(r.number)
        if row is None:
            r.err("no row in TRACKER.md")
        else:
            if not row["link"] or not (ROOT / row["link"]).exists():
                r.err(f"TRACKER.md link is broken: {row['link']}")
            if meta and row["core"] != meta["core"]:
                r.err("TRACKER.md Core ★ doesn't match CLAUDE.md")
            if meta and row["difficulty"] != meta["difficulty"]:
                r.err(f"TRACKER.md difficulty '{row['difficulty']}' doesn't match CLAUDE.md '{meta['difficulty']}'")

        if not skip_tests:
            check_tests(r, folder)

    print(" " * 80, end="\r")
    n_err = sum(len(r.errors) for r in reports)
    n_warn = sum(len(r.warnings) for r in reports)
    for r in reports:
        if r.errors or r.warnings:
            print(f"\n{r.path.relative_to(ROOT)}")
            for e in r.errors:
                print(f"  ERROR    {e}")
            for w in r.warnings:
                print(f"  warning  {w}")
    clean = sum(1 for r in reports if not r.errors and not r.warnings)
    print(f"\n{len(reports)} problems checked: {clean} clean, {n_err} errors, {n_warn} warnings.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
