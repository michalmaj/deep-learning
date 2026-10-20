"""Validate and run one cache-friendly shard of network-dependent tests."""

import argparse
import ast
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NETWORK_TEST_SHARDS: dict[str, tuple[str, ...]] = {
    "fashion": (
        "tests/test_lecture_06_notebook.py",
        "tests/test_lecture_06_notebook_en.py",
        "tests/test_lab_06_scaffold.py",
        "tests/test_lab_06_scaffold_en.py",
        "tests/test_lecture_12_notebook.py",
        "tests/test_lecture_12_notebook_en.py",
        "tests/test_lab_12_scaffold.py",
        "tests/test_lab_12_scaffold_en.py",
        "tests/test_lecture_14_notebook.py",
        "tests/test_lecture_14_notebook_en.py",
        "tests/test_lab_14_scaffold.py",
        "tests/test_lab_14_scaffold_en.py",
        "tests/test_lecture_15_notebook.py",
        "tests/test_lecture_15_notebook_en.py",
    ),
    "vision": (
        "tests/test_lecture_07_notebook.py",
        "tests/test_lecture_07_notebook_en.py",
        "tests/test_lab_07_scaffold.py",
        "tests/test_lab_07_scaffold_en.py",
        "tests/test_lecture_08_notebook.py",
        "tests/test_lecture_08_notebook_en.py",
        "tests/test_lab_08_scaffold.py",
        "tests/test_lab_08_scaffold_en.py",
        "tests/test_lab_15_scaffold.py",
        "tests/test_lab_15_scaffold_en.py",
    ),
    "language": (
        "tests/test_lab_09_scaffold.py",
        "tests/test_lab_09_scaffold_en.py",
        "tests/test_lecture_11_notebook.py",
        "tests/test_lecture_11_notebook_en.py",
        "tests/test_lab_11_scaffold.py",
        "tests/test_lab_11_scaffold_en.py",
    ),
}


def _is_network_marker(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "network"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _contains_network_test(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(
            _is_network_marker(decorator) for decorator in node.decorator_list
        ):
            return True
    return False


def discover_network_test_files(repo_root: Path = REPO_ROOT) -> set[str]:
    tests_dir = repo_root / "tests"
    return {
        path.relative_to(repo_root).as_posix()
        for path in tests_dir.glob("test_*.py")
        if _contains_network_test(path)
    }


def validate_network_test_shards(
    repo_root: Path = REPO_ROOT,
    shards: dict[str, tuple[str, ...]] = NETWORK_TEST_SHARDS,
) -> list[str]:
    problems: list[str] = []
    discovered = discover_network_test_files(repo_root)
    assignments = [path for paths in shards.values() for path in paths]
    assignment_counts = Counter(assignments)

    for name, paths in shards.items():
        if not paths:
            problems.append(f"network shard {name!r} is empty")

    duplicates = sorted(path for path, count in assignment_counts.items() if count > 1)
    if duplicates:
        problems.append(f"network tests assigned more than once: {', '.join(duplicates)}")

    assigned = set(assignments)
    missing = sorted(discovered - assigned)
    if missing:
        problems.append(f"network tests missing from shards: {', '.join(missing)}")

    nonexistent = sorted(path for path in assigned if not (repo_root / path).is_file())
    if nonexistent:
        problems.append(f"shards reference missing files: {', '.join(nonexistent)}")

    unmarked = sorted((assigned - discovered) - set(nonexistent))
    if unmarked:
        problems.append(f"shards reference files without network tests: {', '.join(unmarked)}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", nargs="?", choices=NETWORK_TEST_SHARDS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate complete, unique shard coverage without running tests",
    )
    args = parser.parse_args()

    problems = validate_network_test_shards()
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    if args.check:
        print(
            f"network test shards: OK "
            f"({len(discover_network_test_files())} files across {len(NETWORK_TEST_SHARDS)} shards)"
        )
        return 0
    if args.shard is None:
        parser.error("provide a shard name or use --check")

    paths = NETWORK_TEST_SHARDS[args.shard]
    print(f"running network shard {args.shard!r}: {len(paths)} files", flush=True)
    command = [sys.executable, "-m", "pytest", "-m", "network", "--tb=short", *paths]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
