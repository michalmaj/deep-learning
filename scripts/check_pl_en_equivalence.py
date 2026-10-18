"""Structural PL/EN equivalence check for a bilingual notebook pair.

This is a pilot tool for module 01's PL/EN localization. It checks that a
Polish notebook and its English counterpart have the same *computational*
shape -- same cells, same code structure, same numeric constants, same
identifiers, same TODO positions, same equations -- while allowing their
prose (markdown text, comments, docstrings, print messages, string
literals) to differ, since that prose is exactly what translation changes.

What this script checks, per notebook pair:

1. Both files exist.
2. Same number of cells, same cell-type sequence (markdown/code, in order).
3. Each code cell pair has structurally identical ASTs, where string
   constants are treated as presentational (ignored) but every other node
   (identifiers, attributes, operators, numeric/bool/None constants,
   control flow) must match exactly.
4. The numeric constants appearing in each code cell, in order, are
   identical between PL and EN -- this is the same guarantee as (3) but
   reported separately so a seed/hyperparameter/data-size drift produces a
   readable, specific diagnostic instead of a generic "AST mismatch".
5. The set and order of function names containing a `raise
   NotImplementedError(...)` (i.e. student TODOs) is identical.
6. Markdown cells: same count, none empty, and the LaTeX expressions
   (`$...$` and `$$...$$`, extracted in order) are identical between PL and
   EN -- equations are never translated.

What this script explicitly does NOT check (documented limits, not bugs):

- It does not judge whether the English prose is a *faithful* translation
  of the Polish prose -- only that code/numeric/equation structure lines
  up. Reading both versions is still required.
- It does not execute the notebooks. Running `examples.en.ipynb` from a
  clean kernel, and validating `lab.en.ipynb` against the private solution,
  are separate validation steps.
- It does not understand whether a string literal is purely presentational
  or affects computation (e.g. a column name, a class label used for
  lookups). For module 01, this was checked manually: no string constant in
  either notebook is consumed by downstream code (the only
  computation-adjacent strings are `"cuda"`/`"cpu"` inside
  `torch.device(...)`, which are produced by `torch.cuda.is_available()`
  branching, not by any translated text, so they are identical in both
  languages by construction and are not flagged by this limitation).
- It requires PL and EN notebooks to have the same number of cells in the
  same order -- it cannot handle content reorganized between languages.
  For a 1:1 localization this is the intended requirement, not a gap.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import nbformat

LATEX_PATTERN = re.compile(r"\$\$.+?\$\$|\$.+?\$", re.DOTALL)


def _normalize_ast(node: ast.AST | list | object) -> object:
    """Recursively reduce an AST (sub)tree to a comparable, order-preserving
    structure, replacing string constants with a presentational placeholder.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            return ("Constant", "STR")
        return ("Constant", repr(value))
    if isinstance(node, ast.AST):
        return (
            type(node).__name__,
            [_normalize_ast(value) for _, value in ast.iter_fields(node)],
        )
    if isinstance(node, list):
        return [_normalize_ast(item) for item in node]
    return node


def _numeric_constants(tree: ast.AST) -> list[str]:
    """Non-string Constant values, in the order they appear in the AST."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, str):
            out.append(repr(node.value))
    return out


def _todo_function_names(tree: ast.AST) -> list[str]:
    """Names of functions (in source order) whose body directly raises
    NotImplementedError -- i.e. student TODOs, independent of the message
    text passed to the exception (which is translated).
    """
    names = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.Raise)
                    and isinstance(stmt.exc, ast.Call)
                    and isinstance(stmt.exc.func, ast.Name)
                    and stmt.exc.func.id == "NotImplementedError"
                ):
                    names.append(node.name)
                    break
            self.generic_visit(node)

        def visit_Module(self, node: ast.Module) -> None:
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Raise)
                    and isinstance(stmt.exc, ast.Call)
                    and isinstance(stmt.exc.func, ast.Name)
                    and stmt.exc.func.id == "NotImplementedError"
                ):
                    names.append("<module level>")
            self.generic_visit(node)

    _Visitor().visit(tree)
    return names


def _extract_latex(markdown_source: str) -> list[str]:
    return LATEX_PATTERN.findall(markdown_source)


def check_notebook_pair(pl_path: Path, en_path: Path) -> list[str]:
    """Returns a list of human-readable problem descriptions; empty if the
    PL/EN pair is structurally equivalent under the rules documented in the
    module docstring.
    """
    problems: list[str] = []

    if not pl_path.exists():
        return [f"missing PL notebook: {pl_path}"]
    if not en_path.exists():
        return [f"missing EN notebook: {en_path}"]

    pl_nb = nbformat.read(pl_path, as_version=4)
    en_nb = nbformat.read(en_path, as_version=4)

    pl_cells = pl_nb.cells
    en_cells = en_nb.cells

    if len(pl_cells) != len(en_cells):
        problems.append(
            f"cell count mismatch: PL has {len(pl_cells)} cells, "
            f"EN has {len(en_cells)} cells"
        )

    for i, (pl_cell, en_cell) in enumerate(zip(pl_cells, en_cells)):
        if pl_cell.cell_type != en_cell.cell_type:
            problems.append(
                f"cell {i}: type mismatch (PL={pl_cell.cell_type!r}, "
                f"EN={en_cell.cell_type!r})"
            )
            continue

        if pl_cell.cell_type == "code":
            try:
                pl_tree = ast.parse(pl_cell.source)
            except SyntaxError as exc:
                problems.append(f"cell {i}: PL code does not parse: {exc}")
                continue
            try:
                en_tree = ast.parse(en_cell.source)
            except SyntaxError as exc:
                problems.append(f"cell {i}: EN code does not parse: {exc}")
                continue

            if _normalize_ast(pl_tree) != _normalize_ast(en_tree):
                problems.append(
                    f"cell {i}: code structure differs between PL and EN "
                    f"beyond string literals (identifiers, operators, "
                    f"constants, or control flow do not match)"
                )

            pl_nums = _numeric_constants(pl_tree)
            en_nums = _numeric_constants(en_tree)
            if pl_nums != en_nums:
                problems.append(
                    f"cell {i}: numeric constants differ -- PL={pl_nums}, "
                    f"EN={en_nums} (seed/hyperparameter/data-size drift?)"
                )

            pl_todos = _todo_function_names(pl_tree)
            en_todos = _todo_function_names(en_tree)
            if pl_todos != en_todos:
                problems.append(
                    f"cell {i}: TODO (NotImplementedError) function names "
                    f"differ -- PL={pl_todos}, EN={en_todos}"
                )

        elif pl_cell.cell_type == "markdown":
            if not pl_cell.source.strip():
                problems.append(f"cell {i}: PL markdown cell is empty")
            if not en_cell.source.strip():
                problems.append(f"cell {i}: EN markdown cell is empty")

            pl_latex = _extract_latex(pl_cell.source)
            en_latex = _extract_latex(en_cell.source)
            if pl_latex != en_latex:
                problems.append(
                    f"cell {i}: LaTeX expressions differ -- PL={pl_latex}, "
                    f"EN={en_latex}"
                )

    return problems


MODULE_01_PAIRS = [
    (
        Path("lectures/01_intro/examples.ipynb"),
        Path("lectures/01_intro/examples.en.ipynb"),
    ),
    (
        Path("labs/01_intro/lab.ipynb"),
        Path("labs/01_intro/lab.en.ipynb"),
    ),
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    all_problems: dict[str, list[str]] = {}

    for pl_rel, en_rel in MODULE_01_PAIRS:
        pl_path = repo_root / pl_rel
        en_path = repo_root / en_rel
        problems = check_notebook_pair(pl_path, en_path)
        if problems:
            all_problems[f"{pl_rel} <-> {en_rel}"] = problems

    if not all_problems:
        print("OK: all checked PL/EN notebook pairs are structurally equivalent.")
        return 0

    for pair, problems in all_problems.items():
        print(f"[{pair}]")
        for problem in problems:
            print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
