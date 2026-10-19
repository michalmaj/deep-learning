"""Structural PL/EN equivalence check for a bilingual notebook pair.

This is a localization-infrastructure tool, piloted on module 01 and
extended here to module 10. It checks that a Polish notebook and its
English counterpart have the same *computational* shape -- same cells,
same code structure, same numeric constants, same identifiers, same TODO
positions, same equations -- while allowing genuinely presentational text
(markdown prose, comments, docstrings, print/plot messages) to differ,
since that is exactly what translation changes.

What this script checks, per notebook pair:

1. Both files exist.
2. Same number of cells, same cell-type sequence (markdown/code, in order).
3. Each code cell pair has structurally identical ASTs. String constants
   are compared *exactly* by default -- a string literal is only allowed
   to differ between PL and EN when it sits in a narrow, explicitly
   recognized presentational context (see "String-literal protection"
   below). Every other node (identifiers, attributes, operators,
   numeric/bool/None constants, control flow) must match exactly
   regardless of context.
4. The numeric constants appearing in each code cell, in order, are
   identical between PL and EN -- this is already implied by (3) but
   reported separately so a seed/hyperparameter/data-size drift produces a
   readable, specific diagnostic instead of a generic "AST mismatch".
5. The set and order of function names containing a `raise
   NotImplementedError(...)` (i.e. student TODOs) is identical.
6. Markdown cells: same count, none empty, and the LaTeX expressions
   (`$...$` and `$$...$$`, extracted in order) are identical between PL and
   EN -- equations are never translated.

## String-literal protection

Earlier (module 01 pilot), every string constant was treated as
presentational and ignored -- this is too permissive: it would silently
accept a translated dictionary key, data identifier, or configuration
string that actually changes what the code computes, not just how it
reads. A string literal is now exempted from the exact-match requirement
only when it is one of:

- an argument (positional or keyword, including inside an f-string) to a
  call recognized as display-only: `print`, or a call whose attribute name
  is a plotting label/title setter (`title`, `xlabel`, `ylabel`,
  `suptitle`, `set_title`, `set_xlabel`, `set_ylabel`, `set_label`);
- a `label=` keyword argument to *any* call (matplotlib's standard way of
  giving a plotted series its legend text, e.g. `plt.scatter(..., label=...)`
  or `plt.plot(..., label=...)`) -- this is matched on the keyword name
  alone, not the function, since legend labels appear on many different
  plotting calls;
- the exception message passed to `NotImplementedError(...)` or
  `AssertionError(...)`;
- the `msg` of a bare `assert condition, "message"`;
- a module/function/class docstring (the literal first statement of its
  body, if it is a bare string expression);
- a string bound, within the same cell, to a plain variable name (by a
  direct `name = "..."` assignment, or as a `for name, other in
  [("...", ...), ...]:` loop-unpacking target) *and* that same name is
  referenced -- anywhere in the cell, not necessarily in the same
  statement -- inside the arguments of a presentational call. This is a
  single, same-cell, one-hop back-reference (an f-string header assigned
  to a variable before being printed; a human-readable label walked over
  in a `for` loop and only ever interpolated into a later `print(...)`),
  not a general data-flow analysis -- a string reaching a presentational
  call through two hops of assignment, a conditional, or a function
  boundary is not covered and will correctly surface as a mismatch for
  manual review.

Every other string constant -- a dict key, a column/label identifier, an
argument to an unrecognized function, a value compared against with `==`,
anything not matching one of the categories above -- must be byte-identical
between PL and EN, or the check reports a mismatch. This is deliberately a
narrow allowlist, not a general "is this string used in a computation?"
analysis: see the limitations section below.

What this script explicitly does NOT check (documented limits, not bugs):

- It does not judge whether the English prose is a *faithful* translation
  of the Polish prose -- only that code/numeric/equation structure lines
  up. Reading both versions is still required.
- It does not execute the notebooks. Running `examples.en.ipynb` from a
  clean kernel, and validating `lab.en.ipynb` against the private solution,
  are separate validation steps.
- The string-literal allowlist above is a syntactic heuristic, not a data-
  flow analysis. A string literal used in a genuinely presentational way
  through a call this script does not recognize (e.g. a custom logging
  helper) will be flagged as a mismatch even though it is harmless --
  this is a false positive to fix by widening the allowlist deliberately,
  not a reason to loosen the default to "ignore all strings" again. The
  reverse case -- a presentational-looking call that happens to carry a
  computationally meaningful string -- was checked manually for modules 01
  and 10 (the only computation-adjacent strings found are `"cuda"`/`"cpu"`
  inside `torch.device(...)`, which are produced by
  `torch.cuda.is_available()` branching, not by translated text, and
  `torch.device` is not in the presentational allowlist so they are
  compared exactly anyway).
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

PRESENTATIONAL_CALL_NAMES = {"print", "NotImplementedError", "AssertionError"}
PRESENTATIONAL_ATTR_NAMES = {
    "title",
    "xlabel",
    "ylabel",
    "suptitle",
    "set_title",
    "set_xlabel",
    "set_ylabel",
    "set_label",
}


def _mark_string_constants(node: ast.AST, out: set[int]) -> None:
    """Record id() of every ast.Constant(str) in node's subtree into out."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.add(id(sub))


def _mark_docstring(node: ast.AST, out: set[int]) -> None:
    body = getattr(node, "body", None)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        _mark_string_constants(body[0].value, out)


def _is_presentational_call(node: ast.Call) -> bool:
    func = node.func
    return (isinstance(func, ast.Name) and func.id in PRESENTATIONAL_CALL_NAMES) or (
        isinstance(func, ast.Attribute) and func.attr in PRESENTATIONAL_ATTR_NAMES
    )


def _referenced_names(node: ast.AST) -> set[str]:
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }


def _presentational_string_ids(tree: ast.AST) -> set[int]:
    """id() of every string Constant node allowed to differ between PL and
    EN -- see the "String-literal protection" section in the module
    docstring for the exact, narrow set of recognized contexts.
    """
    ids: set[int] = set()
    presentational_names: set[str] = set()

    class _DirectVisitor(ast.NodeVisitor):
        """First pass: strings directly inside a presentational context, and
        the set of plain names referenced there (for the second pass)."""

        def visit_Call(self, node: ast.Call) -> None:
            if _is_presentational_call(node):
                for arg in node.args:
                    _mark_string_constants(arg, ids)
                    presentational_names.update(_referenced_names(arg))
                for kw in node.keywords:
                    _mark_string_constants(kw.value, ids)
                    presentational_names.update(_referenced_names(kw.value))
            else:
                for kw in node.keywords:
                    if kw.arg == "label":
                        _mark_string_constants(kw.value, ids)
                        presentational_names.update(_referenced_names(kw.value))
            self.generic_visit(node)

        def visit_Assert(self, node: ast.Assert) -> None:
            if node.msg is not None:
                _mark_string_constants(node.msg, ids)
                presentational_names.update(_referenced_names(node.msg))
            self.generic_visit(node)

        def visit_Module(self, node: ast.Module) -> None:
            _mark_docstring(node, ids)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            _mark_docstring(node, ids)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            _mark_docstring(node, ids)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            _mark_docstring(node, ids)
            self.generic_visit(node)

    _DirectVisitor().visit(tree)

    def _mark_if_plain_string(node: ast.AST) -> None:
        """Only mark node itself as presentational if it IS a plain string
        literal or f-string -- never recurse into an arbitrary expression
        (e.g. a nested function call) just because it is reachable from a
        presentational-bound name. This is what keeps the one-hop rule from
        swallowing a computationally meaningful string nested inside some
        unrelated call on the right-hand side, such as `torch.device("cpu")`
        assigned to a variable that later gets printed."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            ids.add(id(node))
        elif isinstance(node, ast.JoinedStr):
            _mark_string_constants(node, ids)

    class _OneHopVisitor(ast.NodeVisitor):
        """Second pass: a string bound to a name in presentational_names via
        a direct assignment or for-loop tuple-unpacking is presentational
        too -- see the docstring's one-hop back-reference bullet. Only the
        bound expression itself is considered, never an expression nested
        inside some other call on the right-hand side (see
        `_mark_if_plain_string`)."""

        def visit_Assign(self, node: ast.Assign) -> None:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in presentational_names:
                    _mark_if_plain_string(node.value)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            target = node.target
            iterable = node.iter
            if (
                isinstance(target, ast.Tuple)
                and isinstance(iterable, (ast.List, ast.Tuple))
                and all(isinstance(elt, ast.Tuple) for elt in iterable.elts)
            ):
                for pos, name_node in enumerate(target.elts):
                    if (
                        isinstance(name_node, ast.Name)
                        and name_node.id in presentational_names
                    ):
                        for elt in iterable.elts:
                            if pos < len(elt.elts):
                                _mark_if_plain_string(elt.elts[pos])
            elif isinstance(target, ast.Name) and target.id in presentational_names:
                if isinstance(iterable, (ast.List, ast.Tuple)):
                    for elt in iterable.elts:
                        _mark_if_plain_string(elt)
            self.generic_visit(node)

    _OneHopVisitor().visit(tree)
    return ids


def _normalize_ast(
    node: ast.AST | list | object, presentational_ids: set[int]
) -> object:
    """Recursively reduce an AST (sub)tree to a comparable, order-preserving
    structure. A string constant becomes a presentational placeholder only
    if its id() is in `presentational_ids`; every other string constant
    keeps its literal value, so an unrecognized or computationally
    meaningful string difference still shows up as a mismatch.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            if id(node) in presentational_ids:
                return ("Constant", "STR")
            return ("Constant", "STR_LITERAL", value)
        return ("Constant", repr(value))
    if isinstance(node, ast.AST):
        return (
            type(node).__name__,
            [_normalize_ast(value, presentational_ids) for _, value in ast.iter_fields(node)],
        )
    if isinstance(node, list):
        return [_normalize_ast(item, presentational_ids) for item in node]
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

            pl_presentational = _presentational_string_ids(pl_tree)
            en_presentational = _presentational_string_ids(en_tree)
            if _normalize_ast(pl_tree, pl_presentational) != _normalize_ast(
                en_tree, en_presentational
            ):
                problems.append(
                    f"cell {i}: code structure differs between PL and EN -- "
                    f"identifiers, operators, numeric/bool/None constants, "
                    f"control flow, or a non-presentational string literal "
                    f"do not match (see 'String-literal protection' in this "
                    f"script's module docstring for which strings are "
                    f"exempt)"
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


MODULE_NOTEBOOK_PAIRS = [
    (
        Path("lectures/01_intro/examples.ipynb"),
        Path("lectures/01_intro/examples.en.ipynb"),
    ),
    (
        Path("labs/01_intro/lab.ipynb"),
        Path("labs/01_intro/lab.en.ipynb"),
    ),
    (
        Path("lectures/10_attention/examples.ipynb"),
        Path("lectures/10_attention/examples.en.ipynb"),
    ),
    (
        Path("labs/10_attention/lab.ipynb"),
        Path("labs/10_attention/lab.en.ipynb"),
    ),
]
"""Every localized PL/EN notebook pair this project currently maintains.
Adding a new module's pair here is the only change needed to cover it --
the checking logic above is module-agnostic."""


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    all_problems: dict[str, list[str]] = {}

    for pl_rel, en_rel in MODULE_NOTEBOOK_PAIRS:
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
