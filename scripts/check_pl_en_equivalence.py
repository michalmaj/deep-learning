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
   EN -- equations are never translated. Two narrow, bounded
   normalizations are applied to both sides before comparing:
   - **Decimal notation**: a Polish `X{,}Y` (the LaTeX-escaped decimal
     comma) is normalized to `X.Y`, since localizing the decimal separator
     is required, not a translation error -- this only touches digits
     wrapped in the literal `{,}` escape, never a bare comma (an argument
     separator, a coordinate pair, a list), so it cannot mask an actual
     change to a formula.
   - **Thousands separator**: Polish large numbers use a thin-space escape
     between digit groups (`2\\,000`, rendering as "2 000"); this course's
     English localization drops the separator entirely (`2000`) rather
     than switching to the English comma convention, to avoid colliding
     with the decimal-comma normalization above. The `\\,` is therefore
     stripped (not replaced) from both sides before comparing, touching
     only a digit-`\\,`-digit sequence.
   - **`\text{...}` content**: the content of every `\text{...}` command is
     replaced with a placeholder before comparing. This is a deliberate,
     bounded risk: this course's localization rule explicitly requires
     translating a prose label inside `\text{...}` while preserving the
     expression's mathematical meaning (e.g. $\varepsilon_{\text{maszyny}}$
     -> $\varepsilon_{\text{machine}}$, $\text{podwarstwa}(x)$ ->
     $\text{sublayer}(x)$), so the content is *expected* to differ. The
     real risk this accepts: if `\text{...}` is used as a technical
     subscript rather than a prose label (e.g. $d_{\text{model}}$) and a
     translation mistakenly changes it to a different identifier (e.g.
     $d_{\text{data}}$) while leaving the rest of the equation unchanged,
     this normalization will not catch it -- there is no syntactic way to
     tell a prose label apart from a technical identifier inside
     `\text{...}`. Guarding against that specific mistake is left to the
     manual equation-by-equation review every lecture translation already
     gets, not to this script. Every other part of the equation (the
     number and position of `\text{...}` commands, every operator, symbol,
     and numeric value outside them) is still compared exactly.

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
  manual review. The loop-unpacking variant also unwraps two common
  wrapper shapes around the loop's iterable before matching: `zip(a, b)`
  (each (sub-target, sub-iterable) pair from the zip is matched
  independently, so `for ax, (title, data) in zip(axes, [("...", ...),
  ...]):` resolves `title` against the second zip argument) and
  `enumerate(x)` (the index is irrelevant; the value half of the `(idx,
  val)` target is matched directly against `x`'s elements, so `for i, tok
  in enumerate(VOCAB):` resolves `tok` against `VOCAB`'s elements). In
  both cases the wrapped collection may itself be a literal or a plain
  `name = [...]` assigned earlier in the same cell (one further hop on the
  *iterable* side only, not the call-argument side above). A third shape,
  `name.items()` with a `for key, value in ...:` target, is handled the
  same way: when `key` is presentational, the dict literal's own key
  Constants are marked presentational too -- its *value* Constants are
  not touched, so a dict mapping a display label to actual experimental
  text (e.g. a sentence fed to a tokenizer) can have its label translated
  while the text itself stays exact-matched;
- an argument passed, anywhere in the notebook (not just the same cell),
  to a parameter of a locally defined helper function, when that
  parameter is used *exclusively* inside a presentational call within the
  function's own body -- e.g. `def report(name, model, loss): print(f"{name}
  ...")` makes `report`'s `name` parameter presentational, so
  `report("Linear", ...)` vs `report("Linear model", ...)` is allowed, but
  `report(model, ...)`'s `model` argument is untouched because `model` is
  also passed to a non-presentational call inside `report`. This is
  resolved once per notebook (all code cells parsed together, since a
  helper is typically defined in one cell and called from a later one),
  matching positional arguments by position and keyword arguments by
  name; `*args`/`**kwargs` and parameters used inside a nested function
  are not resolved and fall back to exact-match, which is the safe
  direction for a limitation to fail in.

- a string key of a notebook-local **display/grouping accumulator**
  dict: a bare `name = {}` or a dict literal with plain-string-only keys,
  used *only* to collect results for later printing/plotting (every
  other reference to `name` in the cell is `name[<str>] = ...`,
  `name[<str>] += ...`, `name[<str>].append(/extend(/update(...)`, one of
  `name.items()/.values()/.keys()`, or the bare iterable of a `for`
  loop). A single use anywhere else -- passed to a call, subscripted
  with a computed key, compared with `in`/`==`, reassigned to something
  other than a dict -- disqualifies the name entirely and falls back to
  exact-match, so a genuine config/parameter dict (e.g.
  `{"params": layer.parameters(), "lr": 1e-4}` fed straight into an
  optimizer) is never exempted just because its keys happen to be
  strings.

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


def _name_load_counts(node: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            counts[sub.id] = counts.get(sub.id, 0) + 1
    return counts


def _presentational_name_load_counts(node: ast.AST) -> dict[str, int]:
    """Like `_name_load_counts`, but only counting Name(Load) references
    that occur inside a presentational-call argument (or a `label=`
    keyword) -- used to decide whether a function parameter's *every*
    reference is presentational."""
    counts: dict[str, int] = {}

    def _add(sub_tree: ast.AST) -> None:
        for name, n in _name_load_counts(sub_tree).items():
            counts[name] = counts.get(name, 0) + n

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if _is_presentational_call(node):
                for arg in node.args:
                    _add(arg)
                for kw in node.keywords:
                    _add(kw.value)
            else:
                for kw in node.keywords:
                    if kw.arg == "label":
                        _add(kw.value)
            self.generic_visit(node)

    _Visitor().visit(node)
    return counts


def _collect_function_presentational_params(
    cell_sources: list[str],
) -> dict[str, tuple[list[str], set[str]]]:
    """Resolved once per notebook (all code cells parsed together, since a
    helper is commonly defined in one cell and called from a later one):
    for every locally defined function, the ordered list of its parameter
    names and the subset of those parameters whose *every* reference
    inside the function's own body is a presentational-call argument (see
    the "String-literal protection" section). Cells that fail to parse are
    skipped for this registry (they still get their own syntax-error
    report elsewhere); `*args`/`**kwargs` and parameters only used inside a
    nested function are not resolved, which is the safe direction for a
    limitation to fail in.
    """
    registry: dict[str, tuple[list[str], set[str]]] = {}
    for source in cell_sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            param_names = [a.arg for a in node.args.args]
            total = _name_load_counts(node)
            presentational = _presentational_name_load_counts(node)
            presentational_params = {
                name
                for name in param_names
                if presentational.get(name, 0) > 0 and total.get(name, 0) == presentational[name]
            }
            registry[node.name] = (param_names, presentational_params)
    return registry


def _collect_accumulator_dict_names(cell_sources: list[str]) -> set[str]:
    """Resolved once per notebook (all code cells parsed together, since
    the dict is commonly initialized in one cell and populated/read in a
    later one): the names of every dict that behaves as a notebook-wide
    *display/grouping accumulator* -- see the "dict-accumulator" bullet
    in the "String-literal protection" section. A name is a candidate the
    moment any cell assigns it a `{}` or a dict literal with
    plain-string-only keys; it is then dropped if ANY cell contains a
    reference to that name outside the narrow accumulation/iteration
    patterns the bullet describes (`name[<str>] = ...`,
    `name[<str>] += ...`, `name[<str>].append(/extend(/update(...)`, one
    of `name.items()/.values()/.keys()`, or the bare iterable of a `for`
    loop) -- so a dict used for anything else (passed to a call,
    subscripted with a computed key, compared with `in`/`==`, ...)
    anywhere in the notebook is never treated as a display accumulator.
    Cells that fail to parse are skipped, the safe direction for a
    limitation to fail in.
    """
    trees = []
    for source in cell_sources:
        try:
            trees.append(ast.parse(source))
        except SyntaxError:
            continue

    candidates: set[str] = set()
    for tree in trees:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)
                and all(
                    isinstance(k, ast.Constant) and isinstance(k.value, str)
                    for k in node.value.keys
                )
            ):
                candidates.add(node.targets[0].id)

    disqualified: set[str] = set()
    for tree in trees:
        parents: dict[int, ast.AST] = {}
        for parent_node in ast.walk(tree):
            for child_node in ast.iter_child_nodes(parent_node):
                parents[id(child_node)] = parent_node

        def _is_allowed(name_node: ast.Name) -> bool:
            parent = parents.get(id(name_node))
            if isinstance(parent, ast.Subscript) and parent.value is name_node:
                key = parent.slice
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    return False
                grandparent = parents.get(id(parent))
                if isinstance(grandparent, ast.Assign) and grandparent.targets == [parent]:
                    return True
                if isinstance(grandparent, ast.AugAssign) and grandparent.target is parent:
                    return True
                if (
                    isinstance(grandparent, ast.Attribute)
                    and grandparent.value is parent
                    and grandparent.attr in {"append", "extend", "update"}
                ):
                    return True
                return False
            if (
                isinstance(parent, ast.Attribute)
                and parent.value is name_node
                and parent.attr in {"items", "values", "keys"}
            ):
                return True
            return isinstance(parent, ast.For) and parent.iter is name_node

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and node.id in candidates
                and isinstance(node.ctx, ast.Load)
                and not _is_allowed(node)
            ):
                disqualified.add(node.id)

    return candidates - disqualified


def _presentational_string_ids(
    tree: ast.AST,
    global_function_params: dict[str, tuple[list[str], set[str]]] | None = None,
    accumulator_names: set[str] | None = None,
) -> set[int]:
    """id() of every string Constant node allowed to differ between PL and
    EN -- see the "String-literal protection" section in the module
    docstring for the exact, narrow set of recognized contexts.
    """
    ids: set[int] = set()
    presentational_names: set[str] = set()
    accumulator_names = accumulator_names or set()

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
        assigned to a variable that later gets printed. A ternary
        (`"a" if cond else "b"`) is the one bounded exception: each branch
        is checked independently by the same plain-string-only rule, so a
        status label like `status = "used" if cond else "UNUSED"` is
        covered, but `status = "used" if cond else compute_status()` still
        leaves the `compute_status()` branch unmarked and exact-matched."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            ids.add(id(node))
        elif isinstance(node, ast.JoinedStr):
            _mark_string_constants(node, ids)
        elif isinstance(node, ast.IfExp):
            _mark_if_plain_string(node.body)
            _mark_if_plain_string(node.orelse)

    # Same-cell `name = [...]`/`name = (...)` literal assignments, so a
    # `for ... in zip(axes, configs):` can resolve `configs` back to the
    # list it was built from a few lines above -- see the one-hop
    # docstring bullet extended below for `zip`/`enumerate`.
    list_literal_assigns: dict[str, ast.List | ast.Tuple] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            list_literal_assigns[node.targets[0].id] = node.value

    def _resolve_list_literal(node: ast.AST) -> ast.List | ast.Tuple | None:
        if isinstance(node, (ast.List, ast.Tuple)):
            return node
        if isinstance(node, ast.Name):
            return list_literal_assigns.get(node.id)
        return None

    # Same-cell `name = {...}` dict-literal assignments, so `for key, val
    # in name.items():` can resolve `name` back to the dict it was built
    # from -- see the one-hop docstring bullet's `.items()` case below.
    dict_literal_assigns: dict[str, ast.Dict] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            dict_literal_assigns[node.targets[0].id] = node.value

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

        def _process_pair(self, target: ast.AST, iterable: ast.AST) -> None:
            """Match a for-loop target against its iterable, after
            unwrapping the two call shapes this course's plotting/labeling
            code commonly loops over: `zip(a, b)` (processed pairwise,
            independently, for each (sub-target, sub-iterable)) and
            `enumerate(x)` (the index is irrelevant, only the value half of
            the (idx, val) target is matched against x's elements
            directly, not nested in a pair)."""
            if isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name):
                if (
                    iterable.func.id == "enumerate"
                    and len(iterable.args) == 1
                    and isinstance(target, ast.Tuple)
                    and len(target.elts) == 2
                ):
                    self._process_pair(target.elts[1], iterable.args[0])
                    return
                if (
                    iterable.func.id == "zip"
                    and isinstance(target, ast.Tuple)
                    and len(target.elts) == len(iterable.args)
                ):
                    for t_elt, it_arg in zip(target.elts, iterable.args):
                        self._process_pair(t_elt, it_arg)
                    return

            if (
                isinstance(iterable, ast.Call)
                and isinstance(iterable.func, ast.Attribute)
                and iterable.func.attr == "items"
                and not iterable.args
                and isinstance(target, ast.Tuple)
                and len(target.elts) == 2
            ):
                key_target = target.elts[0]
                if isinstance(key_target, ast.Name) and key_target.id in presentational_names:
                    dict_literal = (
                        iterable.func.value
                        if isinstance(iterable.func.value, ast.Dict)
                        else dict_literal_assigns.get(
                            iterable.func.value.id
                            if isinstance(iterable.func.value, ast.Name)
                            else ""
                        )
                    )
                    if dict_literal is not None:
                        for key_node in dict_literal.keys:
                            _mark_if_plain_string(key_node)
                return

            resolved = _resolve_list_literal(iterable)
            if resolved is None:
                return
            if isinstance(target, ast.Tuple) and all(
                isinstance(elt, ast.Tuple) for elt in resolved.elts
            ):
                for pos, name_node in enumerate(target.elts):
                    if isinstance(name_node, ast.Name) and name_node.id in presentational_names:
                        for elt in resolved.elts:
                            if pos < len(elt.elts):
                                _mark_if_plain_string(elt.elts[pos])
            elif isinstance(target, ast.Name) and target.id in presentational_names:
                for elt in resolved.elts:
                    _mark_if_plain_string(elt)

        def visit_For(self, node: ast.For) -> None:
            self._process_pair(node.target, node.iter)
            self.generic_visit(node)

    _OneHopVisitor().visit(tree)

    # Fourth pass: string keys of a notebook-wide *display/grouping
    # accumulator* dict (`accumulator_names`, resolved once per notebook
    # by `_collect_accumulator_dict_names` -- see that function and the
    # docstring's dict-accumulator bullet for the qualifying conditions).
    if accumulator_names:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in accumulator_names
                and isinstance(node.value, ast.Dict)
            ):
                for key_node in node.value.keys:
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        ids.add(id(key_node))

        class _DictAccumulatorVisitor(ast.NodeVisitor):
            def visit_Subscript(self, node: ast.Subscript) -> None:
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in accumulator_names
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    ids.add(id(node.slice))
                self.generic_visit(node)

        _DictAccumulatorVisitor().visit(tree)

    if global_function_params:

        class _HelperCallVisitor(ast.NodeVisitor):
            """Third pass: an argument passed to a presentational parameter
            of a notebook-wide, locally defined helper function (resolved in
            `global_function_params`) is presentational too -- see the
            docstring's helper-function bullet."""

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if isinstance(func, ast.Name) and func.id in global_function_params:
                    param_names, presentational_params = global_function_params[func.id]
                    for idx, arg in enumerate(node.args):
                        if idx < len(param_names) and param_names[idx] in presentational_params:
                            _mark_if_plain_string(arg)
                    for kw in node.keywords:
                        if kw.arg in presentational_params:
                            _mark_if_plain_string(kw.value)
                self.generic_visit(node)

        _HelperCallVisitor().visit(tree)

    return ids


def _normalize_ast(node: ast.AST | list | object, presentational_ids: set[int]) -> object:
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


DECIMAL_COMMA_PATTERN = re.compile(r"(?<=\d)\{,\}(?=\d)")
THOUSANDS_THINSPACE_PATTERN = re.compile(r"(?<=\d)\\,(?=\d)")
LATEX_TEXT_COMMAND_PATTERN = re.compile(r"\\text\{[^}]*\}")


def _normalize_latex_for_comparison(expr: str) -> str:
    """Apply the narrow, bounded LaTeX normalizations documented in the
    module docstring: a Polish decimal comma `X{,}Y` becomes `X.Y`, a
    Polish thousands-separator thin space `X\\,YYY` becomes `XYYY` (this
    course's English localization drops the separator rather than
    switching conventions), and every `\\text{...}` command's content
    becomes a placeholder (since translating a prose label inside
    `\\text{...}` while preserving the expression's meaning is this
    course's explicit localization rule, not an error -- see the module
    docstring for the accepted risk)."""
    expr = DECIMAL_COMMA_PATTERN.sub(".", expr)
    expr = THOUSANDS_THINSPACE_PATTERN.sub("", expr)
    expr = LATEX_TEXT_COMMAND_PATTERN.sub(r"\\text{STR}", expr)
    return expr


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
            f"cell count mismatch: PL has {len(pl_cells)} cells, EN has {len(en_cells)} cells"
        )

    pl_code_sources = [c.source for c in pl_cells if c.cell_type == "code"]
    en_code_sources = [c.source for c in en_cells if c.cell_type == "code"]
    pl_global_params = _collect_function_presentational_params(pl_code_sources)
    en_global_params = _collect_function_presentational_params(en_code_sources)
    pl_accumulator_names = _collect_accumulator_dict_names(pl_code_sources)
    en_accumulator_names = _collect_accumulator_dict_names(en_code_sources)

    for i, (pl_cell, en_cell) in enumerate(zip(pl_cells, en_cells)):
        if pl_cell.cell_type != en_cell.cell_type:
            problems.append(
                f"cell {i}: type mismatch (PL={pl_cell.cell_type!r}, EN={en_cell.cell_type!r})"
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

            pl_presentational = _presentational_string_ids(
                pl_tree, pl_global_params, pl_accumulator_names
            )
            en_presentational = _presentational_string_ids(
                en_tree, en_global_params, en_accumulator_names
            )
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

            pl_latex = [
                _normalize_latex_for_comparison(expr) for expr in _extract_latex(pl_cell.source)
            ]
            en_latex = [
                _normalize_latex_for_comparison(expr) for expr in _extract_latex(en_cell.source)
            ]
            if pl_latex != en_latex:
                problems.append(
                    f"cell {i}: LaTeX expressions differ -- PL={pl_latex}, EN={en_latex}"
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
    (
        Path("lectures/02_mlp/examples.ipynb"),
        Path("lectures/02_mlp/examples.en.ipynb"),
    ),
    (
        Path("labs/02_mlp/lab.ipynb"),
        Path("labs/02_mlp/lab.en.ipynb"),
    ),
    (
        Path("lectures/03_backprop/examples.ipynb"),
        Path("lectures/03_backprop/examples.en.ipynb"),
    ),
    (
        Path("labs/03_backprop/lab.ipynb"),
        Path("labs/03_backprop/lab.en.ipynb"),
    ),
    (
        Path("lectures/04_optimization/examples.ipynb"),
        Path("lectures/04_optimization/examples.en.ipynb"),
    ),
    (
        Path("labs/04_optimization/lab.ipynb"),
        Path("labs/04_optimization/lab.en.ipynb"),
    ),
    (
        Path("lectures/05_generalization/examples.ipynb"),
        Path("lectures/05_generalization/examples.en.ipynb"),
    ),
    (
        Path("labs/05_generalization/lab.ipynb"),
        Path("labs/05_generalization/lab.en.ipynb"),
    ),
    (
        Path("lectures/06_cnn/examples.ipynb"),
        Path("lectures/06_cnn/examples.en.ipynb"),
    ),
    (
        Path("labs/06_cnn/lab.ipynb"),
        Path("labs/06_cnn/lab.en.ipynb"),
    ),
    (
        Path("lectures/07_transfer_learning/examples.ipynb"),
        Path("lectures/07_transfer_learning/examples.en.ipynb"),
    ),
    (
        Path("labs/07_transfer_learning/lab.ipynb"),
        Path("labs/07_transfer_learning/lab.en.ipynb"),
    ),
    (
        Path("lectures/08_embeddings/examples.ipynb"),
        Path("lectures/08_embeddings/examples.en.ipynb"),
    ),
    (
        Path("labs/08_embeddings/lab.ipynb"),
        Path("labs/08_embeddings/lab.en.ipynb"),
    ),
    (
        Path("lectures/09_rnn/examples.ipynb"),
        Path("lectures/09_rnn/examples.en.ipynb"),
    ),
    (
        Path("labs/09_rnn/lab.ipynb"),
        Path("labs/09_rnn/lab.en.ipynb"),
    ),
    (
        Path("lectures/11_llm_engineering/examples.ipynb"),
        Path("lectures/11_llm_engineering/examples.en.ipynb"),
    ),
    (
        Path("labs/11_llm_engineering/lab.ipynb"),
        Path("labs/11_llm_engineering/lab.en.ipynb"),
    ),
    (
        Path("lectures/12_autoencoders/examples.ipynb"),
        Path("lectures/12_autoencoders/examples.en.ipynb"),
    ),
    (
        Path("labs/12_autoencoders/lab.ipynb"),
        Path("labs/12_autoencoders/lab.en.ipynb"),
    ),
    (
        Path("lectures/13_diffusion/examples.ipynb"),
        Path("lectures/13_diffusion/examples.en.ipynb"),
    ),
    (
        Path("labs/13_diffusion/lab.ipynb"),
        Path("labs/13_diffusion/lab.en.ipynb"),
    ),
    (
        Path("lectures/14_evaluation_interpretability/examples.ipynb"),
        Path("lectures/14_evaluation_interpretability/examples.en.ipynb"),
    ),
    (
        Path("labs/14_evaluation_interpretability/lab.ipynb"),
        Path("labs/14_evaluation_interpretability/lab.en.ipynb"),
    ),
    (
        Path("lectures/15_production/examples.ipynb"),
        Path("lectures/15_production/examples.en.ipynb"),
    ),
    (
        Path("labs/15_production/lab.ipynb"),
        Path("labs/15_production/lab.en.ipynb"),
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
