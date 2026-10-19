"""Controlled negative/positive cases for `scripts/check_pl_en_equivalence.py`.

The equivalence checker is itself a safety mechanism, so it needs its own
test coverage proving it actually detects the kinds of drift it claims to
detect -- passing on two genuinely equivalent notebooks is necessary but not
sufficient evidence. Each case here builds a minimal synthetic PL/EN
notebook pair, in memory, and asserts whether the checker reports a problem.

See `scripts/check_pl_en_equivalence.py`'s module docstring for the exact,
narrow set of contexts in which a string literal is allowed to differ
("String-literal protection") and for the checker's documented limitations.
"""

import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_pl_en_equivalence import check_notebook_pair  # noqa: E402


def _notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    return nb


def _write(tmp_path, name, nb):
    path = tmp_path / name
    nbf.write(nb, path)
    return path


def _check(tmp_path, pl, en):
    return check_notebook_pair(_write(tmp_path, "pl.ipynb", pl), _write(tmp_path, "en.ipynb", en))


def test_seed_change_is_detected(tmp_path):
    pl = _notebook([nbf.v4.new_code_cell("torch.manual_seed(0)\nx = 1")])
    en = _notebook([nbf.v4.new_code_cell("torch.manual_seed(99)\nx = 1")])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed seed must be reported"


def test_hyperparameter_change_is_detected(tmp_path):
    pl = _notebook([nbf.v4.new_code_cell("d_model, n_heads = 8, 2")])
    en = _notebook([nbf.v4.new_code_cell("d_model, n_heads = 16, 2")])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed hyperparameter constant must be reported"


def test_removed_equation_is_detected(tmp_path):
    pl = _notebook([nbf.v4.new_markdown_cell("The formula is $d_k=4$ here.")])
    en = _notebook([nbf.v4.new_markdown_cell("The formula is described here.")])
    problems = _check(tmp_path, pl, en)
    assert problems, "a dropped equation must be reported"


def test_mask_logic_change_is_detected(tmp_path):
    pl = _notebook([nbf.v4.new_code_cell("allow_mask = torch.tensor([True, True, False])")])
    en = _notebook([nbf.v4.new_code_cell("allow_mask = torch.tensor([False, False, True])")])
    problems = _check(tmp_path, pl, en)
    assert problems, "an inverted boolean mask must be reported"


def test_computationally_meaningful_string_change_is_detected(tmp_path):
    """A string that is NOT in a recognized presentational context (here:
    the device argument to torch.device, not print/title/label/docstring)
    must be compared exactly -- changing it must be reported, even though it
    is "just a string"."""
    pl = _notebook([nbf.v4.new_code_cell('device = torch.device("cpu")')])
    en = _notebook([nbf.v4.new_code_cell('device = torch.device("meta")')])
    problems = _check(tmp_path, pl, en)
    assert problems, "a computationally meaningful string literal change must be reported"


def test_plot_title_translation_alone_is_allowed(tmp_path):
    pl = _notebook([nbf.v4.new_code_cell('plt.title("Krzywa strat")\nx = 1')])
    en = _notebook([nbf.v4.new_code_cell('plt.title("Loss curve")\nx = 1')])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a translated plot title must not be reported"


def test_todo_message_translation_alone_is_allowed(tmp_path):
    pl = _notebook(
        [nbf.v4.new_code_cell('def f(x):\n    raise NotImplementedError("TODO: zaimplementuj f")')]
    )
    en = _notebook(
        [nbf.v4.new_code_cell('def f(x):\n    raise NotImplementedError("TODO: implement f")')]
    )
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a translated NotImplementedError message must not be reported"


def test_docstring_translation_alone_is_allowed(tmp_path):
    pl = _notebook([nbf.v4.new_code_cell('def f(x):\n    """Zwraca x."""\n    return x')])
    en = _notebook([nbf.v4.new_code_cell('def f(x):\n    """Returns x."""\n    return x')])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a translated docstring must not be reported"


def test_legend_label_translation_alone_is_allowed(tmp_path):
    pl = _notebook([nbf.v4.new_code_cell('plt.plot(x, label="dane")')])
    en = _notebook([nbf.v4.new_code_cell('plt.plot(x, label="data")')])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a translated legend label=... keyword must not be reported"


def test_print_only_loop_label_translation_is_allowed(tmp_path):
    """The one-hop back-reference case: a string bound via for-loop
    tuple-unpacking, used only inside a later print(f"...") call."""
    src_pl = (
        'for name, flag in [("z PE", True), ("bez PE", False)]:\n'
        '    print(f"{name}: {flag}")'
    )
    src_en = (
        'for name, flag in [("with PE", True), ("without PE", False)]:\n'
        '    print(f"{name}: {flag}")'
    )
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a print-only for-loop label must not be reported"


def test_print_only_loop_label_does_not_hide_a_real_flag_change(tmp_path):
    """The one-hop rule must only exempt the *label* string, never the
    paired boolean/flag value in the same tuple position."""
    src_pl = (
        'for name, flag in [("z PE", True), ("bez PE", False)]:\n'
        '    print(f"{name}: {flag}")'
    )
    src_en = (
        'for name, flag in [("with PE", False), ("without PE", False)]:\n'
        '    print(f"{name}: {flag}")'
    )
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed boolean flag next to a translated label must still be reported"


def test_variable_assigned_header_string_translation_is_allowed(tmp_path):
    """The one-hop back-reference case for a direct assignment: an f-string
    header assigned to a variable before being printed."""
    pl = _notebook(
        [nbf.v4.new_code_cell('header = f"{\'warunek\':10s}"\nprint(header)')]
    )
    en = _notebook(
        [nbf.v4.new_code_cell('header = f"{\'condition\':10s}"\nprint(header)')]
    )
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a print-only variable-assigned header string must not be reported"


def test_one_hop_rule_does_not_swallow_unrelated_nested_string(tmp_path):
    """Regression guard for the bug found while building this script: a
    variable later printed must not make an UNRELATED string nested inside
    its right-hand side (e.g. the device name inside torch.device(...))
    exempt just because the variable itself is also printed."""
    pl = _notebook(
        [nbf.v4.new_code_cell('device = torch.device("cpu")\nprint(f"Using device: {device}")')]
    )
    en = _notebook(
        [nbf.v4.new_code_cell('device = torch.device("meta")\nprint(f"Using device: {device}")')]
    )
    problems = _check(tmp_path, pl, en)
    assert problems, "torch.device(...)'s argument must stay exact even though `device` is printed"
