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


def test_decimal_comma_localization_alone_is_allowed(tmp_path):
    """Localizing a Polish decimal comma to an English decimal point in
    LaTeX (lecture.md's `{,}` escape) is required, not a translation
    error."""
    pl = _notebook([nbf.v4.new_markdown_cell(r"The loss reaches $\ln 2 \approx 0{,}69$ here.")])
    en = _notebook([nbf.v4.new_markdown_cell(r"The loss reaches $\ln 2 \approx 0.69$ here.")])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a localized decimal comma must not be reported"


def test_decimal_comma_normalization_does_not_hide_a_real_value_change(tmp_path):
    """The decimal-comma normalization must only touch digit{,}digit, never
    mask an actual change to the numeric value of a formula."""
    pl = _notebook([nbf.v4.new_markdown_cell(r"The loss reaches $\ln 2 \approx 0{,}69$ here.")])
    en = _notebook([nbf.v4.new_markdown_cell(r"The loss reaches $\ln 2 \approx 0.75$ here.")])
    problems = _check(tmp_path, pl, en)
    assert problems, "a genuinely changed numeric value must still be reported"


def test_decimal_comma_normalization_does_not_touch_bare_commas(tmp_path):
    """A bare comma (coordinate pair, argument separator, list) must never
    be treated as a decimal separator -- only digit{,}digit, literally
    wrapped in the LaTeX escape, is normalized."""
    pl = _notebook([nbf.v4.new_markdown_cell(r"The point $(0,0)$ is the origin.")])
    en = _notebook([nbf.v4.new_markdown_cell(r"The point $(0,1)$ is the origin.")])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed coordinate must still be reported, not normalized away"


def test_thousands_separator_dropped_in_english_is_allowed(tmp_path):
    """This course's English localization drops the Polish thin-space
    thousands separator entirely (2\\,000 -> 2000) rather than switching
    to the English comma convention, to avoid colliding with the
    decimal-comma normalization."""
    pl = _notebook([nbf.v4.new_markdown_cell(r"We generate $N=2\,000$ examples.")])
    en = _notebook([nbf.v4.new_markdown_cell(r"We generate $N=2000$ examples.")])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "dropping the thousands separator in English must not be reported"


def test_thousands_separator_normalization_does_not_hide_a_value_change(tmp_path):
    pl = _notebook([nbf.v4.new_markdown_cell(r"We generate $N=2\,000$ examples.")])
    en = _notebook([nbf.v4.new_markdown_cell(r"We generate $N=3000$ examples.")])
    problems = _check(tmp_path, pl, en)
    assert problems, "a genuinely changed sample count must still be reported"


def test_text_command_label_translation_alone_is_allowed(tmp_path):
    """This course's localization rule explicitly requires translating a
    prose label inside \\text{...} while preserving the expression's
    meaning (e.g. machine-epsilon's subscript) -- the content is expected
    to differ, only the rest of the equation must still match."""
    pl = _notebook([
        nbf.v4.new_markdown_cell(r"Rounding error, order $\varepsilon_{\text{maszyny}}/h$.")
    ])
    en = _notebook([
        nbf.v4.new_markdown_cell(r"Rounding error, order $\varepsilon_{\text{machine}}/h$.")
    ])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a translated \\text{...} prose label must not be reported"


def test_text_command_normalization_does_not_hide_an_operator_change(tmp_path):
    """The \\text{...} normalization only touches text-command content --
    a real change to the surrounding equation (here: + instead of /) must
    still be reported."""
    pl = _notebook([
        nbf.v4.new_markdown_cell(r"Rounding error, order $\varepsilon_{\text{maszyny}}/h$.")
    ])
    en = _notebook([
        nbf.v4.new_markdown_cell(r"Rounding error, order $\varepsilon_{\text{machine}}+h$.")
    ])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed operator next to a translated text label must still be reported"


def test_helper_function_presentational_label_translation_is_allowed(tmp_path):
    """A locally defined helper function's parameter that is used *only*
    for display inside the helper's own body (here: `report`'s `name`,
    called from a cell AFTER the one that defines it) is presentational --
    this is the cross-cell generalization of the one-hop rule, needed
    because module 02's `report(name, model, loss)` is defined in one cell
    and called from a later one."""
    pl = _notebook([
        nbf.v4.new_code_cell(
            'def report(name, value):\n'
            '    print(f"{name}: {value:.3f}")'
        ),
        nbf.v4.new_code_cell('report("Liniowy", 0.28)\nreport("MLP", 0.01)'),
    ])
    en = _notebook([
        nbf.v4.new_code_cell(
            'def report(name, value):\n'
            '    print(f"{name}: {value:.3f}")'
        ),
        nbf.v4.new_code_cell('report("Linear", 0.28)\nreport("MLP", 0.01)'),
    ])
    problems = _check(tmp_path, pl, en)
    assert problems == [], (
        "a translated label passed to a display-only helper parameter must not be reported"
    )


def test_helper_function_rule_does_not_exempt_a_computationally_used_parameter(tmp_path):
    """Guard against over-generalizing: if a helper's parameter is used for
    anything beyond display (here: `key` indexes a dict), an argument
    passed to it must still be compared exactly."""
    pl = _notebook([
        nbf.v4.new_code_cell(
            'def lookup(key, table):\n'
            '    print(f"looking up {key}")\n'
            '    return table[key]'
        ),
        nbf.v4.new_code_cell('lookup("cpu", {"cpu": 1, "cuda": 2})'),
    ])
    en = _notebook([
        nbf.v4.new_code_cell(
            'def lookup(key, table):\n'
            '    print(f"looking up {key}")\n'
            '    return table[key]'
        ),
        nbf.v4.new_code_cell('lookup("device", {"cpu": 1, "cuda": 2})'),
    ])
    problems = _check(tmp_path, pl, en)
    assert problems, "a parameter used for more than display must still be compared exactly"


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
