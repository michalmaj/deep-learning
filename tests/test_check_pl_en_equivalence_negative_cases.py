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


def test_zip_nested_tuple_loop_label_translation_is_allowed(tmp_path):
    """The one-hop rule's zip-unwrapping extension (added for module 06):
    `for ax, (title, data) in zip(axes, [(...), ...]):` must resolve
    `title` against the second zip argument, even though the target is a
    nested tuple and the iterable is wrapped in zip(), not a bare list."""
    src_pl = (
        'for ax, (title, data) in zip(axes, [("Obraz", 1), ("Odpowiedz", 2)]):\n'
        '    ax.set_title(title)'
    )
    src_en = (
        'for ax, (title, data) in zip(axes, [("Image", 1), ("Response", 2)]):\n'
        '    ax.set_title(title)'
    )
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a zip-paired, nested-tuple plot title must not be reported"


def test_zip_nested_tuple_loop_does_not_hide_a_real_value_change(tmp_path):
    """The zip-unwrapping extension must only exempt the label half of the
    nested tuple, never the paired numeric value."""
    src_pl = (
        'for ax, (title, data) in zip(axes, [("Obraz", 1), ("Odpowiedz", 2)]):\n'
        '    ax.set_title(title)'
    )
    src_en = (
        'for ax, (title, data) in zip(axes, [("Image", 1), ("Response", 99)]):\n'
        '    ax.set_title(title)'
    )
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed numeric value next to a translated title must still be reported"


def test_zip_resolves_same_cell_list_variable_not_just_a_literal(tmp_path):
    """The zip-unwrapping extension must also resolve the second zip
    argument when it is a `name = [...]` variable assigned earlier in the
    same cell (module 06's `configs` pattern), not only an inline list."""
    src_pl = (
        'configs = [("wariant A", 1), ("wariant B", 2)]\n'
        'for ax, (title, n) in zip(axes, configs):\n'
        '    ax.set_title(title)'
    )
    src_en = (
        'configs = [("variant A", 1), ("variant B", 2)]\n'
        'for ax, (title, n) in zip(axes, configs):\n'
        '    ax.set_title(title)'
    )
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a title translated via a same-cell list variable must not be reported"


def test_enumerate_loop_label_translation_is_allowed(tmp_path):
    """The one-hop rule's enumerate-unwrapping extension (added for module
    08's VOCAB demo): `for i, tok in enumerate(VOCAB):` must resolve `tok`
    against VOCAB's own elements directly, ignoring the index half of the
    (idx, val) target."""
    src_pl = 'VOCAB = ["kot", "pies"]\nfor i, tok in enumerate(VOCAB):\n    print(f"{i}: {tok}")'
    src_en = 'VOCAB = ["cat", "dog"]\nfor i, tok in enumerate(VOCAB):\n    print(f"{i}: {tok}")'
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a print-only enumerate() loop label must not be reported"


def test_enumerate_loop_does_not_hide_a_changed_vocabulary_size(tmp_path):
    """The enumerate-unwrapping extension exempts VOCAB's string elements,
    but a genuinely changed list (here: a dropped entry) must still
    surface as a mismatch via the differing cell/numeric structure."""
    src_pl = 'VOCAB = ["kot", "pies", "samochod"]\nfor i, tok in enumerate(VOCAB):\n    print(tok)'
    src_en = 'VOCAB = ["cat", "dog"]\nfor i, tok in enumerate(VOCAB):\n    print(tok)'
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems, "a dropped vocabulary entry must still be reported"


def test_ternary_status_label_translation_is_allowed(tmp_path):
    """The ternary extension to `_mark_if_plain_string` (added for module
    08): a status label assigned via `"a" if cond else "b"` and only ever
    printed must have both branches treated as presentational."""
    src_pl = 'status = "uzyty" if used else "NIEUZYTY"\nprint(status)'
    src_en = 'status = "used" if used else "UNUSED"\nprint(status)'
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a print-only ternary status label must not be reported"


def test_ternary_extension_does_not_hide_a_non_string_branch_change(tmp_path):
    """The ternary extension only marks a branch that IS itself a plain
    string/f-string -- a branch that calls a function must stay exact, so
    a change to which function is called is still caught."""
    src_pl = 'status = "ok" if used else compute_status()\nprint(status)'
    src_en = 'status = "ok" if used else compute_other_status()\nprint(status)'
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed function call in a ternary's non-string branch must be reported"


def test_dict_accumulator_key_translation_is_allowed(tmp_path):
    """The dict-accumulator extension (added for module 07): a notebook-
    wide `results = {}` dict populated via `results[<label>] = ...` in one
    cell and only ever read back via `.items()` in another is a display
    grouping, not a meaningful identifier -- its keys may be translated."""
    pl = _notebook([
        nbf.v4.new_code_cell('results = {}'),
        nbf.v4.new_code_cell(
            'results["od zera"] = 1\n'
            'results["czesciowy fine-tuning"] = 2\n'
            'for name, val in results.items():\n'
            '    print(f"{name}: {val}")'
        ),
    ])
    en = _notebook([
        nbf.v4.new_code_cell('results = {}'),
        nbf.v4.new_code_cell(
            'results["from scratch"] = 1\n'
            'results["partial fine-tuning"] = 2\n'
            'for name, val in results.items():\n'
            '    print(f"{name}: {val}")'
        ),
    ])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a translated display-accumulator dict key must not be reported"


def test_dict_accumulator_does_not_hide_a_changed_value(tmp_path):
    """The dict-accumulator extension only exempts the key strings -- a
    genuinely changed value stored under one of those keys must still be
    reported."""
    pl = _notebook([
        nbf.v4.new_code_cell('results = {}'),
        nbf.v4.new_code_cell(
            'results["od zera"] = 1\n'
            'for name, val in results.items():\n'
            '    print(f"{name}: {val}")'
        ),
    ])
    en = _notebook([
        nbf.v4.new_code_cell('results = {}'),
        nbf.v4.new_code_cell(
            'results["from scratch"] = 99\n'
            'for name, val in results.items():\n'
            '    print(f"{name}: {val}")'
        ),
    ])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed value under a translated accumulator key must still be reported"


def test_dict_accumulator_rule_does_not_exempt_a_real_config_dict(tmp_path):
    """Guard against over-generalizing: a string-keyed dict that is fed
    straight into a call (here: an optimizer-style parameter dict) is
    disqualified from the accumulator exemption, so a changed key name
    (a real bug) still surfaces as a mismatch."""
    pl = _notebook([nbf.v4.new_code_cell(
        'param_groups = {"params": p, "lr": 0.0001}\n'
        'build_optimizer(param_groups)'
    )])
    en = _notebook([nbf.v4.new_code_cell(
        'param_groups = {"params": p, "learning_rate": 0.0001}\n'
        'build_optimizer(param_groups)'
    )])
    problems = _check(tmp_path, pl, en)
    assert problems, "a renamed key in a dict passed to a call must still be reported"


def test_dict_items_loop_label_translation_is_allowed(tmp_path):
    """The one-hop rule's `.items()` extension (added for module 11): a
    dict mapping a presentational category label to actual experimental
    text (e.g. a sentence handed to a tokenizer) may have its label
    translated when the key is only ever used for display."""
    src_pl = (
        'examples = {"angielskie zdanie": "The quick brown fox.", '
        '"polskie zdanie": "Szybki lis."}\n'
        'for name, text in examples.items():\n'
        '    print(name, text)'
    )
    src_en = (
        'examples = {"English sentence": "The quick brown fox.", '
        '"Polish sentence": "Szybki lis."}\n'
        'for name, text in examples.items():\n'
        '    print(name, text)'
    )
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems == [], "a translated dict-items display label must not be reported"


def test_dict_items_loop_does_not_hide_a_changed_experimental_value(tmp_path):
    """The `.items()` extension marks only the key Constants as
    presentational -- the dict's *values* (the actual text fed to a
    tokenizer/model) must still be compared exactly, since translating an
    experimental prompt would change what the experiment measures."""
    src_pl = (
        'examples = {"angielskie zdanie": "The quick brown fox.", '
        '"polskie zdanie": "Szybki lis."}\n'
        'for name, text in examples.items():\n'
        '    print(name, text)'
    )
    src_en = (
        'examples = {"English sentence": "The quick brown fox.", '
        '"Polish sentence": "Szybki brazowy lis."}\n'
        'for name, text in examples.items():\n'
        '    print(name, text)'
    )
    pl = _notebook([nbf.v4.new_code_cell(src_pl)])
    en = _notebook([nbf.v4.new_code_cell(src_en)])
    problems = _check(tmp_path, pl, en)
    assert problems, "a changed experimental sentence under a translated label must be reported"
