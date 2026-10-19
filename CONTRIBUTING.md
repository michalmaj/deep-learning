# Maintaining this bilingual course

This file is for anyone editing course content, not for students. It
documents the conventions behind the Polish/English bilingual structure so
a future change doesn't quietly break parity between the two languages.
There is no `CONTRIBUTING.pl.md` -- like `data/README.md` and
`third_party/README.md`, this is a maintainer-facing document, not student
content, and follows the course's existing precedent of leaving
maintainer-only docs in English only.

## File convention

Every module (`lectures/NN_name/`, `labs/NN_name/`) has four pairs:

| Role | Polish | English |
|---|---|---|
| Lecture text | `lecture.md` | `lecture.en.md` |
| Lecture notebook | `examples.ipynb` | `examples.en.ipynb` |
| Lab instructions | `README.pl.md` | `README.md` |
| Lab notebook | `lab.ipynb` | `lab.en.ipynb` |

The lab README pair is deliberately inverted (`README.md` is English, not
Polish) because GitHub renders `README.md` by default, and English is this
repository's default language. Every one of these eight files carries a
one-line language switch at the very top (`Language: **English
(default)** | [Polski](...)` / `Język: [English](...) | **Polski**`),
pointing at its counterpart. When you add or edit one of these files, add
or update that line.

There is no `en/`/`pl/` subtree and no second repository -- both
languages live side by side in the same module directory.

**Exception**: a lab whose scaffold is a real application, not just a
notebook (currently only module 15: `app/`, `benchmark.py`,
`monitoring.py`, `model_colleague.py`, `prepare_artifact.py`) keeps those
files as **shared, single-language code**. They are already English
(identifiers, docstrings, comments) because they are the service's actual
API/data contract, not instructional prose -- `lab.ipynb` and
`lab.en.ipynb` both import from the exact same files. Don't duplicate them
per language; if you find Polish text in one of them, that's a bug (an
English-only student would see it too), fix it in place.

## Editing an existing module

1. Change the Polish source first (it's the one with the open PR history
   and review context for this course's content).
2. Make the identical substantive change in the English file: same math,
   same seeds/hyperparameters/architecture, same TODOs and grading
   criteria, same numeric results. Only the prose, comments, and
   presentational strings differ.
3. Re-run `scripts/check_pl_en_equivalence.py` (see below) before
   executing either notebook -- it catches most structural drift (a
   dropped equation, a changed seed, a mismatched TODO) in seconds,
   before you spend minutes re-running a notebook that's already wrong.
4. Execute both notebooks from a clean kernel and compare. Numbers should
   match exactly, except measured CPU timings (train/eval duration,
   latency, throughput), which naturally vary run to run.

## Adding a new module

Follow the same four-pair convention above from the start. Add both
notebook pairs to `MODULE_NOTEBOOK_PAIRS` in
`scripts/check_pl_en_equivalence.py` (one tuple each for the lecture pair
and the lab pair) -- this is the only change the checker itself needs for
a new module; its checking logic is module-agnostic.

## Three kinds of strings

When writing or reviewing a translation, every string literal in a
notebook or lab script falls into one of three categories:

1. **Presentational** -- instructions, comments, print/plot messages,
   TODO wording, docstrings, student-facing questions. Translate freely.
2. **Computational / contract** -- dict keys, config names, model IDs,
   API endpoint paths, JSON response field names, class-name mappings,
   checkpoint/metadata field names. These must stay byte-identical
   between languages, because code (or a client, or a test) depends on
   the exact string, not its meaning.
3. **Experimental** -- natural-language text that is itself the model's
   input (a prompt, a sentence fed to a tokenizer, a few-shot example).
   Translating it would change what the experiment measures, even though
   it reads like ordinary prose. If a Polish source intentionally uses
   such a string as experimental data (e.g. module 11's PL/EN
   tokenizer-comparison sentence), keep it byte-identical in the English
   notebook too, and add an English explanation near it (not a
   translation of the string itself) saying why it's left in Polish and
   what it says.

## The equivalence checker

```bash
uv run python scripts/check_pl_en_equivalence.py
```

Verifies every registered PL/EN notebook pair is structurally equivalent:
same cell count and type sequence, same TODO functions raising
`NotImplementedError`, same LaTeX expressions in markdown cells (in
order), and an AST-level match on code cells where every identifier,
operator, numeric/bool/None constant, and control-flow node must be
identical -- string literals are exempted from the exact-match requirement
only in the narrow, enumerated contexts documented in the script's own
"String-literal protection" docstring section (print/plot-label
arguments, `NotImplementedError`/`assert` messages, docstrings, a few
narrowly-scoped same-notebook back-reference patterns for display-only
dicts and loops). Read that section before deciding a mismatch is a false
positive -- it usually isn't.

If you hit a genuine new false positive (a presentational string the
checker doesn't yet recognize as such), extend the checker with the
narrowest mechanism that covers your specific case, and add a test to
`tests/test_check_pl_en_equivalence_negative_cases.py` proving the new
mechanism still catches a *real* change in the same position (not just
that it allows your translation). Never add a broad exception keyed on a
variable or parameter *name* (e.g. "any string argument called `prompt`
is exempt") -- that silently stops protecting every other call site using
that name. A rejected narrowing, considered during this project: allowing
a bare `dict_name["literal key"]` read anywhere in the notebook once the
dict qualifies as a display accumulator -- this would also have exempted
a genuine config dict read by key (e.g. `hyperparams["lr"]`) from
detecting a renamed key, so it was not added; a handful of old,
intentionally-untranslated dict keys (module 05's three model-size
labels) were left as a documented, low-impact trade-off instead.

Run `uv run pytest tests/test_check_pl_en_equivalence_negative_cases.py`
after any change to the checker -- every mechanism it implements has at
least one test proving it catches a real change, not just that it allows
a translation.

## Private solutions

Full lab solutions live only in `decisions/solutions/<module>/`, which is
entirely git-ignored (confirm with `git ls-files decisions/` -- it must
print nothing). Never publish a TODO's actual implementation in
`labs/*/lab.ipynb` or `lab.en.ipynb`, and never let a solution file's
content leak into a commit message, a test, or the public scaffold's
comments. A solution's own `.py`/`.ipynb` files inherit the shared-code
exception above: module 15's private solution reuses the same
already-English `app/`, `benchmark.py`, etc. it built from, not a
duplicated pair.

## Validation before opening a PR

```bash
uv run ruff check .
CI=true uv run pytest          # fast, network-free subset
uv run pytest                  # full suite, needs downloads
uv run python scripts/verify_data.py
uv run python scripts/check_pl_en_equivalence.py
git ls-files decisions/        # must print nothing
git log <base>..HEAD --format=%B | grep -i "claude\|anthropic"   # must print nothing
```

Also check, by hand or with a short script: every relative Markdown/
notebook link in changed files resolves to a file that exists, and
`git diff --stat` doesn't include a stray dataset/checkpoint/cache
directory.

## No AI attribution

This repository's commits, PR titles/descriptions, docs, code, and
notebooks never carry AI-attribution trailers or mentions, regardless of
what tooling produced a given change. This is a standing project rule,
not a per-PR decision.

## Course readiness, as of this audit

All 15 modules have a complete English translation, all 30 PL/EN notebook
pairs pass the equivalence checker, and the full test suite is green in
both the CI-safe and full-download configurations (see the dated review
in `decisions/reviews/` for the exact run this covers -- that file is
local-only, so ask whoever ran the audit if you need the numbers and
don't have local access). What this localization pass does **not**
claim:

- That the English prose reads as naturally as the Polish original
  everywhere -- the equivalence checker proves structural/numeric parity,
  not translation quality. Treat a module as provisionally reviewed, not
  final, until someone has read it end-to-end as a student would.
- That every terminology choice is cross-checked against every other
  module for consistency beyond the explicit symbol-collision notes
  already in each `lecture.md`/`lecture.en.md`.
- That the course has been taught from, in either language, by anyone
  other than the people who built it.

Before relying on this course to actually teach a cohort, at minimum: have
a native-or-fluent English speaker read through the lecture texts once;
dry-run at least one full lab in each language under the stated time
budget; and decide a process for keeping both languages in sync when a
future correction is needed (the simplest version: whoever finds a bug
fixes it in both languages in the same PR, as this project has done
throughout -- see "Editing an existing module" above).
