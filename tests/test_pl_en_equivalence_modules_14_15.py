"""PL/EN structural equivalence checks for modules 14-15's bilingual
notebooks -- the last batch of the course's English localization.

Same pattern as `test_pl_en_equivalence_module_01.py` /
`test_pl_en_equivalence_module_10.py` / `test_pl_en_equivalence_modules_02_05.py` /
`test_pl_en_equivalence_modules_06_09.py` / `test_pl_en_equivalence_modules_11_13.py`,
consolidated into one file since this batch covers both modules translated
together in one PR. After this batch, `MODULE_NOTEBOOK_PAIRS` covers all
30 PL/EN notebook pairs across modules 01-15.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_pl_en_equivalence import (  # noqa: E402
    MODULE_NOTEBOOK_PAIRS,
    check_notebook_pair,
)


def test_lecture_14_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "14_evaluation_interpretability" / "examples.ipynb",
        REPO_ROOT / "lectures" / "14_evaluation_interpretability" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_14_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "14_evaluation_interpretability" / "lab.ipynb",
        REPO_ROOT / "labs" / "14_evaluation_interpretability" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_15_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "15_production" / "examples.ipynb",
        REPO_ROOT / "lectures" / "15_production" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_15_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "15_production" / "lab.ipynb",
        REPO_ROOT / "labs" / "15_production" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_module_notebook_pairs_covers_all_15_modules():
    """After this batch, the checker should cover all 30 PL/EN notebook
    pairs (one lecture + one lab per module, modules 01-15)."""
    assert len(MODULE_NOTEBOOK_PAIRS) == 30
