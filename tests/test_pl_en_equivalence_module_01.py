"""PL/EN structural equivalence checks for module 01's bilingual notebooks.

This runs the pilot equivalence tool (`scripts/check_pl_en_equivalence.py`)
against module 01's two notebook pairs. It is a static, network-free, AST
and regex based check -- it does not execute any notebook and does not
judge translation quality. It exists to catch an accidental computational
change (seed, hyperparameter, data size, equation) introduced while editing
only one language's copy of a notebook. See the script's module docstring
for the full list of what it checks and its documented limitations.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_pl_en_equivalence import check_notebook_pair  # noqa: E402


def test_lecture_01_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "01_intro" / "examples.ipynb",
        REPO_ROOT / "lectures" / "01_intro" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_01_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "01_intro" / "lab.ipynb",
        REPO_ROOT / "labs" / "01_intro" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)
