"""PL/EN structural equivalence checks for modules 02-05's bilingual
notebooks.

Same pattern as `test_pl_en_equivalence_module_01.py` /
`test_pl_en_equivalence_module_10.py`, consolidated into one file since
this batch covers four modules translated together in one PR.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_pl_en_equivalence import check_notebook_pair  # noqa: E402


def test_lecture_02_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "02_mlp" / "examples.ipynb",
        REPO_ROOT / "lectures" / "02_mlp" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_02_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "02_mlp" / "lab.ipynb",
        REPO_ROOT / "labs" / "02_mlp" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_03_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "03_backprop" / "examples.ipynb",
        REPO_ROOT / "lectures" / "03_backprop" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_03_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "03_backprop" / "lab.ipynb",
        REPO_ROOT / "labs" / "03_backprop" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_04_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "04_optimization" / "examples.ipynb",
        REPO_ROOT / "lectures" / "04_optimization" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_04_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "04_optimization" / "lab.ipynb",
        REPO_ROOT / "labs" / "04_optimization" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_05_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "05_generalization" / "examples.ipynb",
        REPO_ROOT / "lectures" / "05_generalization" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_05_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "05_generalization" / "lab.ipynb",
        REPO_ROOT / "labs" / "05_generalization" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)
