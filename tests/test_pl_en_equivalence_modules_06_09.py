"""PL/EN structural equivalence checks for modules 06-09's bilingual
notebooks.

Same pattern as `test_pl_en_equivalence_module_01.py` /
`test_pl_en_equivalence_module_10.py` / `test_pl_en_equivalence_modules_02_05.py`,
consolidated into one file since this batch covers four modules translated
together in one PR.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_pl_en_equivalence import check_notebook_pair  # noqa: E402


def test_lecture_06_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "06_cnn" / "examples.ipynb",
        REPO_ROOT / "lectures" / "06_cnn" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_06_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "06_cnn" / "lab.ipynb",
        REPO_ROOT / "labs" / "06_cnn" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_07_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "07_transfer_learning" / "examples.ipynb",
        REPO_ROOT / "lectures" / "07_transfer_learning" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_07_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "07_transfer_learning" / "lab.ipynb",
        REPO_ROOT / "labs" / "07_transfer_learning" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_08_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "08_embeddings" / "examples.ipynb",
        REPO_ROOT / "lectures" / "08_embeddings" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_08_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "08_embeddings" / "lab.ipynb",
        REPO_ROOT / "labs" / "08_embeddings" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_09_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "09_rnn" / "examples.ipynb",
        REPO_ROOT / "lectures" / "09_rnn" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_09_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "09_rnn" / "lab.ipynb",
        REPO_ROOT / "labs" / "09_rnn" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)
