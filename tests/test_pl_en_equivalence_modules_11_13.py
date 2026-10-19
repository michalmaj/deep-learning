"""PL/EN structural equivalence checks for modules 11-13's bilingual
notebooks.

Same pattern as `test_pl_en_equivalence_module_01.py` /
`test_pl_en_equivalence_module_10.py` / `test_pl_en_equivalence_modules_02_05.py` /
`test_pl_en_equivalence_modules_06_09.py`, consolidated into one file since
this batch covers three modules translated together in one PR.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_pl_en_equivalence import check_notebook_pair  # noqa: E402


def test_lecture_11_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "11_llm_engineering" / "examples.ipynb",
        REPO_ROOT / "lectures" / "11_llm_engineering" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_11_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "11_llm_engineering" / "lab.ipynb",
        REPO_ROOT / "labs" / "11_llm_engineering" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_12_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "12_autoencoders" / "examples.ipynb",
        REPO_ROOT / "lectures" / "12_autoencoders" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_12_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "12_autoencoders" / "lab.ipynb",
        REPO_ROOT / "labs" / "12_autoencoders" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lecture_13_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "13_diffusion" / "examples.ipynb",
        REPO_ROOT / "lectures" / "13_diffusion" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_13_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "13_diffusion" / "lab.ipynb",
        REPO_ROOT / "labs" / "13_diffusion" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)
