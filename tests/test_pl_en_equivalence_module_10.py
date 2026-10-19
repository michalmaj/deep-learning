"""PL/EN structural equivalence checks for module 10's bilingual notebooks.

Same pattern as `test_pl_en_equivalence_module_01.py`: a static,
network-free AST/regex check (`scripts/check_pl_en_equivalence.py`) run
against module 10's two notebook pairs. Module 10 has more string literals
inside presentational calls (f-string table headers, loop labels) than
module 01 did, which is exactly what exercises the script's string-literal
protection logic -- see that script's module docstring for what counts as
presentational and what does not.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_pl_en_equivalence import check_notebook_pair  # noqa: E402


def test_lecture_10_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "lectures" / "10_attention" / "examples.ipynb",
        REPO_ROOT / "lectures" / "10_attention" / "examples.en.ipynb",
    )
    assert problems == [], "\n".join(problems)


def test_lab_10_notebooks_are_structurally_equivalent():
    problems = check_notebook_pair(
        REPO_ROOT / "labs" / "10_attention" / "lab.ipynb",
        REPO_ROOT / "labs" / "10_attention" / "lab.en.ipynb",
    )
    assert problems == [], "\n".join(problems)
