"""The English lab 15 scaffold (`lab.en.ipynb`) must run, exactly like its
Polish counterpart (`test_lab_15_scaffold.py`). Needs a real EuroSAT +
ResNet18 artifact, so it is skipped under CI=true.

`lab.en.ipynb` imports the SAME `app/`, `benchmark.py`, `monitoring.py`
modules as `lab.ipynb` -- those files are shared, English-only code (not
duplicated per language), already covered by `test_lab_15_scaffold.py`'s
given/TODO mechanics tests. This file only verifies the English notebook
itself runs to the same first TODO.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = REPO_ROOT / "labs" / "15_production"
NOTEBOOK_PATH = LAB_DIR / "lab.en.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="needs a real EuroSAT + ResNet18 artifact; run locally before opening a PR",
)
def test_lab_15_scaffold_en_runs_until_first_todo():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(LAB_DIR)}},
    )
    client.execute()

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [(i, _cell_error(c)) for i, c in enumerate(code_cells)]
    first_error_index, first_error = next(e for e in errors if e[1] is not None)

    for i, error in errors:
        if i < first_error_index:
            assert error is None, f"scaffold cell {i} failed unexpectedly: {error}"

    assert first_error["ename"] == "NotImplementedError", (
        f"expected the first failure to be the artifact-consistency TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )
