"""The lab 07 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_0{1..6}_scaffold.py: the public notebook has
intentionally unfinished `# TODO` cells that raise `NotImplementedError`
until a student fills them in. This test executes the notebook
permissively, then checks that every cell before the first student TODO
ran with no error, and that the first error is exactly the expected
`NotImplementedError` from the first TODO (building the preprocessing
transform).

Like tests/test_lecture_07_notebook.py, the full execution needs to
download EuroSAT + ResNet18 weights, so it is skipped under CI=true.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "07_transfer_learning" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="downloads EuroSAT + ResNet18 weights; run locally before opening a PR",
)
def test_lab_07_scaffold_runs_until_first_todo():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [(i, _cell_error(c)) for i, c in enumerate(code_cells)]
    first_error_index, first_error = next(e for e in errors if e[1] is not None)

    for i, error in errors:
        if i < first_error_index:
            assert error is None, f"scaffold cell {i} failed unexpectedly: {error}"

    assert first_error["ename"] == "NotImplementedError", (
        f"expected the first failure to be the build_preprocess TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )
