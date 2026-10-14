"""The lab 03 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_01_scaffold.py and test_lab_02_scaffold.py:
the public notebook has intentionally unfinished `# TODO` cells that raise
`NotImplementedError` until a student fills them in. This test executes
the notebook permissively, then checks that every cell before the first
student TODO ran with no error, and that the first error is exactly the
expected `NotImplementedError` from the first TODO (`forward`).
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "03_backprop" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


def test_lab_03_scaffold_runs_until_first_todo():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
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
        f"expected the first failure to be the forward TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )
