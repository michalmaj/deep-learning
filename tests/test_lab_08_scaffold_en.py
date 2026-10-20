"""The English lab 08 scaffold (`lab.en.ipynb`) must run, exactly like its
Polish counterpart (`test_lab_08_scaffold.py`). Downloads EuroSAT and
ResNet18 pretrained weights, so it is skipped under CI=true.
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "08_embeddings" / "lab.en.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


@pytest.mark.network
def test_lab_08_scaffold_en_runs_until_first_todo():
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
        f"expected the first failure to be the gallery/query disjointness TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )
