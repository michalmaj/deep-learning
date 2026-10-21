"""Lecture 04 notebook must run end-to-end from a clean kernel.

Automated equivalent of "Restart Kernel -> Run All": a fresh kernel
executes every cell in order, and any raised exception fails this test.
"""

from pathlib import Path

import nbformat
import pytest
import torch
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "04_optimization" / "examples.ipynb"


def test_lecture_04_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


@pytest.mark.parametrize("notebook_name", ["examples.ipynb", "examples.en.ipynb"])
def test_high_momentum_example_converges_over_a_long_enough_horizon(notebook_name):
    """Guard the distinction between a long oscillatory transient and instability."""
    notebook_path = NOTEBOOK_PATH.with_name(notebook_name)
    nb = nbformat.read(notebook_path, as_version=4)
    definition_cell = next(cell.source for cell in nb.cells if "def run(" in cell.source)
    namespace = {"torch": torch}
    exec(definition_cell, namespace)

    _, losses = namespace["run"]("momentum", 0.06, steps=100, momentum=0.9)

    assert losses[20] == pytest.approx(1.9163, abs=1e-4)
    assert losses[40] < 0.01
    assert losses[100] < 0.0001
