"""Lecture 03 notebook must run end-to-end from a clean kernel.

Automated equivalent of "Restart Kernel -> Run All": a fresh kernel
executes every cell in order, and any raised exception (including every
`assert` that cross-checks manual/autograd/finite-difference gradients
and the gradient-accumulation/detach() checks) fails this test.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "03_backprop" / "examples.ipynb"


def test_lecture_03_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
