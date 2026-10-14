"""Lecture 04 notebook must run end-to-end from a clean kernel.

Automated equivalent of "Restart Kernel -> Run All": a fresh kernel
executes every cell in order, and any raised exception fails this test.
"""

from pathlib import Path

import nbformat
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
