"""Lecture 15's English notebook (`examples.en.ipynb`) must run end-to-end
from a clean kernel, exactly like its Polish counterpart
(`test_lecture_15_notebook.py`). Downloads Fashion-MNIST, so it is
skipped under CI=true.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "15_production" / "examples.en.ipynb"


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="downloads Fashion-MNIST; run locally before opening a PR",
)
def test_lecture_15_notebook_en_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
