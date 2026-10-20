"""Lecture 12's English notebook (`examples.en.ipynb`) must run end-to-end
from a clean kernel, exactly like its Polish counterpart
(`test_lecture_12_notebook.py`). Downloads KMNIST, so it is marked `network`
and runs in scheduled network CI.
"""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "12_autoencoders" / "examples.en.ipynb"


@pytest.mark.network
def test_lecture_12_notebook_en_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
