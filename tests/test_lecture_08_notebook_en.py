"""Lecture 08's English notebook (`examples.en.ipynb`) must run end-to-end
from a clean kernel, exactly like its Polish counterpart
(`test_lecture_08_notebook.py`). Reuses the Caltech101/ResNet18 cache from
module 07, so it is skipped under CI=true.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "08_embeddings" / "examples.en.ipynb"


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="downloads Caltech101 + ResNet18 weights; run locally before opening a PR",
)
def test_lecture_08_notebook_en_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
