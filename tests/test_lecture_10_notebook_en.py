"""Lecture 10's English notebook (`examples.en.ipynb`) must run end-to-end
from a clean kernel, exactly like its Polish counterpart
(`test_lecture_10_notebook.py`).

This is the automated equivalent of "Restart Kernel -> Run All": a fresh
kernel executes every cell in order, and any raised exception (including
the mask-semantics and manual-attention assertions inside the notebook)
fails this test.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "10_attention" / "examples.en.ipynb"


def test_lecture_10_notebook_en_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
