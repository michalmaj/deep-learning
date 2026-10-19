"""Lecture 13's English notebook (`examples.en.ipynb`) must run end-to-end
from a clean kernel, exactly like its Polish counterpart
(`test_lecture_13_notebook.py`). Only locally-generated synthetic 2D data
(no download, no network), so -- same as the Polish test -- this runs
unconditionally.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "13_diffusion" / "examples.en.ipynb"


def test_lecture_13_notebook_en_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
