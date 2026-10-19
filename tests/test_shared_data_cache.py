"""The course should reuse one repository-level dataset cache."""

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_15_DIR = REPO_ROOT / "labs" / "15_production"


def _code_source(notebook_path):
    notebook = json.loads(notebook_path.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_dataset_notebooks_use_the_shared_cache_contract():
    notebook_paths = sorted((REPO_ROOT / "lectures").glob("*/*.ipynb"))
    notebook_paths += sorted((REPO_ROOT / "labs").glob("*/*.ipynb"))
    cache_notebooks = []
    legacy_references = []

    for notebook_path in notebook_paths:
        source = _code_source(notebook_path)
        if 'root="data_cache"' in source or 'CACHE_DIR = "data_cache"' in source:
            legacy_references.append(str(notebook_path.relative_to(REPO_ROOT)))
        if "root=DATA_CACHE_DIR" in source or "CACHE_DIR = DATA_CACHE_DIR" in source:
            cache_notebooks.append(notebook_path)
            assert "DEEP_LEARNING_DATA_CACHE" in source
            assert '(parent / "pyproject.toml").is_file()' in source

    assert legacy_references == []
    assert len(cache_notebooks) == 26


def test_prepare_artifact_cache_precedence(monkeypatch, tmp_path):
    sys.path.insert(0, str(LAB_15_DIR))
    try:
        import prepare_artifact

        monkeypatch.delenv("MODULE15_DATA_ROOT", raising=False)
        monkeypatch.delenv("DEEP_LEARNING_DATA_CACHE", raising=False)
        module = importlib.reload(prepare_artifact)
        assert module.DATA_ROOT == REPO_ROOT / "data_cache"

        shared_override = tmp_path / "shared"
        monkeypatch.setenv("DEEP_LEARNING_DATA_CACHE", str(shared_override))
        module = importlib.reload(prepare_artifact)
        assert module.DATA_ROOT == shared_override

        module_override = tmp_path / "module-15"
        monkeypatch.setenv("MODULE15_DATA_ROOT", str(module_override))
        module = importlib.reload(prepare_artifact)
        assert module.DATA_ROOT == module_override
    finally:
        sys.path.remove(str(LAB_15_DIR))
