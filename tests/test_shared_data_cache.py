"""The course should reuse one repository-level dataset cache."""

import ast
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_15_DIR = REPO_ROOT / "labs" / "15_production"


def _code_source(notebook_path):
    notebook = json.loads(notebook_path.read_text())
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def _execute_cache_setup(source):
    tree = ast.parse(source)
    setup_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names):
            setup_nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            setup_nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"_REPO_ROOT", "DATA_CACHE_DIR"}
            for target in node.targets
        ):
            setup_nodes.append(node)

    namespace = {}
    exec(compile(ast.Module(body=setup_nodes, type_ignores=[]), "<cache-setup>", "exec"), namespace)
    return namespace


def test_dataset_notebooks_use_the_shared_cache_contract():
    notebook_paths = sorted((REPO_ROOT / "lectures").glob("*/*.ipynb"))
    notebook_paths += sorted((REPO_ROOT / "labs").glob("*/*.ipynb"))
    cache_notebooks = []
    legacy_references = []

    for notebook_path in notebook_paths:
        source = _code_source(notebook_path)
        if 'root="data_cache"' in source or 'CACHE_DIR = "data_cache"' in source:
            legacy_references.append(str(notebook_path.relative_to(REPO_ROOT)))
        if any(
            marker in source
            for marker in (
                "root=DATA_CACHE_DIR",
                "CACHE_DIR = DATA_CACHE_DIR",
                "prepare_sms_spam_dataset(DATA_CACHE_DIR)",
            )
        ):
            cache_notebooks.append(notebook_path)
            assert "DEEP_LEARNING_DATA_CACHE" in source
            assert '(parent / "pyproject.toml").is_file()' in source

    assert legacy_references == []
    assert len(cache_notebooks) == 26


def test_notebook_cache_setup_resolves_repo_root_and_override(monkeypatch, tmp_path):
    notebook_path = REPO_ROOT / "labs" / "06_cnn" / "lab.ipynb"
    source = _code_source(notebook_path)
    monkeypatch.chdir(notebook_path.parent)
    monkeypatch.delenv("DEEP_LEARNING_DATA_CACHE", raising=False)

    namespace = _execute_cache_setup(source)
    assert namespace["_REPO_ROOT"] == REPO_ROOT
    assert namespace["DATA_CACHE_DIR"] == REPO_ROOT / "data_cache"

    override = tmp_path / "external-cache"
    monkeypatch.setenv("DEEP_LEARNING_DATA_CACHE", str(override))
    namespace = _execute_cache_setup(source)
    assert namespace["DATA_CACHE_DIR"] == override


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
