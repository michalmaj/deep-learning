"""Validation tests for module 15's PSI examples."""

import ast
from pathlib import Path

import nbformat
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_MONITORING = REPO_ROOT / "labs" / "15_production" / "monitoring.py"
LECTURE_NOTEBOOKS = (
    REPO_ROOT / "lectures" / "15_production" / "examples.ipynb",
    REPO_ROOT / "lectures" / "15_production" / "examples.en.ipynb",
)


def _load_function(source: str, name: str, filename: str):
    tree = ast.parse(source, filename=filename)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_validate_psi_inputs", name}
    ]
    function = next(
        (node for node in functions if node.name == name),
        None,
    )
    if function is None:
        raise AssertionError(f"{name} not found in {filename}")

    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, filename, "exec"), namespace)
    return namespace[name]


def _load_notebook_function(path: Path, name: str):
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and f"def {name}(" in cell.source:
            return _load_function(cell.source, name, f"{path}:{name}")
    raise AssertionError(f"{name} not found in {path}")


def _load_path_function(path: Path, name: str):
    if path.suffix == ".py":
        return _load_function(path.read_text(), name, str(path))
    return _load_notebook_function(path, name)


@pytest.fixture(params=LECTURE_NOTEBOOKS, ids=("lecture-pl", "lecture-en"))
def psi(request):
    return _load_notebook_function(request.param, "psi")


@pytest.fixture(
    params=(LAB_MONITORING, *LECTURE_NOTEBOOKS), ids=("lab", "lecture-pl", "lecture-en")
)
def validated_psi(request):
    return _load_path_function(request.param, "psi")


def test_psi_keeps_known_result_and_is_finite_for_empty_bins(psi):
    known = psi(torch.tensor([50.0, 50.0]), torch.tensor([70.0, 30.0]))
    extreme = psi(torch.tensor([100.0, 0.0]), torch.tensor([0.0, 100.0]))

    assert known == pytest.approx(0.169460, abs=1e-6)
    assert torch.isfinite(torch.tensor(extreme))
    assert extreme > 1.0


@pytest.mark.parametrize(
    ("ref", "cur", "eps", "message"),
    (
        (torch.tensor([]), torch.tensor([]), 1e-4, "non-empty"),
        (torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), 1e-4, "positive total"),
        (torch.tensor([1.0, 1.0]), torch.tensor([1.0]), 1e-4, "same shape"),
        (torch.tensor([1.0, -1.0]), torch.tensor([1.0, 1.0]), 1e-4, "non-negative"),
        (torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 1.0]), 1e-4, "finite"),
        (torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]), 0.0, "0 < eps < 1"),
        (torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]), float("nan"), "0 < eps < 1"),
    ),
)
def test_psi_rejects_inputs_that_would_hide_data_errors(validated_psi, ref, cur, eps, message):
    with pytest.raises(ValueError, match=message):
        validated_psi(ref, cur, eps=eps)
