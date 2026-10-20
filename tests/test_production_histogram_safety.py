"""Boundary tests for module 15's monitoring histograms."""

import ast
from pathlib import Path

import nbformat
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
IMPLEMENTATIONS = (
    REPO_ROOT / "labs" / "15_production" / "monitoring.py",
    REPO_ROOT / "lectures" / "15_production" / "examples.ipynb",
    REPO_ROOT / "lectures" / "15_production" / "examples.en.ipynb",
)


def _load_histogram_counts(path: Path):
    if path.suffix == ".py":
        sources = (path.read_text(),)
    else:
        notebook = nbformat.read(path, as_version=4)
        sources = (cell.source for cell in notebook.cells if cell.cell_type == "code")

    for source in sources:
        tree = ast.parse(source, filename=str(path))
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "histogram_counts"
            ),
            None,
        )
        if function is not None:
            module = ast.Module(body=[function], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {"torch": torch}
            exec(compile(module, str(path), "exec"), namespace)
            return namespace["histogram_counts"]
    raise AssertionError(f"histogram_counts not found in {path}")


@pytest.fixture(params=IMPLEMENTATIONS, ids=("lab", "lecture-pl", "lecture-en"))
def histogram_counts(request):
    return _load_histogram_counts(request.param)


def test_histogram_keeps_underflow_and_overflow_in_boundary_bins(histogram_counts):
    values = torch.tensor([-2.0, 0.0, 0.49, 0.5, 1.0, 3.0])
    edges = torch.tensor([0.0, 0.5, 1.0])

    counts = histogram_counts(values, edges)

    assert torch.equal(counts, torch.tensor([3.0, 3.0]))
    assert counts.sum().item() == values.numel()


@pytest.mark.parametrize(
    ("values", "edges", "message"),
    (
        (torch.tensor([]), torch.tensor([0.0, 1.0]), "values must not be empty"),
        (torch.tensor([float("nan")]), torch.tensor([0.0, 1.0]), "values must be finite"),
        (torch.tensor([0.5]), torch.tensor([0.0]), "at least two"),
        (torch.tensor([0.5]), torch.tensor([0.0, 0.0]), "strictly increasing"),
    ),
)
def test_histogram_rejects_invalid_inputs(histogram_counts, values, edges, message):
    with pytest.raises(ValueError, match=message):
        histogram_counts(values, edges)
