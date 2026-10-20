"""Lecture 15 notebook tests.

This notebook downloads Fashion-MNIST (already registered in
data/registry.yml for module 06) -- the full nbclient execution is marked
`network` and runs in scheduled network CI, matching modules 06/12/14.

The fast, always-on tests below cover the artifact save/load mechanics,
the eval()/train() distinction, percentile math, and PSI -- using tiny
synthetic tensors and models, never the real Fashion-MNIST data. Several
reuse the exact numbers verified by hand in lecture.md's worked examples.
"""

from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "15_production" / "examples.ipynb"


@pytest.mark.network
def test_lecture_15_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


def test_artifact_save_and_reload_gives_bit_exact_predictions(tmp_path):
    torch.manual_seed(0)
    model = _TinyNet()
    model.eval()
    x = torch.randn(1, 4)
    with torch.no_grad():
        logits_before = model(x)

    weights_path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weights_path)

    fresh_model = _TinyNet()
    state = torch.load(weights_path, weights_only=True)
    fresh_model.load_state_dict(state)
    fresh_model.eval()
    with torch.no_grad():
        logits_after = fresh_model(x)

    assert torch.equal(logits_before, logits_after)


def test_forgetting_eval_changes_predictions_due_to_dropout():
    """Mirrors the lecture's core artifact-consistency risk: frozen weights
    do not mean frozen behavior for layers like Dropout/BatchNorm."""
    torch.manual_seed(0)
    model = _TinyNet()
    x = torch.randn(4, 4)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    torch.manual_seed(1)
    model.train()
    with torch.no_grad():
        out_train = model(x)

    assert not torch.equal(out_eval, out_train)


def test_class_reordering_changes_label_not_index():
    """The model's predicted index and logit are unaffected by metadata
    ordering -- only the human-readable label is wrong if class_names is
    out of sync with training order."""
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    pred_idx = logits.argmax(1).item()

    classes_correct = ["A", "B", "C"]
    classes_wrong_order = ["B", "A", "C"]

    assert pred_idx == 1  # same index regardless of metadata
    assert classes_correct[pred_idx] != classes_wrong_order[pred_idx]


def _compute_ece_percentiles_demo():
    latencies = torch.tensor([float(i) for i in range(1, 21)])
    return latencies


def test_percentile_matches_hand_computed_linear_interpolation():
    """Exact numbers from lecture.md's latency worked example."""
    latencies = _compute_ece_percentiles_demo()
    n = len(latencies)
    p95 = torch.quantile(latencies, 0.95).item()

    sorted_lat = latencies.sort().values
    rank = 0.95 * (n - 1)
    lo = int(rank)
    frac = rank - lo
    manual_p95 = (sorted_lat[lo] + frac * (sorted_lat[lo + 1] - sorted_lat[lo])).item()

    assert abs(p95 - manual_p95) < 1e-6
    assert abs(p95 - 19.05) < 1e-3


def test_median_and_quantile_can_disagree_for_even_n():
    """Documents the real gotcha: torch.median uses the lower of the two
    middle elements, torch.quantile(0.5) interpolates -- different
    conventions, not a bug."""
    latencies = torch.tensor([float(i) for i in range(1, 21)])
    median_val = latencies.median().item()
    quantile_val = torch.quantile(latencies, 0.5).item()
    assert median_val == 10.0
    assert quantile_val == 10.5
    assert median_val != quantile_val


def _psi(ref_counts, cur_counts, eps=1e-4):
    ref_props = torch.clamp(ref_counts / ref_counts.sum(), min=eps)
    cur_props = torch.clamp(cur_counts / cur_counts.sum(), min=eps)
    return ((cur_props - ref_props) * torch.log(cur_props / ref_props)).sum().item()


def test_psi_matches_hand_computation():
    """Exact number from lecture.md's 2-bin worked example."""
    ref = torch.tensor([50.0, 50.0])
    cur = torch.tensor([70.0, 30.0])
    result = _psi(ref, cur)
    expected = (
        0.2 * torch.log(torch.tensor(1.4)).item() + (-0.2) * torch.log(torch.tensor(0.6)).item()
    )
    assert abs(result - expected) < 1e-6


def test_psi_is_zero_for_identical_distributions():
    ref = torch.tensor([50.0, 50.0])
    assert abs(_psi(ref, ref)) < 1e-9


def test_psi_without_smoothing_diverges_on_empty_bin():
    """Backs the lecture's explicit zero-frequency demonstration: without
    smoothing, an empty bin gives an infinite PSI."""
    ref = torch.tensor([50.0, 50.0])
    cur = torch.tensor([100.0, 0.0])
    ref_props = ref / ref.sum()
    cur_props = cur / cur.sum()
    unsmoothed = ((cur_props - ref_props) * torch.log(cur_props / ref_props)).sum().item()
    assert unsmoothed == float("inf")

    smoothed = _psi(ref, cur, eps=1e-4)
    assert torch.isfinite(torch.tensor(smoothed))
    assert smoothed > 1.0  # still large, honestly reflecting the near-empty bin


def test_psi_distinguishes_same_from_shifted_distribution():
    torch.manual_seed(0)
    ref_vals = torch.randn(500)
    same_vals = torch.randn(500)
    shifted_vals = torch.randn(500) + 2.0

    bin_edges = torch.linspace(-4, 4, 11)

    def hist(values):
        counts = torch.zeros(10)
        for i in range(10):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (values >= lo) & (values < hi) if i < 9 else (values >= lo) & (values <= hi)
            counts[i] = mask.sum()
        return counts

    psi_same = _psi(hist(ref_vals), hist(same_vals))
    psi_shifted = _psi(hist(ref_vals), hist(shifted_vals))
    assert psi_same < 0.25
    assert psi_shifted > psi_same * 5
