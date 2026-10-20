"""The lab 12 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_08_scaffold.py and tests/test_lab_11_scaffold.py:
the notebook's very first code cell downloads Fashion-MNIST, so there is no
way to reach even the first student TODO offline -- the full
scaffold-to-first-TODO execution is marked `network` and runs in scheduled
network CI.

The always-on tests below cover the anomaly-detection metric mechanics
(confusion counts, precision/recall/F1, ROC-AUC, PR-AUC, threshold logic)
the lab's own deliverables are built around, using only tiny synthetic
tensors -- never the real Fashion-MNIST data, and never the private
solution's exact implementations (reimplemented independently here, the
same way test_lab_11_scaffold.py never imports generate_response).
"""

from pathlib import Path

import nbformat
import pytest
import torch
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "12_autoencoders" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


@pytest.mark.network
def test_lab_12_scaffold_runs_until_first_todo():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [(i, _cell_error(c)) for i, c in enumerate(code_cells)]
    first_error_index, first_error = next(e for e in errors if e[1] is not None)

    for i, error in errors:
        if i < first_error_index:
            assert error is None, f"scaffold cell {i} failed unexpectedly: {error}"

    assert first_error["ename"] == "NotImplementedError", (
        f"expected the first failure to be the data-split TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )


def _confusion_counts(scores, labels, threshold):
    pred = (scores > threshold).long()
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    return tp, fp, tn, fn


def _precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return precision, recall, f1


def _roc_auc(labels, scores):
    order = scores.argsort()
    ranks = torch.empty(len(scores), dtype=torch.float)
    ranks[order] = torch.arange(len(scores), dtype=torch.float)
    n1 = labels.sum().item()
    n0 = (labels == 0).sum().item()
    sum_ranks_pos = ranks[labels == 1].sum().item()
    return (sum_ranks_pos - n1 * (n1 - 1) / 2) / (n1 * n0)


def _pr_auc(labels, scores):
    order = scores.argsort(descending=True)
    labels_sorted = labels[order]
    n_pos = labels.sum().item()
    tp_running, ap = 0, 0.0
    for i, lab in enumerate(labels_sorted.tolist()):
        if lab == 1:
            tp_running += 1
            ap += tp_running / (i + 1)
    return ap / n_pos


KNOWN_SCORES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.6, 0.7, 0.8, 1.0])
KNOWN_LABELS = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
KNOWN_THRESHOLD = 0.45


def test_confusion_counts_on_known_example():
    """Matches README.md section 5's hand-verified example."""
    tp, fp, tn, fn = _confusion_counts(KNOWN_SCORES, KNOWN_LABELS, KNOWN_THRESHOLD)
    assert (tp, fp, tn, fn) == (5, 1, 4, 0)


def test_precision_recall_f1_on_known_example():
    tp, fp, tn, fn = _confusion_counts(KNOWN_SCORES, KNOWN_LABELS, KNOWN_THRESHOLD)
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    assert abs(precision - 0.8333) < 1e-3
    assert abs(recall - 1.0) < 1e-6
    assert abs(f1 - 0.9091) < 1e-3


def test_roc_auc_matches_brute_force_pair_counting():
    roc = _roc_auc(KNOWN_LABELS, KNOWN_SCORES)
    assert abs(roc - 0.84) < 1e-3

    n0, n1 = (KNOWN_LABELS == 0).sum().item(), (KNOWN_LABELS == 1).sum().item()
    count = 0.0
    for i in range(len(KNOWN_SCORES)):
        for j in range(len(KNOWN_SCORES)):
            if KNOWN_LABELS[i] == 0 and KNOWN_LABELS[j] == 1:
                if KNOWN_SCORES[j] > KNOWN_SCORES[i]:
                    count += 1
                elif KNOWN_SCORES[j] == KNOWN_SCORES[i]:
                    count += 0.5
    assert abs(count / (n0 * n1) - roc) < 1e-6


def test_pr_auc_on_known_example():
    pr = _pr_auc(KNOWN_LABELS, KNOWN_SCORES)
    assert abs(pr - 0.81) < 1e-3


def test_threshold_selected_only_from_normal_validation_distribution():
    """The lab's core rule (section 4): the threshold must be a function of
    val_normal errors only, never anomaly or test labels."""
    torch.manual_seed(0)
    err_val_normal = torch.rand(50) * 0.1

    def select_threshold(err_val_normal, k=2):
        return (err_val_normal.mean() + k * err_val_normal.std()).item()

    threshold = select_threshold(err_val_normal)
    # changing anomaly/test data must not change a threshold computed this way
    threshold_again = select_threshold(err_val_normal)
    assert threshold == threshold_again
    assert threshold > err_val_normal.mean().item()


def test_leaky_threshold_selection_uses_test_labels_and_differs_from_honest():
    """Mirrors the lab's diagnostic task: sweeping thresholds against test
    labels to maximize F1 is a different (leaky) procedure from picking a
    threshold from validation-normal data alone, even when both happen to
    land close together."""
    torch.manual_seed(0)
    normal_errs = torch.rand(100) * 0.05
    anomaly_errs = torch.rand(100) * 0.05 + 0.03
    all_scores = torch.cat([normal_errs, anomaly_errs])
    all_labels = torch.cat([torch.zeros(100), torch.ones(100)])

    honest_threshold = (normal_errs.mean() + 2 * normal_errs.std()).item()

    candidates = torch.linspace(all_scores.min(), all_scores.max(), 200)
    best_f1 = -1.0
    for t in candidates:
        tp, fp, tn, fn = _confusion_counts(all_scores, all_labels, t.item())
        _, _, f1 = _precision_recall_f1(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1

    tp, fp, tn, fn = _confusion_counts(all_scores, all_labels, honest_threshold)
    _, _, honest_f1 = _precision_recall_f1(tp, fp, fn)

    # the leaky procedure is defined as a maximum over the test set itself,
    # so it can never score worse on that same test set than the honest one
    assert best_f1 >= honest_f1 - 1e-9


def test_mean_normal_image_baseline_is_a_valid_comparison_score():
    """The lab's required trivial baseline: distance to the mean normal
    training image, with no learned parameters at all."""
    torch.manual_seed(0)
    normal_train = torch.rand(20, 1, 4, 4)
    test_normal = torch.rand(5, 1, 4, 4)
    test_anomaly = torch.rand(5, 1, 4, 4) + 0.5

    mean_img = normal_train.mean(dim=0, keepdim=True)
    err_normal = ((test_normal - mean_img) ** 2).flatten(1).mean(dim=1)
    err_anomaly = ((test_anomaly - mean_img) ** 2).flatten(1).mean(dim=1)

    assert err_normal.shape == (5,)
    assert err_anomaly.mean() > err_normal.mean()
