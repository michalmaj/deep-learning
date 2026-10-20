"""Lecture 14 notebook tests.

This notebook downloads Fashion-MNIST (via torchvision, already registered
in data/registry.yml for module 06) -- the full nbclient execution is
marked `network` and runs in scheduled network CI, matching modules 06/12.

The fast, always-on tests below cover the actual metrics/calibration/
Grad-CAM mechanics the lecture is built around, using tiny synthetic
tensors and a tiny CNN -- never the real Fashion-MNIST data. Several reuse
the exact numbers verified by hand in lecture.md's worked examples, so a
future change to the formulas would have to update both places
consistently.
"""

from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "14_evaluation_interpretability" / "examples.ipynb"


@pytest.mark.network
def test_lecture_14_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def _confusion_matrix(y_true, y_pred, C):
    confusion = torch.zeros(C, C, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        confusion[t, p] += 1
    return confusion


def _per_class_prf1(confusion, C):
    precisions, recalls, f1s = [], [], []
    for c in range(C):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def test_confusion_matrix_and_accuracy_match_lecture_worked_example():
    """Exact numbers from lecture.md's 3-class worked example."""
    y_true = torch.tensor([0] * 10 + [1] * 8 + [2] * 2)
    y_pred = torch.tensor([0] * 9 + [1] + [1] * 8 + [0] * 2)

    confusion = _confusion_matrix(y_true, y_pred, 3)
    expected = torch.tensor([[9, 1, 0], [0, 8, 0], [2, 0, 0]])
    assert torch.equal(confusion, expected)

    accuracy = (y_true == y_pred).float().mean().item()
    assert abs(accuracy - 0.85) < 1e-6


def test_macro_f1_reveals_minority_class_failure_accuracy_hides():
    y_true = torch.tensor([0] * 10 + [1] * 8 + [2] * 2)
    y_pred = torch.tensor([0] * 9 + [1] + [1] * 8 + [0] * 2)
    confusion = _confusion_matrix(y_true, y_pred, 3)
    precisions, recalls, f1s = _per_class_prf1(confusion, 3)

    accuracy = (y_true == y_pred).float().mean().item()
    macro_f1 = sum(f1s) / 3

    assert recalls[2] == 0.0  # minority class never correctly recalled
    assert abs(precisions[2]) != precisions[2]  # NaN: 0/0, class never predicted
    assert accuracy > 0.8 and macro_f1 < 0.65  # high accuracy, much lower macro-F1


def test_micro_f1_equals_accuracy_for_single_label_multiclass():
    torch.manual_seed(0)
    y_true = torch.randint(0, 5, (100,))
    y_pred = torch.randint(0, 5, (100,))
    confusion = _confusion_matrix(y_true, y_pred, 5)

    total_tp = sum(confusion[c, c].item() for c in range(5))
    micro_f1 = total_tp / len(y_true)
    accuracy = (y_true == y_pred).float().mean().item()
    assert abs(micro_f1 - accuracy) < 1e-6


def _compute_ece(confidences, correct, n_bins=10):
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for m in range(n_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]
        mask = (
            (confidences >= lo) & (confidences < hi)
            if m < n_bins - 1
            else (confidences >= lo) & (confidences <= hi)
        )
        n_m = mask.sum().item()
        if n_m == 0:
            continue
        acc_m = correct[mask].float().mean().item()
        conf_m = confidences[mask].mean().item()
        ece += (n_m / n) * abs(acc_m - conf_m)
    return ece


def test_ece_matches_hand_computation():
    """Exact number from lecture.md's 2-bin worked example."""
    confidences = torch.tensor([0.95, 0.95, 0.95, 0.95, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55])
    correct = torch.tensor([1, 1, 1, 0, 1, 1, 0, 0, 0, 0], dtype=torch.float)
    ece = _compute_ece(confidences, correct, n_bins=10)
    assert abs(ece - 0.21) < 1e-6


def test_ece_is_sensitive_to_bin_count_and_sample_size():
    """Mirrors the lecture's measured (not just claimed) ECE instability."""
    torch.manual_seed(0)
    n = 200
    confidences = torch.rand(n) * 0.5 + 0.5
    correct = (torch.rand(n) < (confidences - 0.15)).float()

    ece_10 = _compute_ece(confidences, correct, n_bins=10)
    ece_50 = _compute_ece(confidences, correct, n_bins=50)
    assert ece_10 != pytest.approx(ece_50, abs=1e-6)

    torch.manual_seed(1)
    idx_small = torch.randperm(n)[:20]
    ece_small_sample = _compute_ece(confidences[idx_small], correct[idx_small], n_bins=10)
    assert ece_small_sample != pytest.approx(ece_10, abs=1e-6)


class _TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x):
        return self.fc(self.relu(self.conv(x)).flatten(1))


class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, target_class):
        self.model.zero_grad()
        logits = self.model(x)
        if isinstance(target_class, int):
            target_class = torch.full((x.shape[0],), target_class, dtype=torch.long)
        score = logits.gather(1, target_class.unsqueeze(1)).sum()
        score.backward()
        alpha = self.gradients.mean(dim=(2, 3))
        weighted = (alpha.unsqueeze(-1).unsqueeze(-1) * self.activations).sum(dim=1)
        cam = torch.relu(weighted)
        cam_up = F.interpolate(
            cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        out = torch.empty_like(cam_up)
        for i in range(cam_up.shape[0]):
            cmin, cmax = cam_up[i].min(), cam_up[i].max()
            out[i] = (
                (cam_up[i] - cmin) / (cmax - cmin)
                if (cmax - cmin).item() > 1e-8
                else torch.zeros_like(cam_up[i])
            )
        return out.detach(), logits.detach()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def test_gradcam_matches_manual_retain_grad_computation():
    torch.manual_seed(0)
    model = _TinyCNN()
    x = torch.randn(1, 1, 8, 8)

    gradcam = _GradCAM(model, model.relu)
    cam_hook, _ = gradcam(x, target_class=1)
    gradcam.remove_hooks()

    a_manual = torch.relu(model.conv(x))
    a_manual.retain_grad()
    score_manual = model.fc(a_manual.flatten(1))[0, 1]
    score_manual.backward()
    alpha_manual = a_manual.grad.mean(dim=(2, 3))
    weighted_manual = (alpha_manual.unsqueeze(-1).unsqueeze(-1) * a_manual.detach()).sum(dim=1)
    cam_manual = torch.relu(weighted_manual)
    cam_manual_up = F.interpolate(
        cam_manual.unsqueeze(1), size=(8, 8), mode="bilinear", align_corners=False
    )
    cmin, cmax = cam_manual_up.min(), cam_manual_up.max()
    cam_manual_norm = (
        (cam_manual_up - cmin) / (cmax - cmin)
        if (cmax - cmin).item() > 1e-8
        else torch.zeros_like(cam_manual_up)
    )
    assert torch.allclose(cam_hook, cam_manual_norm, atol=1e-5)


def test_gradcam_handles_zero_spread_map_without_dividing_by_zero():
    torch.manual_seed(0)
    model = _TinyCNN()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
    x = torch.randn(1, 1, 8, 8)
    gradcam = _GradCAM(model, model.relu)
    cam, _ = gradcam(x, target_class=0)
    gradcam.remove_hooks()
    assert torch.isfinite(cam).all()
    assert (cam == 0).all()  # all-zero weighted sum -> zero map, not NaN from /0


def test_gradcam_hooks_are_removed_cleanly():
    model = _TinyCNN()
    gradcam = _GradCAM(model, model.relu)
    assert len(model.relu._forward_hooks) == 1
    assert len(model.relu._backward_hooks) == 1
    gradcam.remove_hooks()
    assert len(model.relu._forward_hooks) == 0
    assert len(model.relu._backward_hooks) == 0


def test_gradcam_batch_matches_single_example_no_cross_leakage():
    torch.manual_seed(0)
    model = _TinyCNN()
    x_batch = torch.randn(3, 1, 8, 8)
    gradcam = _GradCAM(model, model.relu)
    cam_batch, _ = gradcam(x_batch, target_class=0)
    cam_single, _ = gradcam(x_batch[0:1], target_class=0)
    gradcam.remove_hooks()
    assert torch.allclose(cam_batch[0:1], cam_single, atol=1e-4)


def test_eval_alone_does_not_break_gradients():
    model = _TinyCNN()
    model.eval()
    x = torch.randn(1, 1, 8, 8)
    out = model(x)
    out[0, 0].backward()
    assert model.conv.weight.grad is not None
    assert torch.isfinite(model.conv.weight.grad).all()


def test_no_grad_breaks_backward():
    model = _TinyCNN()
    x = torch.randn(1, 1, 8, 8)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            out = model(x)
            out[0, 0].backward()


def test_frozen_backbone_requires_input_grad_for_gradcam():
    """Mirrors the lab's real pitfall: a frozen BACKBONE (conv) with only the
    HEAD (fc) trainable still lets backward() succeed (the head requires
    grad), but the hooked backbone layer's gradient stays None -- unless the
    input itself requires grad, since no graph reaches the frozen conv
    otherwise."""
    model = _TinyCNN()
    model.conv.weight.requires_grad = False
    model.conv.bias.requires_grad = False
    x = torch.randn(1, 1, 8, 8)

    gradcam = _GradCAM(model, model.relu)
    with pytest.raises(AttributeError):
        gradcam(x, target_class=0)

    x_grad = x.clone().requires_grad_(True)
    cam, _ = gradcam(x_grad, target_class=0)
    gradcam.remove_hooks()
    assert torch.isfinite(cam).all()
