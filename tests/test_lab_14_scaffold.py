"""The lab 14 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_07_scaffold.py and tests/test_lab_12_scaffold.py:
the notebook's very first code cell downloads EuroSAT and ResNet18 weights,
so there is no way to reach even the first student TODO offline -- the full
scaffold-to-first-TODO execution is skipped under CI=true.

The always-on tests below cover the metrics, Grad-CAM, and diagnostic-task
mechanics the lab's own deliverables are built around, using tiny synthetic
tensors and a tiny CNN standing in for the frozen-backbone + trainable-head
setup -- never the real EuroSAT/ResNet18 model, and never the private
solution's exact implementations (reimplemented independently here, the
same way test_lab_12_scaffold.py never imports its solution's functions).
"""

import os
from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "14_evaluation_interpretability" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="downloads EuroSAT and ResNet18 weights; run locally before opening a PR",
)
def test_lab_14_scaffold_runs_until_first_todo():
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


def test_confusion_matrix_and_macro_f1_on_known_example():
    """Mirrors the lecture's 3-class worked example (lab section 3)."""
    y_true = torch.tensor([0] * 10 + [1] * 8 + [2] * 2)
    y_pred = torch.tensor([0] * 9 + [1] + [1] * 8 + [0] * 2)

    C = 3
    confusion = torch.zeros(C, C, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        confusion[t, p] += 1

    f1s = []
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
        f1s.append(f1)

    accuracy = (y_true == y_pred).float().mean().item()
    macro_f1 = sum(f1s) / C
    assert abs(accuracy - 0.85) < 1e-6
    assert macro_f1 < 0.65  # minority class's zero recall drags macro-F1 down


def test_data_split_has_no_overlap():
    """Backs the lab's section-1 requirement: train/val/test must be
    verified disjoint programmatically, not assumed."""
    torch.manual_seed(0)
    perm = torch.randperm(1000)
    train_idx, val_idx, test_idx = perm[:600], perm[600:800], perm[800:1000]
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
    assert len(set(val_idx.tolist()) & set(test_idx.tolist())) == 0


class _TinyFrozenBackbone(nn.Module):
    """Stands in for a frozen ResNet18 backbone + trainable head."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 4, 10)
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

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
        score = logits[:, target_class].sum()
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
        return out.detach()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def test_gradcam_fails_on_frozen_backbone_without_input_grad():
    """Backs the lab's section-7/10 real pitfall: a frozen-backbone model
    needs requires_grad on the INPUT, not model.eval() alone."""
    torch.manual_seed(0)
    model = _TinyFrozenBackbone()
    x = torch.randn(1, 3, 8, 8)
    gradcam = _GradCAM(model, model.relu)
    with pytest.raises(AttributeError):
        gradcam(x, target_class=0)
    gradcam.remove_hooks()


def test_gradcam_works_once_input_requires_grad():
    torch.manual_seed(0)
    model = _TinyFrozenBackbone()
    x = torch.randn(1, 3, 8, 8).requires_grad_(True)
    gradcam = _GradCAM(model, model.relu)
    cam = gradcam(x, target_class=0)
    gradcam.remove_hooks()
    assert torch.isfinite(cam).all()


def test_gradcam_inside_no_grad_raises_runtime_error():
    """Backs the lab's diagnostic task bug 1: computing Grad-CAM inside
    torch.no_grad() breaks backward(), independent of the frozen-backbone
    issue (bug 2)."""
    torch.manual_seed(0)
    model = _TinyFrozenBackbone()
    x = torch.randn(1, 3, 8, 8).requires_grad_(True)
    gradcam = _GradCAM(model, model.relu)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model.zero_grad()
            logits = model(x)
            score = logits[:, 0].sum()
            score.backward()
    gradcam.remove_hooks()


def test_occlusion_baseline_must_be_normalized_not_raw_zero():
    """Backs the lab's section-8 requirement: the occlusion baseline value
    must match the model's preprocessing (ImageNet-normalized mid-gray),
    not an arbitrary raw zero which has a different meaning after
    normalization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    baseline_gray = (0.5 - mean) / std
    raw_zero = torch.zeros(3, 1, 1)
    # the normalized mid-gray baseline must differ meaningfully from raw zero
    assert (baseline_gray - raw_zero).abs().max().item() > 0.3


def test_brightness_subgroup_split_has_different_class_distribution():
    """Backs the lab's section-9 critical-thinking requirement: a brightness
    split can be confounded by class distribution, and that confound must
    be checked, not assumed away."""
    torch.manual_seed(0)
    # construct synthetic labels correlated with brightness
    n = 200
    labels = torch.randint(0, 3, (n,))
    brightness = labels.float() * 0.3 + torch.rand(n) * 0.1  # label 2 is brightest
    median_b = brightness.median()
    bright_mask = brightness >= median_b
    dark_mask = ~bright_mask

    import collections

    dist_bright = collections.Counter(labels[bright_mask].tolist())
    dist_dark = collections.Counter(labels[dark_mask].tolist())
    # the two class distributions must NOT be identical -- confound exists
    assert dist_bright != dist_dark
