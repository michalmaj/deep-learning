"""Lecture 07 notebook tests.

Split like module 06: a full `nbclient` execution of the real notebook is
kept (downloads Caltech101 + ResNet18 pretrained weights, several minutes
of CPU training) but is skipped under CI=true. The formula/mechanics
checks below need no network and no real images, so they run
unconditionally in CI as fast smoke tests -- they specifically cover the
non-obvious mechanics this lecture is about: freezing, partial
fine-tuning, and the BatchNorm train()/eval() distinction.
"""

from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "07_transfer_learning" / "examples.ipynb"


@pytest.mark.network
def test_lecture_07_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def _tiny_backbone_head_model(n_classes=3):
    """A tiny stand-in for ResNet18's backbone/head split, built from
    synthetic shapes only -- no pretrained weights, no download. Mirrors
    just the structural pattern (conv backbone -> pooled features -> fc
    head) the lecture's freezing/fine-tuning logic operates on.
    """

    class TinyBackboneHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(16, n_classes)

        def forward(self, x):
            feats = self.backbone(x).flatten(1)
            return self.fc(feats)

    return TinyBackboneHead()


def _freeze_all_but(model, submodules):
    for p in model.parameters():
        p.requires_grad = False
    for m in submodules:
        for p in m.parameters():
            p.requires_grad = True


def _one_training_step(model, param_groups, x, y):
    opt = torch.optim.Adam(param_groups, lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()


@pytest.mark.parametrize(
    ("name", "select_submodules"),
    [
        ("frozen backbone (head only)", lambda m: [m.fc]),
        ("partial (last backbone layer + head)", lambda m: [m.backbone[2], m.fc]),
        ("full fine-tune (everything)", lambda m: [m]),
    ],
)
def test_freezing_updates_only_intended_parameters(name, select_submodules):
    torch.manual_seed(0)
    model = _tiny_backbone_head_model()
    _freeze_all_but(model, select_submodules(model))

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    x = torch.randn(4, 3, 16, 16)
    y = torch.randint(0, 3, (4,))

    model.train()
    for _ in range(3):
        _one_training_step(model, trainable, x, y)

    for n, p in model.named_parameters():
        changed = not torch.equal(before[n], p.detach())
        assert changed == p.requires_grad, (
            f"{name}: parameter {n} requires_grad={p.requires_grad} but changed={changed}"
        )


def test_head_replacement_param_count_matches_formula():
    model = _tiny_backbone_head_model(n_classes=5)
    expected_head_params = 16 * 5 + 5  # nn.Linear(16, 5): weight + bias
    actual_head_params = sum(p.numel() for p in model.fc.parameters())
    assert actual_head_params == expected_head_params


def test_batchnorm_train_mode_drifts_even_when_frozen_eval_mode_does_not():
    class TinyBNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn(self.conv(x)))).flatten(1)
            return self.fc(x)

    def run(train_mode):
        torch.manual_seed(0)
        model = TinyBNModel()
        _freeze_all_but(model, [model.fc])
        before = model.bn.running_mean.detach().clone()

        model.train() if train_mode else model.eval()
        x = torch.randn(4, 3, 8, 8)
        y = torch.randint(0, 3, (4,))
        for _ in range(3):
            _one_training_step(model, list(model.fc.parameters()), x, y)

        return not torch.equal(before, model.bn.running_mean.detach())

    drifted_in_train = run(train_mode=True)
    drifted_in_eval = run(train_mode=False)
    assert drifted_in_train is True, "running_mean should drift in train() even if frozen"
    assert drifted_in_eval is False, "model.eval() should stop running_mean updates"


@pytest.mark.parametrize("n_classes", [3, 10])
def test_model_output_shape_matches_n_classes(n_classes):
    model = _tiny_backbone_head_model(n_classes=n_classes)
    x = torch.randn(4, 3, 16, 16)
    out = model(x)
    assert out.shape == (4, n_classes)
