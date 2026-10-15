"""Lecture 06 notebook tests.

Split in two, unlike modules 01-05, because this lecture downloads
Fashion-MNIST: a full `nbclient` execution of the real notebook is kept
(same rigor as before, same "Restart Kernel -> Run All" guarantee), but
it is skipped automatically in CI (`CI=true`, the variable GitHub Actions
sets) so that pushes never trigger a dataset download. Run it locally
before opening a PR. The formula/shape checks below need no network and
no real images, so they run unconditionally in CI as fast smoke tests.
"""

import os
from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "06_cnn" / "examples.ipynb"


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="downloads Fashion-MNIST; run locally before opening a PR",
)
def test_lecture_06_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def _conv_output_size(size, kernel, stride, padding):
    return (size + 2 * padding - kernel) // stride + 1


@pytest.mark.parametrize(
    ("size", "kernel", "stride", "padding"),
    [
        (12, 3, 1, 0),
        (12, 3, 1, 1),
        (12, 3, 2, 0),
        (12, 3, 2, 1),
        (28, 5, 1, 2),
        (28, 5, 2, 0),
        (7, 3, 1, 1),
        (9, 4, 2, 1),
    ],
)
def test_conv2d_output_size_matches_formula(size, kernel, stride, padding):
    conv = nn.Conv2d(1, 1, kernel_size=kernel, stride=stride, padding=padding)
    x = torch.zeros(1, 1, size, size)
    out = conv(x)
    expected = _conv_output_size(size, kernel, stride, padding)
    assert out.shape[-1] == expected
    assert out.shape[-2] == expected


def _conv_param_count(in_channels, out_channels, kernel, bias=True):
    weight_params = in_channels * kernel * kernel * out_channels
    bias_params = out_channels if bias else 0
    return weight_params + bias_params


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "kernel", "bias"),
    [
        (1, 16, 3, True),
        (16, 32, 3, True),
        (3, 8, 5, True),
        (16, 32, 3, False),
    ],
)
def test_conv2d_param_count_matches_formula(in_channels, out_channels, kernel, bias):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, bias=bias)
    actual = sum(p.numel() for p in conv.parameters())
    expected = _conv_param_count(in_channels, out_channels, kernel, bias=bias)
    assert actual == expected


class _LectureSmallCNN(nn.Module):
    """Mirrors the notebook's SmallCNN; kept in sync manually, not imported."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)


class _LectureMLP(nn.Module):
    """Mirrors the notebook's MLP; kept in sync manually, not imported."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.parametrize("model_cls", [_LectureSmallCNN, _LectureMLP])
def test_lecture_architecture_accepts_fashion_mnist_shape(model_cls):
    model = model_cls()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)
