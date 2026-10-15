"""Lecture 08 notebook tests.

Split like modules 06-07: a full `nbclient` execution of the real
notebook is kept (downloads Caltech101 + ResNet18 pretrained weights)
but is skipped under CI=true. The formula/mechanics checks below need no
network and no real images, so they run unconditionally in CI as fast
smoke tests -- they specifically cover the non-obvious mechanics this
lecture is about: Euclidean vs cosine disagreement, the unit-vector
identity, L2 normalization's zero-vector edge case, k-NN ranking, and
nn.Embedding updating only the rows that appear in training data.
"""

import os
from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "08_embeddings" / "examples.ipynb"


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="downloads Caltech101 + ResNet18 weights; run locally before opening a PR",
)
def test_lecture_08_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def _euclidean(a, b):
    return torch.norm(a - b)


def _cosine(a, b):
    return (a @ b) / (a.norm() * b.norm())


def test_euclidean_and_cosine_can_rank_oppositely():
    q = torch.tensor([1.0, 0.0])
    a = torch.tensor([10.0, 0.0])
    b = torch.tensor([0.9, 0.9])
    closer_euclidean = "a" if _euclidean(q, a) < _euclidean(q, b) else "b"
    closer_cosine = "a" if _cosine(q, a) > _cosine(q, b) else "b"
    assert closer_euclidean != closer_cosine


def test_unit_vector_identity_matches_squared_euclidean():
    torch.manual_seed(0)
    for _ in range(50):
        x = torch.randn(6)
        y = torch.randn(6)
        xu, yu = x / x.norm(), y / y.norm()
        lhs = (xu - yu).pow(2).sum()
        rhs = 2 - 2 * _cosine(xu, yu)
        assert torch.allclose(lhs, rhs, atol=1e-5)


def test_l2_normalization_zero_vector_is_nan_without_epsilon():
    zero_vec = torch.zeros(5)
    naive = zero_vec / zero_vec.norm()
    assert torch.isnan(naive).all()

    safe = zero_vec / (zero_vec.norm() + 1e-8)
    assert not torch.isnan(safe).any()
    assert torch.equal(safe, torch.zeros(5))


def test_knn_ranking_can_differ_between_metrics():
    torch.manual_seed(1)
    n, dim, k = 20, 8, 5
    pool = torch.randn(n, dim)
    pool[3] = pool[0] * 3.0 + torch.randn(dim) * 0.05
    query = pool[0].clone()

    dists = torch.norm(pool - query, dim=1)
    coss = (pool @ query) / (pool.norm(dim=1) * query.norm())

    euclid_top = set(dists.argsort()[1 : k + 1].tolist())
    cosine_top = set((-coss).argsort()[1 : k + 1].tolist())
    assert euclid_top != cosine_top


def test_pca_2d_matches_full_rank_svd():
    torch.manual_seed(0)
    X = torch.randn(50, 10) @ torch.randn(10, 10)
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = eigvals.argsort(descending=True)
    top2 = eigvecs[:, idx[:2]]
    proj = Xc @ top2

    _, _, v = torch.pca_lowrank(Xc, q=10)
    proj_ref = Xc @ v[:, :2]

    for i in range(2):
        corr = torch.corrcoef(torch.stack([proj[:, i], proj_ref[:, i]]))[0, 1]
        assert abs(corr.item()) > 0.999


def test_embedding_updates_only_rows_present_in_training_data():
    torch.manual_seed(0)
    vocab_size, dim = 6, 4
    embedding = nn.Embedding(vocab_size, dim)
    before = embedding.weight.detach().clone()

    train_ids = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
    train_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    head = nn.Linear(dim, 2)
    opt = torch.optim.Adam(list(embedding.parameters()) + list(head.parameters()), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(head(embedding(train_ids)), train_labels)
        loss.backward()
        opt.step()

    after = embedding.weight.detach()
    used_ids = {0, 1, 2, 3}
    for i in range(vocab_size):
        changed = not torch.equal(before[i], after[i])
        assert changed == (i in used_ids), f"row {i}: changed={changed}, expected {i in used_ids}"
