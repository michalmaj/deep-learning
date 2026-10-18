"""Lecture 12 notebook tests.

This notebook downloads KMNIST (via torchvision, see data/registry.yml) --
the full `nbclient` execution is skipped under CI=true, the same pattern as
modules 06-08/11.

The fast, always-on tests below cover the actual VAE/AE math (KL formula,
reparameterization trick + gradient flow, loss-reduction correctness, shape
consistency) using tiny synthetic tensors -- never the real KMNIST data, and
never requiring network access. Several reuse the exact numbers verified by
hand in `lecture.md`'s worked examples, so a future change to the formulas
would have to update both places consistently.
"""

import os
from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn.functional as F
from nbclient import NotebookClient
from torch.distributions import Normal, kl_divergence

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "12_autoencoders" / "examples.ipynb"


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="downloads KMNIST; run locally before opening a PR",
)
def test_lecture_12_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def test_ae_hand_example_matches_lecture_md():
    """Exact numbers from lecture.md's 4-dim worked AE example."""
    x = torch.tensor([1.0, 0.5, -0.5, 0.0])
    W_enc = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    e = W_enc @ x
    x_hat = W_enc.T @ e
    loss = ((x - x_hat) ** 2).sum()
    assert torch.equal(e, torch.tensor([1.0, 0.0]))
    assert torch.equal(x_hat, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert abs(loss.item() - 0.5) < 1e-6


def test_kl_formula_matches_torch_distributions():
    """Exact numbers from lecture.md's 2D KL worked example."""
    mu = torch.tensor([0.5, -0.3])
    logvar = torch.tensor([-0.2, 0.4])
    sigma = torch.exp(0.5 * logvar)

    kl_manual = 0.5 * (sigma.pow(2) + mu.pow(2) - 1 - logvar)
    kl_torch = kl_divergence(Normal(mu, sigma), Normal(torch.zeros(2), torch.ones(2)))

    assert torch.allclose(kl_manual, kl_torch, atol=1e-6)
    assert torch.allclose(kl_manual, torch.tensor([0.1344, 0.0909]), atol=1e-3)


def test_kl_divergence_is_never_negative():
    torch.manual_seed(0)
    mu = torch.randn(1000, 4)
    logvar = torch.randn(1000, 4)
    kl = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar)
    assert kl.min().item() >= -1e-5


def test_reparameterization_trick_matches_hand_computation():
    """Exact numbers from lecture.md's reparameterization worked example."""
    mu = torch.tensor([0.5, -0.3], requires_grad=True)
    logvar = torch.tensor([-0.2, 0.4], requires_grad=True)
    eps = torch.tensor([1.0, -0.5])

    z = mu + torch.exp(0.5 * logvar) * eps
    assert torch.allclose(z, torch.tensor([1.4048, -0.9107]), atol=1e-3)

    loss = (z**2).sum()
    loss.backward()

    assert torch.allclose(mu.grad, 2 * z.detach(), atol=1e-4)
    expected_logvar_grad = 2 * z.detach() * eps * 0.5 * torch.exp(0.5 * logvar.detach())
    assert torch.allclose(logvar.grad, expected_logvar_grad, atol=1e-4)
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(logvar.grad).all()


def test_reparameterization_is_stochastic_without_fixed_epsilon():
    """Same mu/logvar, two calls without a fixed eps must generally differ --
    confirms the randomness is still present (the trick only relocates it)."""
    torch.manual_seed(0)
    mu = torch.zeros(8)
    logvar = torch.zeros(8)

    def reparameterize(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    z1 = reparameterize(mu, logvar)
    z2 = reparameterize(mu, logvar)
    assert not torch.equal(z1, z2)


def test_vae_mu_logvar_z_reconstruction_shapes_are_consistent():
    batch, d_in, d_z = 5, 16, 3
    mu = torch.randn(batch, d_z)
    logvar = torch.randn(batch, d_z)
    eps = torch.randn(batch, d_z)
    z = mu + torch.exp(0.5 * logvar) * eps
    x_logits = torch.randn(batch, d_in)

    assert mu.shape == logvar.shape == eps.shape == z.shape == (batch, d_z)
    assert x_logits.shape == (batch, d_in)


def test_sum_vs_mean_kl_reduction_changes_effective_weight():
    """Mirrors the lecture's loss-scaling misconception: replacing sum with
    mean over latent dims divides the effective KL weight by d_z."""
    torch.manual_seed(0)
    d_z = 8
    mu = torch.randn(4, d_z) * 0.5
    logvar = torch.randn(4, d_z) * 0.3

    kl_sum = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()
    kl_mean = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    assert torch.allclose(kl_sum, kl_mean * d_z, atol=1e-4)


def test_vae_loss_recon_and_kl_reduction_is_mean_over_batch():
    torch.manual_seed(0)
    batch, d_in, d_z = 6, 20, 4
    x = torch.bernoulli(torch.full((batch, d_in), 0.5))
    x_logits = torch.randn(batch, d_in)
    mu = torch.randn(batch, d_z)
    logvar = torch.randn(batch, d_z)

    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch

    # manually recompute per-example then average, must match the batched formula
    recon_manual = torch.stack(
        [
            F.binary_cross_entropy_with_logits(x_logits[i], x[i], reduction="sum")
            for i in range(batch)
        ]
    ).mean()
    kl_manual = torch.stack(
        [-0.5 * torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp()) for i in range(batch)]
    ).mean()

    assert torch.allclose(recon, recon_manual, atol=1e-4)
    assert torch.allclose(kl, kl_manual, atol=1e-4)


def test_overcomplete_vs_undercomplete_identity_mapping_on_full_rank_data():
    """Fast, tiny version of the notebook's synthetic AE verification --
    confirms overcomplete AE can reach near-zero loss while undercomplete
    cannot, on the same full-rank data."""
    torch.manual_seed(0)
    d = 6
    X = torch.randn(80, d)

    def train(latent, epochs=200):
        torch.manual_seed(0)
        enc = torch.nn.Linear(d, latent)
        dec = torch.nn.Linear(latent, d)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.05)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.mse_loss(dec(torch.relu(enc(X))), X)
            loss.backward()
            opt.step()
        return loss.item()

    overcomplete_mse = train(latent=16)
    undercomplete_mse = train(latent=1)
    assert overcomplete_mse < 1e-2
    assert undercomplete_mse > 0.2


def test_reconstruction_and_prior_generation_use_different_latent_sources():
    """Mirrors the lecture's distinction between reconstruction, interpolation
    and generation: the z fed to the decoder comes from a different source
    in each case, and must not be accidentally identical."""
    torch.manual_seed(0)
    mu_real = torch.randn(1, 3)
    z_reconstruction = mu_real  # uses the encoded real example's mean
    z_generation = torch.randn(1, 3)  # uses a fresh prior sample
    assert not torch.equal(z_reconstruction, z_generation)
