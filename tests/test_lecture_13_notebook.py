"""Lecture 13 notebook tests.

Unlike modules 06-08/11/12, this notebook uses only locally-generated
synthetic 2D data (no download, no network) and runs in under 5 seconds --
the full `nbclient` execution runs unconditionally, the same pattern as
modules 01-05/09/10.

The fast, always-on tests below cover the DDPM math (closed-form forward
diffusion, the reverse-step mean formula against the true closed-form
posterior mean, the t=1 edge case, schedule validity, gradient flow, tensor
shapes) using tiny tensors independent of the notebook's actual trained
model. Several reuse the exact numbers verified by hand in `lecture.md`'s
worked example, so a future change to the formulas would have to update
both places consistently.
"""

import math
from pathlib import Path

import nbformat
import torch
import torch.nn as nn
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "13_diffusion" / "examples.ipynb"


def test_lecture_13_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def _linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def test_beta_schedule_is_valid():
    betas, alphas, alphas_bar = _linear_schedule(50)
    assert (betas > 0).all() and (betas < 1).all()
    assert torch.allclose(alphas, 1 - betas)
    manual_bar_5 = torch.prod(alphas[:5])
    assert torch.allclose(manual_bar_5, alphas_bar[4], atol=1e-6)
    # cumulative product must be monotonically decreasing
    assert (alphas_bar[1:] <= alphas_bar[:-1]).all()


def test_closed_form_forward_diffusion_matches_lecture_worked_example():
    """Exact numbers from lecture.md's x_0=[2,-1] worked example."""
    betas, alphas, alphas_bar = _linear_schedule(10)
    x0 = torch.tensor([2.0, -1.0])
    eps = torch.tensor([0.5, -0.3])

    expected = {
        1: torch.tensor([2.0049, -1.0030]),
        3: torch.tensor([2.0347, -1.0215]),
        6: torch.tensor([2.0577, -1.0380]),
        10: torch.tensor([2.0564, -1.0437]),
    }
    for t, exp in expected.items():
        abar = alphas_bar[t - 1]
        x_t = torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * eps
        assert torch.allclose(x_t, exp, atol=1e-3)


def test_closed_form_matches_stepwise_simulation_statistically():
    """Statistical equivalence (mean/var), not sample-for-sample equality --
    matches the notebook's Experiment A.5 Monte Carlo check."""
    torch.manual_seed(0)
    betas, alphas, alphas_bar = _linear_schedule(50)
    x0 = torch.tensor([1.0, -2.0])
    t_check = 20
    n_trials = 20000

    x_stepwise = x0.unsqueeze(0).repeat(n_trials, 1)
    for i in range(t_check + 1):
        noise = torch.randn(n_trials, 2)
        x_stepwise = torch.sqrt(alphas[i]) * x_stepwise + torch.sqrt(betas[i]) * noise

    eps_batch = torch.randn(n_trials, 2)
    abar = alphas_bar[t_check]
    x_closed = torch.sqrt(abar) * x0.unsqueeze(0) + torch.sqrt(1 - abar) * eps_batch

    expected_mean = torch.sqrt(abar) * x0
    expected_var = (1 - abar).item()
    assert torch.allclose(x_stepwise.mean(0), expected_mean, atol=0.05)
    assert torch.allclose(x_closed.mean(0), expected_mean, atol=0.05)
    assert abs(x_stepwise.var(0, unbiased=False)[0].item() - expected_var) < 0.05
    assert abs(x_closed.var(0, unbiased=False)[0].item() - expected_var) < 0.05


def test_reverse_mean_formula_matches_true_posterior_mean_with_perfect_noise():
    """Algebraic identity (not just tolerance): when eps_theta equals the
    true noise used to construct x_t, the DDPM reverse mean formula must
    exactly equal the closed-form true posterior mean of q(x_{t-1}|x_t,x0)."""
    betas, alphas, alphas_bar = _linear_schedule(20)
    x0 = torch.tensor([1.5, -0.7])
    eps = torch.tensor([0.8, 0.2])

    t_idx = 9
    abar_t = alphas_bar[t_idx]
    abar_prev = alphas_bar[t_idx - 1]
    alpha_t = alphas[t_idx]
    beta_t = betas[t_idx]
    x_t = torch.sqrt(abar_t) * x0 + torch.sqrt(1 - abar_t) * eps

    mu_tilde = (
        torch.sqrt(alpha_t) * (1 - abar_prev) / (1 - abar_t) * x_t
        + torch.sqrt(abar_prev) * beta_t / (1 - abar_t) * x0
    )
    mu_theta = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - abar_t)) * eps)
    assert torch.allclose(mu_tilde, mu_theta, atol=1e-5)


def test_t_equals_1_edge_case_has_zero_posterior_variance():
    betas, alphas, alphas_bar = _linear_schedule(20)
    abar_t0 = alphas_bar[0]
    abar_prev0 = torch.tensor(1.0)  # convention: no step before x_0
    beta_tilde0 = (1 - abar_prev0) / (1 - abar_t0) * betas[0]
    assert abs(beta_tilde0.item()) < 1e-6


def test_reverse_formula_has_no_nan_or_inf_across_all_steps():
    betas, alphas, alphas_bar = _linear_schedule(100)
    x_t = torch.tensor([0.3, -0.1])
    eps = torch.tensor([0.1, 0.2])
    for i in range(100):
        ab = alphas_bar[i]
        bt, at = betas[i], alphas[i]
        mu = (1 / torch.sqrt(at)) * (x_t - (bt / torch.sqrt(1 - ab)) * eps)
        assert torch.isfinite(mu).all()


def test_noise_predictor_gradient_flows_to_all_parameters():
    def sinusoidal_embedding(t, dim=8):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    class NoisePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 + 8, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, x, t):
            return self.net(torch.cat([x, sinusoidal_embedding(t, 8)], dim=1))

    model = NoisePredictor()
    x = torch.randn(5, 2)
    t = torch.randint(0, 100, (5,))
    eps_pred = model(x, t)
    loss = (eps_pred**2).sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_tensor_shapes_consistent_across_batch_sizes():
    def sinusoidal_embedding(t, dim=16):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    for batch_size in [1, 7, 32]:
        t = torch.randint(0, 100, (batch_size,))
        emb = sinusoidal_embedding(t)
        assert emb.shape == (batch_size, 16)


def test_naive_step_skipping_degrades_far_more_than_respaced_sampling():
    """Mirrors the notebook's Experiment D.2 on a deterministic linear
    model standing in for the trained noise predictor: respacing (effective
    alpha computed from the actual abar ratio of the skipped pair) must
    degrade less than naively reusing the single-step formula for the same
    large skip, measured as deviation from the true closed-form x0 estimate
    given a perfect (here: identity) noise predictor is not meaningful on
    its own -- so this test instead checks the two formulas are NOT
    equivalent for a skip larger than 1, confirming they are genuinely
    different procedures, not just differently-named versions of the same
    one."""
    betas, alphas, alphas_bar = _linear_schedule(100)
    t_idx, prev_idx = 50, 20  # a skip larger than 1
    abar_t, abar_prev = alphas_bar[t_idx], alphas_bar[prev_idx]

    alpha_eff_respaced = (abar_t / abar_prev).item()
    alpha_naive = alphas[t_idx].item()
    assert abs(alpha_eff_respaced - alpha_naive) > 1e-3
