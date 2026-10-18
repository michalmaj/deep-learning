"""The lab 13 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_11_scaffold.py and tests/test_lab_12_scaffold.py,
except this lab uses only locally-generated synthetic 2D data (no download,
no network), so the full scaffold-to-first-TODO execution runs
unconditionally -- there is no CI skip needed here.

The always-on tests below cover the DDPM mechanics the lab's own
deliverables are built around (schedule validity, q_sample, the reverse-step
formula, the t=1 edge case, MMD on known cases) using tiny, independently
reimplemented versions -- never the private solution's exact code, the same
way test_lab_12_scaffold.py never imports its solution's functions.
"""

from pathlib import Path

import nbformat
import torch
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "13_diffusion" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


def test_lab_13_scaffold_runs_until_first_todo():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
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
        f"expected the first failure to be the data-visualization TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )


def _linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def test_schedule_produces_small_terminal_abar_for_T_500():
    """Backs the lab's section-2 check: T=500 with this beta range should
    leave a small residual signal, unlike T=100 (discussed in the lecture)."""
    _, _, alphas_bar = _linear_schedule(500)
    assert alphas_bar[-1].item() < 0.01


def _q_sample(x0, t, eps, alphas_bar):
    a = alphas_bar[t].unsqueeze(1)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps


def test_q_sample_matches_stepwise_simulation_statistically():
    torch.manual_seed(0)
    betas, alphas, alphas_bar = _linear_schedule(50)
    x0 = torch.tensor([0.5, -1.5])
    t_check = 15
    n_trials = 20000

    x_stepwise = x0.unsqueeze(0).repeat(n_trials, 1)
    for i in range(t_check + 1):
        noise = torch.randn(n_trials, 2)
        x_stepwise = torch.sqrt(alphas[i]) * x_stepwise + torch.sqrt(betas[i]) * noise

    t_batch = torch.full((n_trials,), t_check, dtype=torch.long)
    eps_batch = torch.randn(n_trials, 2)
    x0_batch = x0.unsqueeze(0).expand(n_trials, -1)
    x_closed = _q_sample(x0_batch, t_batch, eps_batch, alphas_bar)

    expected_mean = torch.sqrt(alphas_bar[t_check]) * x0
    assert torch.allclose(x_stepwise.mean(0), expected_mean, atol=0.05)
    assert torch.allclose(x_closed.mean(0), expected_mean, atol=0.05)


def test_reverse_step_handles_last_step_without_added_noise():
    """The lab's section-6 requirement: at t_idx == 0, no random term is
    added -- the update is deterministic (mu only)."""
    betas, alphas, alphas_bar = _linear_schedule(20)
    x = torch.tensor([[0.2, -0.1]])
    eps_pred = torch.tensor([[0.05, 0.03]])

    def reverse_step(x, t_idx, add_noise, seed=0):
        torch.manual_seed(seed)
        alpha_t, beta_t, abar_t = alphas[t_idx], betas[t_idx], alphas_bar[t_idx]
        mu = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - abar_t)) * eps_pred)
        if add_noise:
            return mu + torch.sqrt(beta_t) * torch.randn_like(x)
        return mu

    out1 = reverse_step(x, 0, add_noise=False, seed=1)
    out2 = reverse_step(x, 0, add_noise=False, seed=2)
    assert torch.equal(out1, out2)  # deterministic: different seeds, same result

    out3 = reverse_step(x, 5, add_noise=True, seed=1)
    out4 = reverse_step(x, 5, add_noise=True, seed=2)
    assert not torch.equal(out3, out4)  # stochastic: different seeds, different result


def test_reverse_step_formula_has_no_nan_or_inf():
    betas, alphas, alphas_bar = _linear_schedule(100)
    x = torch.tensor([[0.1, 0.2]])
    eps_pred = torch.tensor([[0.05, -0.02]])
    for t_idx in range(100):
        alpha_t, beta_t, abar_t = alphas[t_idx], betas[t_idx], alphas_bar[t_idx]
        mu = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - abar_t)) * eps_pred)
        assert torch.isfinite(mu).all()


def _rbf_mmd2(X, Y, sigma=None):
    XX, YY, XY = torch.cdist(X, X) ** 2, torch.cdist(Y, Y) ** 2, torch.cdist(X, Y) ** 2
    if sigma is None:
        combined = torch.cat([XX.flatten(), YY.flatten(), XY.flatten()])
        sigma = combined[combined > 0].median().sqrt().item()
    gamma = 1.0 / (2 * sigma**2)
    Kxx, Kyy, Kxy = torch.exp(-gamma * XX), torch.exp(-gamma * YY), torch.exp(-gamma * XY)
    nx, ny = X.shape[0], Y.shape[0]
    sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (nx * (nx - 1))
    sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (ny * (ny - 1))
    sum_xy = Kxy.sum() / (nx * ny)
    return (sum_xx + sum_yy - 2 * sum_xy).item(), sigma


def test_mmd_distinguishes_same_from_different_distributions():
    """Backs the lab's section-7 requirement to verify MMD on a known case
    before using it on generated samples."""
    torch.manual_seed(0)
    A = torch.randn(300, 2)
    B_same = torch.randn(300, 2)
    B_shifted = torch.randn(300, 2) + torch.tensor([5.0, 5.0])

    mmd_same, sigma = _rbf_mmd2(A, B_same)
    mmd_diff, _ = _rbf_mmd2(A, B_shifted, sigma=sigma)
    assert mmd_same < 0.01
    assert mmd_diff > mmd_same * 10


def test_mean_nn_distance_is_shape_sensitive_baseline_check():
    """A trivial Gaussian baseline sitting in the 'hole' between two moons
    should show a much larger mean NN-distance to real data than samples
    that actually respect the concave shape -- this is the metric the lab
    uses to catch what MMD alone might not."""
    torch.manual_seed(0)

    def sample_two_moons(n):
        import math

        n1 = n // 2
        theta1 = torch.rand(n1) * math.pi
        moon1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=1)
        theta2 = torch.rand(n - n1) * math.pi
        moon2 = torch.stack([1 - torch.cos(theta2), 1 - torch.sin(theta2) - 0.5], dim=1)
        return torch.cat([moon1, moon2], dim=0)

    real = sample_two_moons(500)
    # a point placed in the empty region between the two moons
    hole_points = torch.tensor([[0.5, 0.2]]).repeat(50, 1) + 0.02 * torch.randn(50, 2)

    def mean_nn_distance(samples, reference):
        return torch.cdist(samples, reference).min(dim=1).values.mean().item()

    nn_real_subset = mean_nn_distance(real[:50], real[50:])
    nn_hole = mean_nn_distance(hole_points, real)
    assert nn_hole > nn_real_subset * 2
