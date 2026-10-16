"""The lab 10 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_0{1..9}_scaffold.py: the public notebook has
intentionally unfinished `# TODO` cells that raise `NotImplementedError`
until a student fills them in. This test executes the notebook
permissively, then checks that every cell before the first student TODO
ran with no error, and that the first error is exactly the expected
`NotImplementedError` from the first TODO (balance/length-independence
check).

Unlike module 09's lab, this notebook needs no download -- all data is
synthetic (random token sequences), exactly like module 09's *lecture*.
So the full execution runs unconditionally in CI, not skipped (the
established refinement: only skip when a notebook actually needs
network access). Always-on synthetic tests below cover the mechanics
this lab is actually about: the correct vs. inverted `key_padding_mask`
convention for `nn.MultiheadAttention`, and the resulting NaN on a
fully-unpadded sequence that the diagnostic task (section 12) is built
around.
"""

from pathlib import Path

import nbformat
import torch
import torch.nn as nn
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "10_attention" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


def test_lab_10_scaffold_runs_until_first_todo():
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
        f"expected the first failure to be the balance/length-independence TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )


PAD_ID, MAX_LEN = 12, 8


class _TinyBlock(nn.Module):
    def __init__(self, d_model=8, n_heads=2, d_ff=16):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask):
        # need_weights=True (not the default False) to match the lab notebook,
        # which inspects per-head attention weights in section 10 -- this matters
        # here because it forces nn.MultiheadAttention's explicit-softmax path
        # rather than the fused fast path, and only the explicit path produces
        # NaN (rather than a safe zero) for an all-masked row.
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ff(x))


def _masked_mean_pool(out, key_padding_mask, lengths):
    real_mask = (~key_padding_mask).unsqueeze(-1).float()
    return (out * real_mask).sum(dim=1) / lengths.unsqueeze(1).float()


def test_correct_padding_mask_gives_identical_representation_regardless_of_batching():
    torch.manual_seed(0)
    embed = nn.Embedding(13, 8, padding_idx=PAD_ID)
    block = _TinyBlock()

    def run(x, lengths):
        key_padding_mask = x == PAD_ID  # True = padding, correct nn.MultiheadAttention convention
        e = embed(x)
        out = block(e, key_padding_mask)
        return _masked_mean_pool(out, key_padding_mask, lengths)

    short_seq = torch.tensor([[1, 2, 3, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]])
    longer_seq = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    batched = torch.cat([short_seq, longer_seq], dim=0)

    with torch.no_grad():
        alone = run(short_seq, torch.tensor([3]))
        batched_out = run(batched, torch.tensor([3, 8]))

    assert torch.allclose(alone, batched_out[0:1], atol=1e-5)


def test_inverted_mask_convention_is_finite_with_padding_but_nan_without_it():
    """The diagnostic bug (section 12): key_padding_mask = (x != PAD_ID), i.e.
    True = real token, is the opposite of what nn.MultiheadAttention expects.
    With padding present, at least one key stays "allowed" (a padding column,
    under the inverted convention) so the output is finite but wrong. Once a
    sequence has NO padding at all, every key is "blocked" for every query --
    the all-masked-row case from the lecture -- and the output is NaN."""
    torch.manual_seed(0)
    embed = nn.Embedding(13, 8, padding_idx=PAD_ID)
    block = _TinyBlock()

    def run_buggy(x):
        key_padding_mask = x != PAD_ID  # BUG: inverted convention
        e = embed(x)
        return block(e, key_padding_mask)

    seq_with_padding = torch.tensor([[1, 2, 3, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]])
    seq_full_length = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    with torch.no_grad():
        out_with_padding = run_buggy(seq_with_padding)
        out_full_length = run_buggy(seq_full_length)

    assert torch.isfinite(out_with_padding).all()
    assert torch.isnan(out_full_length).all()


def test_fixing_mask_convention_restores_finite_output_for_full_length_sequence():
    torch.manual_seed(0)
    embed = nn.Embedding(13, 8, padding_idx=PAD_ID)
    block = _TinyBlock()

    seq_full_length = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    key_padding_mask = seq_full_length == PAD_ID  # fixed convention: True = padding

    with torch.no_grad():
        e = embed(seq_full_length)
        out = block(e, key_padding_mask)

    assert torch.isfinite(out).all()


def test_mean_pool_baseline_ignores_padding_in_average():
    torch.manual_seed(0)
    d_model = 6
    embed = nn.Embedding(13, d_model, padding_idx=PAD_ID)

    short_seq = [1, 2, 3]
    x = torch.tensor([short_seq + [PAD_ID] * 5, [4, 5, 6, 7, 8, 9, 10, 11]])
    lengths = torch.tensor([3, 8])

    e = embed(x)
    real_mask = (x != PAD_ID).unsqueeze(-1).float()
    pooled = (e * real_mask).sum(dim=1) / lengths.unsqueeze(1).float()

    expected = embed(torch.tensor(short_seq)).mean(dim=0)
    assert torch.allclose(pooled[0], expected, atol=1e-6)


def test_class_balance_independent_of_length_by_construction():
    """Sanity-check the dataset generator's core invariant (section 1): token
    position of A vs B is drawn uniformly without replacement from the actual
    sequence length, so P(A before B) = 0.5 regardless of length."""
    torch.manual_seed(0)
    min_len, max_len = 10, 40

    def make_dataset(n, seed):
        g = torch.Generator().manual_seed(seed)
        lengths = torch.randint(min_len, max_len + 1, (n,), generator=g)
        labels = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            length = lengths[i].item()
            pos_ab = torch.randperm(length, generator=g)[:2]
            labels[i] = 1 if pos_ab[0].item() < pos_ab[1].item() else 0
        return lengths, labels

    lengths, labels = make_dataset(2000, seed=0)
    balance = labels.float().mean().item()
    assert 0.45 < balance < 0.55

    mean_len_0 = lengths[labels == 0].float().mean().item()
    mean_len_1 = lengths[labels == 1].float().mean().item()
    assert abs(mean_len_0 - mean_len_1) < 1.5
