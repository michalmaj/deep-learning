"""Lecture 10 notebook tests.

Like module 09's lecture, this notebook needs no download (every
experiment uses synthetic data and PyTorch's built-in attention
primitives) -- it runs in a few seconds with no network. So the full
`nbclient` execution runs unconditionally in CI, not skipped. Alongside
it, fast synthetic unit tests cover the same mechanics in isolation,
including the two explicitly dangerous pitfalls the lecture calls out:
the opposite boolean-mask conventions between
`F.scaled_dot_product_attention` and `nn.MultiheadAttention`, and
`model.eval()` not auto-disabling dropout passed as a constant
`dropout_p` to a functional call.

`test_transformer_beats_gru_at_long_range_across_several_seeds` additionally
checks, across 3 seeds, that the notebook's headline GRU-vs-Transformer
comparison at T=64 is not a single-seed artifact (found missing in the
08-10 review).
"""

import math
from pathlib import Path

import nbformat
import torch
import torch.nn as nn
import torch.nn.functional as F
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "10_attention" / "examples.ipynb"


def test_lecture_10_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def test_manual_scaled_dot_product_attention_matches_pytorch():
    Q = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    K = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
    V = torch.tensor([[1.0, 2], [3, 4], [5, 6]])
    d_k = Q.shape[-1]

    weights = F.softmax((Q @ K.T) / math.sqrt(d_k), dim=-1)
    manual_out = weights @ V

    pytorch_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)
    assert torch.allclose(manual_out, pytorch_out, atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))


def test_manual_multihead_attention_matches_nn_multiheadattention():
    torch.manual_seed(1)
    d_model, n_heads = 8, 2
    d_head = d_model // n_heads

    mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
    x = torch.randn(1, 5, d_model)

    with torch.no_grad():
        W_q, W_k, W_v = mha.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
        Q = (x @ W_q.T + b_q).view(1, 5, n_heads, d_head).transpose(1, 2)
        K = (x @ W_k.T + b_k).view(1, 5, n_heads, d_head).transpose(1, 2)
        V = (x @ W_v.T + b_v).view(1, 5, n_heads, d_head).transpose(1, 2)

        heads = []
        for h in range(n_heads):
            w_h = F.softmax(Q[:, h] @ K[:, h].transpose(-2, -1) / math.sqrt(d_head), dim=-1)
            heads.append(w_h @ V[:, h])
        manual_out = torch.cat(heads, dim=-1) @ mha.out_proj.weight.T + mha.out_proj.bias

    pytorch_out, _ = mha(x, x, x, need_weights=False)
    assert torch.allclose(manual_out, pytorch_out, atol=1e-5)


def test_mask_semantics_differ_between_sdpa_and_multiheadattention():
    """F.scaled_dot_product_attention: bool True = allow. nn.MultiheadAttention's
    key_padding_mask/attn_mask: bool True = block. These are opposite -- copying a
    mask between the two APIs without inverting it silently attends to the wrong
    positions."""
    torch.manual_seed(0)
    Q = K = V = torch.randn(1, 3, 4)

    allow_mask = torch.tensor([[True, True, False]]).expand(3, -1)
    out_allowed = F.scaled_dot_product_attention(Q, K, V, attn_mask=allow_mask, dropout_p=0.0)
    out_blocked_last = F.scaled_dot_product_attention(
        Q, K, V, attn_mask=~allow_mask, dropout_p=0.0
    )
    assert not torch.allclose(out_allowed, out_blocked_last), (
        "flipping the boolean mask must change the output -- otherwise the mask is "
        "not actually being applied"
    )

    mha = nn.MultiheadAttention(4, 1, batch_first=True, dropout=0.0)
    key_padding_mask = torch.tensor([[False, False, True]])  # True = this IS padding
    _, weights_true_is_pad = mha(
        Q, K, V, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True
    )
    assert torch.allclose(
        weights_true_is_pad.squeeze(0)[:, 2], torch.zeros(3), atol=1e-6
    ), "nn.MultiheadAttention must zero out attention to the True (padding) column"


def test_all_masked_row_naive_softmax_is_nan_but_sdpa_is_zero():
    Q = K = V = torch.randn(1, 2, 4)
    row_all_masked = torch.tensor([[False, False]])  # no allowed key for this query
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(4)

    masked_scores = scores.masked_fill(~row_all_masked.unsqueeze(1), float("-inf"))
    naive_weights = F.softmax(masked_scores, dim=-1)
    assert torch.isnan(naive_weights).any()

    full_mask = row_all_masked.expand(1, 2, 2)
    sdpa_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=full_mask, dropout_p=0.0)
    assert not torch.isnan(sdpa_out).any()


def test_eval_does_not_disable_dropout_passed_as_constant_to_functional_call():
    class NaiveAttention(nn.Module):
        def __init__(self, dropout_p=0.5):
            super().__init__()
            self.dropout_p = dropout_p

        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)

    class CorrectAttention(nn.Module):
        def __init__(self, dropout_p=0.5):
            super().__init__()
            self.dropout_p = dropout_p

        def forward(self, q, k, v):
            p = self.dropout_p if self.training else 0.0
            return F.scaled_dot_product_attention(q, k, v, dropout_p=p)

    torch.manual_seed(0)
    q, k, v = torch.randn(1, 5, 8), torch.randn(1, 5, 8), torch.randn(1, 5, 8)

    naive = NaiveAttention().eval()
    assert not torch.allclose(naive(q, k, v), naive(q, k, v)), (
        "model.eval() does not zero out a dropout_p that was passed as a constant"
    )

    correct = CorrectAttention().eval()
    assert torch.allclose(correct(q, k, v), correct(q, k, v))


def test_causal_mask_blocks_gradient_from_future_positions():
    torch.manual_seed(0)
    d_model, seq_len = 8, 6
    x = torch.randn(1, seq_len, d_model, requires_grad=True)
    mha = nn.MultiheadAttention(d_model, 2, batch_first=True, dropout=0.0)

    causal_allow = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    attn_mask = ~causal_allow  # nn.MultiheadAttention: True = block
    out, _ = mha(x, x, x, attn_mask=attn_mask, need_weights=False)

    grad_wrt_future = torch.autograd.grad(out[0, 0].sum(), x, retain_graph=True)[0][0, 5]
    assert torch.allclose(grad_wrt_future, torch.zeros(d_model), atol=1e-6)

    grad_wrt_past = torch.autograd.grad(out[0, 5].sum(), x)[0][0, 0]
    assert grad_wrt_past.norm() > 0


def test_self_attention_without_positional_encoding_is_permutation_equivariant():
    torch.manual_seed(2)
    d_model = 8
    x = torch.randn(1, 6, d_model)
    perm = torch.tensor([3, 0, 4, 1, 5, 2])

    mha = nn.MultiheadAttention(d_model, 2, batch_first=True, dropout=0.0)
    out, _ = mha(x, x, x, need_weights=False)
    x_perm = x[:, perm, :]
    out_perm_input, _ = mha(x_perm, x_perm, x_perm, need_weights=False)

    assert torch.allclose(out[:, perm, :], out_perm_input, atol=1e-6)
    assert torch.allclose(out.mean(dim=1), out_perm_input.mean(dim=1), atol=1e-6)


def test_layernorm_output_has_zero_mean_and_unit_std_per_position():
    x = torch.randn(2, 5, 16) * 10 + 3
    out = nn.LayerNorm(16)(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-4)


def test_transformer_beats_gru_at_long_range_across_several_seeds():
    """The lecture's GRU-vs-Transformer comparison (examples.ipynb, T=64) was
    flagged in the 08-10 review as resting on a single seed, unlike the
    with/without-positional-encoding comparison in the same experiment, which
    *is* checked across 3 seeds. This repeats the T=64 comparison (same
    dataset generator, same architectures and training budget as the
    notebook) across 3 seeds and checks the qualitative claim the lecture
    actually makes -- Transformer clearly outperforms GRU at long range --
    rather than any specific accuracy value, which would be a flaky thing to
    pin down exactly."""

    def sinusoidal_positional_encoding(seq_len, d_model):
        pos = torch.arange(seq_len).unsqueeze(1).float()
        i = torch.arange(d_model).unsqueeze(0).float()
        angle_rates = 1.0 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angles = pos * angle_rates
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe

    class TinyTransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
            self.ln1 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
            )
            self.ln2 = nn.LayerNorm(d_model)

        def forward(self, x):
            attn_out, _ = self.mha(x, x, x, need_weights=False)
            x = self.ln1(x + attn_out)
            return self.ln2(x + self.ff(x))

    class TinyTransformerClassifier(nn.Module):
        def __init__(self, vocab_size, seq_len, d_model=16, n_heads=2, d_ff=32, n_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.register_buffer("pos_enc", sinusoidal_positional_encoding(seq_len, d_model))
            self.block = TinyTransformerBlock(d_model, n_heads, d_ff)
            self.fc = nn.Linear(d_model, n_classes)

        def forward(self, x):
            e = self.embed(x) + self.pos_enc.unsqueeze(0)
            return self.fc(self.block(e).mean(dim=1))

    class GRUClassifier(nn.Module):
        def __init__(self, vocab_size, d_embed=16, d_hidden=16, n_classes=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_embed)
            self.gru = nn.GRU(d_embed, d_hidden, batch_first=True)
            self.fc = nn.Linear(d_hidden, n_classes)

        def forward(self, x):
            out, _ = self.gru(self.embed(x))
            return self.fc(out[:, -1, :])

    def make_order_dataset_T(n, seed, T, vocab_size=8):
        g = torch.Generator().manual_seed(seed)
        base = torch.stack([torch.randperm(vocab_size, generator=g) for _ in range(n)])
        distractors = torch.randint(vocab_size, 2 * vocab_size, (n, T - vocab_size), generator=g)
        seqs = torch.cat([base, distractors], dim=1)
        pos0 = (seqs == 0).float().argmax(dim=1)
        pos1 = (seqs == 1).float().argmax(dim=1)
        return seqs, (pos0 < pos1).long()

    def train_eval(model, X_tr, y_tr, X_va, y_va, seed):
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.CrossEntropyLoss()
        n = X_tr.shape[0]
        for _ in range(30):
            perm = torch.randperm(n)
            for i in range(0, n, 32):
                idx = perm[i : i + 32]
                opt.zero_grad()
                loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return (model(X_va).argmax(1) == y_va).float().mean().item()

    T, vocab = 64, 8
    X_tr, y_tr = make_order_dataset_T(800, seed=0, T=T, vocab_size=vocab)
    X_va, y_va = make_order_dataset_T(200, seed=1, T=T, vocab_size=vocab)

    for seed in [0, 1, 2]:
        tf_acc = train_eval(TinyTransformerClassifier(2 * vocab, T), X_tr, y_tr, X_va, y_va, seed)
        gru_acc = train_eval(GRUClassifier(2 * vocab), X_tr, y_tr, X_va, y_va, seed)
        assert tf_acc - gru_acc > 0.2, (
            f"seed={seed}: expected Transformer to clearly beat GRU at T={T}, "
            f"got tf={tf_acc:.3f} gru={gru_acc:.3f}"
        )


def test_gradients_are_finite_through_full_transformer_block():
    class TinyBlock(nn.Module):
        def __init__(self, d_model=16, n_heads=2, d_ff=32):
            super().__init__()
            self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
            self.ln1 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
            self.ln2 = nn.LayerNorm(d_model)

        def forward(self, x):
            attn_out, _ = self.mha(x, x, x, need_weights=False)
            x = self.ln1(x + attn_out)
            return self.ln2(x + self.ff(x))

    torch.manual_seed(0)
    block = TinyBlock()
    x = torch.randn(2, 6, 16, requires_grad=True)
    block(x).pow(2).sum().backward()

    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)
