"""Lecture 09 notebook tests.

Unlike modules 06-08, this notebook needs no download (every experiment
uses synthetic data and `torch.nn.RNN`/`LSTM`/`GRU` with random or
explicit weights) -- it runs in a few seconds with no network. So the
full `nbclient` execution runs unconditionally in CI, not skipped.
Alongside it, fast synthetic unit tests cover the same mechanics in
isolation for quicker, more targeted failures: a hand-rolled RNN cell
against `nn.RNN`, LSTM/GRU gate equations against `nn.LSTM`/`nn.GRU`,
and that shared weights really are shared across time steps.
"""

from pathlib import Path

import nbformat
import torch
import torch.nn as nn
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "09_rnn" / "examples.ipynb"


def test_lecture_09_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def test_manual_rnn_cell_matches_nn_rnn():
    d_in, d_h, seq_len = 2, 2, 3
    Wx = torch.tensor([[0.5, -0.3], [0.1, 0.8]])
    Wh = torch.tensor([[0.2, 0.0], [-0.1, 0.4]])
    b = torch.tensor([0.1, -0.1])
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    h0 = torch.zeros(d_h)

    h_prev = h0
    manual_states = []
    for t in range(seq_len):
        h_t = torch.tanh(Wx @ X[t] + Wh @ h_prev + b)
        manual_states.append(h_t)
        h_prev = h_t
    manual_states = torch.stack(manual_states)

    rnn = nn.RNN(d_in, d_h, num_layers=1, nonlinearity="tanh", batch_first=True)
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(Wx)
        rnn.weight_hh_l0.copy_(Wh)
        rnn.bias_ih_l0.copy_(b)
        rnn.bias_hh_l0.zero_()
    out, _ = rnn(X.unsqueeze(0), h0.view(1, 1, d_h))

    assert torch.allclose(manual_states, out.squeeze(0), atol=1e-6)


def test_shared_weights_mean_every_timestep_uses_same_matrices():
    torch.manual_seed(0)
    rnn = nn.RNN(3, 4, batch_first=True)
    n_params_one_step = sum(p.numel() for p in rnn.parameters())

    x_short = torch.randn(1, 3, 3)
    x_long = torch.randn(1, 50, 3)
    out_short, _ = rnn(x_short)
    out_long, _ = rnn(x_long)
    n_params_after = sum(p.numel() for p in rnn.parameters())

    assert n_params_one_step == n_params_after, "parameter count must not depend on T"
    assert out_short.shape == (1, 3, 4)
    assert out_long.shape == (1, 50, 4)


def test_manual_lstm_gates_match_nn_lstm():
    torch.manual_seed(0)
    d_in, d_h, seq_len = 3, 2, 4
    X = torch.randn(1, seq_len, d_in)

    lstm = nn.LSTM(d_in, d_h, batch_first=True)
    out, (hn, cn) = lstm(X)

    W_ih, W_hh = lstm.weight_ih_l0, lstm.weight_hh_l0
    b_ih, b_hh = lstm.bias_ih_l0, lstm.bias_hh_l0
    W_ii, W_if, W_ig, W_io = W_ih.chunk(4, dim=0)
    W_hi, W_hf, W_hg, W_ho = W_hh.chunk(4, dim=0)
    b_ii, b_if, b_ig, b_io = b_ih.chunk(4, dim=0)
    b_hi, b_hf, b_hg, b_ho = b_hh.chunk(4, dim=0)

    h_prev, c_prev = torch.zeros(d_h), torch.zeros(d_h)
    manual_h = []
    for t in range(seq_len):
        x_t = X[0, t]
        i_t = torch.sigmoid(W_ii @ x_t + b_ii + W_hi @ h_prev + b_hi)
        f_t = torch.sigmoid(W_if @ x_t + b_if + W_hf @ h_prev + b_hf)
        g_t = torch.tanh(W_ig @ x_t + b_ig + W_hg @ h_prev + b_hg)
        o_t = torch.sigmoid(W_io @ x_t + b_io + W_ho @ h_prev + b_ho)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        manual_h.append(h_t)
        h_prev, c_prev = h_t, c_t

    assert torch.allclose(torch.stack(manual_h), out.squeeze(0), atol=1e-6)
    assert torch.allclose(h_prev, hn.squeeze(), atol=1e-6)
    assert torch.allclose(c_prev, cn.squeeze(), atol=1e-6)


def test_manual_gru_gates_match_nn_gru():
    torch.manual_seed(0)
    d_in, d_h, seq_len = 3, 2, 4
    X = torch.randn(1, seq_len, d_in)

    gru = nn.GRU(d_in, d_h, batch_first=True)
    out, hn = gru(X)

    W_ih, W_hh = gru.weight_ih_l0, gru.weight_hh_l0
    b_ih, b_hh = gru.bias_ih_l0, gru.bias_hh_l0
    W_ir, W_iz, W_in = W_ih.chunk(3, dim=0)
    W_hr, W_hz, W_hn = W_hh.chunk(3, dim=0)
    b_ir, b_iz, b_in = b_ih.chunk(3, dim=0)
    b_hr, b_hz, b_hn = b_hh.chunk(3, dim=0)

    h_prev = torch.zeros(d_h)
    manual_h = []
    for t in range(seq_len):
        x_t = X[0, t]
        r_t = torch.sigmoid(W_ir @ x_t + b_ir + W_hr @ h_prev + b_hr)
        z_t = torch.sigmoid(W_iz @ x_t + b_iz + W_hz @ h_prev + b_hz)
        n_t = torch.tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_prev + b_hn))
        h_t = (1 - z_t) * n_t + z_t * h_prev
        manual_h.append(h_t)
        h_prev = h_t

    assert torch.allclose(torch.stack(manual_h), out.squeeze(0), atol=1e-6)
    assert torch.allclose(h_prev, hn.squeeze(), atol=1e-6)


def test_lstm_and_gru_param_ratios_match_rnn_exactly():
    torch.manual_seed(0)
    d_in, d_h = 5, 7
    rnn_params = sum(p.numel() for p in nn.RNN(d_in, d_h, batch_first=True).parameters())
    gru_params = sum(p.numel() for p in nn.GRU(d_in, d_h, batch_first=True).parameters())
    lstm_params = sum(p.numel() for p in nn.LSTM(d_in, d_h, batch_first=True).parameters())

    assert gru_params == 3 * rnn_params
    assert lstm_params == 4 * rnn_params
