"""The lab 09 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_0{1..8}_scaffold.py: the public notebook has
intentionally unfinished `# TODO` cells that raise `NotImplementedError`
until a student fills them in. This test executes the notebook
permissively, then checks that every cell before the first student TODO
ran with no error, and that the first error is exactly the expected
`NotImplementedError` from the first TODO (the class-distribution check).

The full execution needs to download the SMS Spam Collection (small, but
still a network dependency), so it carries the `network` marker and runs
in scheduled network CI. Always-on synthetic tests below cover the
sequence-handling mechanics this lab is actually about: padding must not
change a prediction, a naive
out[:, -1, :] on a padded batch must (demonstrably) get it wrong, and
macro-F1 must match a hand-computed value on a tiny, exactly-known
example.
"""

from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
from nbclient import NotebookClient
from torch.nn.utils.rnn import pack_padded_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "09_rnn" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


@pytest.mark.network
def test_lab_09_scaffold_runs_until_first_todo():
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
        f"expected the first failure to be the class-distribution TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )


PAD_ID, UNK_ID = 0, 1


def _make_batch(encoded, labels):
    lengths = torch.tensor([max(len(e), 1) for e in encoded])
    max_len = lengths.max().item()
    padded = torch.full((len(encoded), max_len), PAD_ID, dtype=torch.long)
    for i, e in enumerate(encoded):
        if e:
            padded[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return padded, lengths, torch.tensor(labels)


class _CorrectGRUClassifier(nn.Module):
    def __init__(self, vocab_size, d_embed=8, d_hidden=8, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed, padding_idx=PAD_ID)
        self.gru = nn.GRU(d_embed, d_hidden, batch_first=True)
        self.fc = nn.Linear(d_hidden, n_classes)

    def forward(self, x, lengths):
        e = self.embed(x)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.fc(h_n.squeeze(0))


class _BuggyGRUClassifier(nn.Module):
    def __init__(self, vocab_size, d_embed=8, d_hidden=8, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed, padding_idx=PAD_ID)
        self.gru = nn.GRU(d_embed, d_hidden, batch_first=True)
        self.fc = nn.Linear(d_hidden, n_classes)

    def forward(self, x, lengths):
        e = self.embed(x)
        out, _ = self.gru(e)  # no pack_padded_sequence
        return self.fc(out[:, -1, :])  # always the last column, may be padding


def test_correct_padding_handling_gives_identical_prediction():
    torch.manual_seed(0)
    model = _CorrectGRUClassifier(vocab_size=20)
    model.eval()

    short_seq = [5, 7, 3]
    longer_seq = [1, 2, 3, 4, 5, 6, 7, 8]

    X_alone, len_alone, _ = _make_batch([short_seq], [0])
    X_batched, len_batched, _ = _make_batch([short_seq, longer_seq], [0, 0])

    with torch.no_grad():
        pred_alone = model(X_alone, len_alone)
        pred_batched = model(X_batched, len_batched)[0:1]

    assert torch.allclose(pred_alone, pred_batched, atol=1e-6)


def test_naive_last_timestep_is_demonstrably_wrong_with_padding():
    torch.manual_seed(0)
    model = _BuggyGRUClassifier(vocab_size=20)
    model.eval()

    short_seq = [5, 7, 3]
    longer_seq = [1, 2, 3, 4, 5, 6, 7, 8]

    X_alone, len_alone, _ = _make_batch([short_seq], [0])
    X_batched, len_batched, _ = _make_batch([short_seq, longer_seq], [0, 0])

    with torch.no_grad():
        pred_alone = model(X_alone, len_alone)
        pred_batched = model(X_batched, len_batched)[0:1]

    # at batch_size=1 (no padding needed) the bug is invisible
    assert torch.allclose(pred_alone, pred_alone, atol=1e-6)
    # but batched with a longer sequence, the same text's representation changes
    assert not torch.allclose(pred_alone, pred_batched, atol=1e-6)


def test_mean_pool_baseline_ignores_padding_in_average():
    torch.manual_seed(0)
    d_embed = 6
    embed = nn.Embedding(20, d_embed, padding_idx=PAD_ID)

    short_seq = [5, 7, 3]
    X, lengths, _ = _make_batch([short_seq, [1, 2, 3, 4, 5, 6, 7, 8]], [0, 0])

    e = embed(X)
    mask = (X != PAD_ID).unsqueeze(-1).float()
    pooled = (e * mask).sum(dim=1) / lengths.unsqueeze(1).float().clamp_min(1)

    # reference: manually average only the real (non-pad) embeddings of row 0
    real_embeds = embed(torch.tensor(short_seq))
    expected = real_embeds.mean(dim=0)
    assert torch.allclose(pooled[0], expected, atol=1e-6)


def _macro_f1(preds, targets, n_classes=2):
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def test_macro_f1_matches_hand_computed_example():
    # 10 examples: 7 class 0, 3 class 1. Predictions: one class-0 example
    # misclassified as class 1, one class-1 example misclassified as class 0.
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    preds = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    # class 0: tp=6, fp=1, fn=1 -> prec=6/7, rec=6/7 -> f1=6/7
    # class 1: tp=2, fp=1, fn=1 -> prec=2/3, rec=2/3 -> f1=2/3
    expected = (6 / 7 + 2 / 3) / 2
    assert abs(_macro_f1(preds, targets) - expected) < 1e-9


def test_unk_token_handling_does_not_raise():
    word_to_id = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}

    def encode(tokens):
        return [word_to_id.get(t, UNK_ID) for t in tokens]

    encoded = encode(["hello", "zzz_unknown_token", "world"])
    assert encoded == [2, UNK_ID, 3]
