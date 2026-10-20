"""Lecture 11 notebook tests.

Unlike modules 09/10, this notebook downloads real model weights
(HuggingFaceTB/SmolLM2-135M-Instruct, ~257MB) -- the full `nbclient`
execution is skipped under CI=true, the same pattern as modules 06-08.

The fast, always-on tests below cover the actual decoding/generation
mechanics using only synthetic logits or a tiny `LlamaForCausalLM` built
from a from-scratch `LlamaConfig` (random weights, no download, no
network) -- never the real downloaded model. Several reuse the exact
numbers verified by hand in `lecture.md`'s worked example, so a future
change to the formulas would have to update both places consistently.
"""

from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn.functional as F
from nbclient import NotebookClient
from transformers import AutoModelForCausalLM, LlamaConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "lectures" / "11_llm_engineering" / "examples.ipynb"


@pytest.mark.network
def test_lecture_11_notebook_executes_cleanly():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()


def _tiny_model(seed=0, vocab_size=50, eos_token_id=1, pad_token_id=1):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def test_softmax_distribution_matches_manual_computation():
    logits = torch.tensor([2.0, 1.5, 0.5, 0.1, -1.0])
    probs = F.softmax(logits, dim=-1)
    manual = torch.exp(logits) / torch.exp(logits).sum()
    assert torch.allclose(probs, manual, atol=1e-6)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))


def test_temperature_matches_lecture_worked_example():
    """Exact numbers from lecture.md's 5-token worked example."""
    logits = torch.tensor([2.0, 1.5, 0.5, 0.1, -1.0])

    p_sharp = F.softmax(logits / 0.5, dim=-1)
    assert torch.allclose(
        p_sharp, torch.tensor([0.6932, 0.2550, 0.0345, 0.0155, 0.0017]), atol=1e-3
    )

    p_neutral = F.softmax(logits / 1.0, dim=-1)
    assert torch.allclose(
        p_neutral, torch.tensor([0.4928, 0.2989, 0.1100, 0.0737, 0.0245]), atol=1e-3
    )

    p_flat = F.softmax(logits / 2.0, dim=-1)
    assert torch.allclose(
        p_flat, torch.tensor([0.3495, 0.2722, 0.1651, 0.1352, 0.0780]), atol=1e-3
    )

    # lower temperature -> sharper (higher max probability); higher -> flatter
    assert p_sharp.max() > p_neutral.max() > p_flat.max()


def test_temperature_has_no_effect_under_greedy_argmax():
    """do_sample=False takes argmax regardless of temperature -- a monotonic
    rescaling of logits never changes which one is largest."""
    logits = torch.tensor([2.0, 1.5, 0.5, 0.1, -1.0])
    argmax_ref = logits.argmax()
    for tau in [0.1, 0.5, 1.0, 2.0, 10.0]:
        assert (logits / tau).argmax() == argmax_ref


def test_top_k_filtering_matches_lecture_worked_example():
    logits = torch.tensor([2.0, 1.5, 0.5, 0.1, -1.0])
    k = 3
    topk_vals, topk_idx = logits.topk(k)
    masked = torch.full_like(logits, float("-inf"))
    masked[topk_idx] = logits[topk_idx]
    p_topk = F.softmax(masked, dim=-1)

    assert set(topk_idx.tolist()) == {0, 1, 2}
    assert torch.allclose(p_topk, torch.tensor([0.5465, 0.3315, 0.1220, 0.0, 0.0]), atol=1e-3)
    assert torch.allclose(p_topk.sum(), torch.tensor(1.0))


def test_top_p_nucleus_filtering_matches_lecture_worked_example():
    logits = torch.tensor([2.0, 1.5, 0.5, 0.1, -1.0])
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    def kept_indices(p_threshold):
        n_keep = (cumsum < p_threshold).sum().item() + 1
        return set(sorted_idx[:n_keep].tolist())

    assert kept_indices(0.7) == {0, 1}
    assert kept_indices(0.9) == {0, 1, 2}


def test_greedy_decoding_matches_manual_argmax_loop():
    """Mirrors the notebook's Experiment B check, on a tiny local-config
    model (no download) instead of the real one."""
    model = _tiny_model(seed=0)
    input_ids = torch.randint(2, 50, (1, 5))

    manual_ids = input_ids.clone()
    manual_tokens = []
    with torch.no_grad():
        for _ in range(5):
            next_id = model(manual_ids).logits[0, -1, :].argmax().item()
            manual_tokens.append(next_id)
            manual_ids = torch.cat([manual_ids, torch.tensor([[next_id]])], dim=1)

    with torch.no_grad():
        gen_out = model.generate(input_ids, max_new_tokens=5, do_sample=False, pad_token_id=1)
    generate_tokens = gen_out[0, input_ids.shape[1] :].tolist()

    assert manual_tokens == generate_tokens


def test_greedy_decoding_is_deterministic_across_repeated_calls():
    model = _tiny_model(seed=1)
    input_ids = torch.randint(2, 50, (1, 4))
    with torch.no_grad():
        out1 = model.generate(input_ids, max_new_tokens=6, do_sample=False, pad_token_id=1)
        out2 = model.generate(input_ids, max_new_tokens=6, do_sample=False, pad_token_id=1)
    assert torch.equal(out1, out2)


def test_eos_stops_generation_before_max_new_tokens():
    model = _tiny_model(seed=0)
    input_ids = torch.randint(2, 50, (1, 5))
    with torch.no_grad():
        natural_next = model(input_ids).logits[0, -1, :].argmax().item()
        out = model.generate(
            input_ids, max_new_tokens=20, do_sample=False,
            eos_token_id=natural_next, pad_token_id=natural_next,
        )
    assert out.shape[1] == input_ids.shape[1] + 1
    assert out[0, -1].item() == natural_next


def test_max_new_tokens_limits_length_without_eos():
    model = _tiny_model(seed=2, eos_token_id=None, pad_token_id=0)
    input_ids = torch.randint(2, 50, (1, 5))
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=12, min_new_tokens=12, do_sample=False,
            eos_token_id=None, pad_token_id=0,
        )
    assert out.shape[1] == input_ids.shape[1] + 12


def test_new_tokens_are_correctly_sliced_and_prompt_is_not_leaked_as_response():
    model = _tiny_model(seed=3, eos_token_id=None, pad_token_id=0)
    input_ids = torch.randint(2, 50, (1, 7))
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=5, min_new_tokens=5, do_sample=False,
            eos_token_id=None, pad_token_id=0,
        )
    # the prompt must still be the prefix, unchanged
    assert torch.equal(out[:, :prompt_len], input_ids)
    # the "response" (what a pipeline would report back to a user) is only
    # the newly generated suffix, never re-including the prompt itself
    response = out[:, prompt_len:]
    assert response.shape[1] == 5
    assert not torch.equal(response, input_ids[:, :5])


def test_throughput_metric_formula():
    n_generated_tokens = 64
    elapsed_seconds = 0.5
    tokens_per_second = n_generated_tokens / elapsed_seconds
    assert tokens_per_second == 128.0


def test_generation_runs_fully_offline_with_local_config_model():
    """No AutoTokenizer.from_pretrained / AutoModelForCausalLM.from_pretrained
    anywhere in this test -- the whole forward+generate pipeline works on a
    model built purely from a local config and raw integer token ids."""
    model = _tiny_model(seed=4)
    input_ids = torch.randint(2, 50, (1, 6))
    with torch.no_grad():
        logits = model(input_ids).logits
        generated = model.generate(input_ids, max_new_tokens=4, do_sample=False, pad_token_id=1)
    assert torch.isfinite(logits).all()
    assert generated.shape[1] >= input_ids.shape[1]
