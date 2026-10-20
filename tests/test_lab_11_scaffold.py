"""The lab 11 scaffold (provided code, not the student TODOs) must run.

Same pattern as tests/test_lab_08_scaffold.py: the notebook's very first
code cell (section 1, loading the real model) needs a network download, so
there is no way to reach even the first student TODO offline -- the full
scaffold-to-first-TODO execution is skipped under CI=true.

The always-on tests below cover the pipeline mechanics the lab's own
deliverables (`format_prompt`, `generate_response`) are built around, and
the diagnostic task's bug, using only a tiny `LlamaForCausalLM` built from
a from-scratch config (no download) -- never the real downloaded model,
and never the private solution's actual function implementations (writing
a complete, reusable version of `generate_response` here would itself be
most of the solution to section 4).
"""

from pathlib import Path

import nbformat
import pytest
import torch
from nbclient import NotebookClient
from transformers import AutoModelForCausalLM, LlamaConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "labs" / "11_llm_engineering" / "lab.ipynb"


def _cell_error(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            return output
    return None


@pytest.mark.network
def test_lab_11_scaffold_runs_until_first_todo():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
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
        f"expected the first failure to be the Q/KV-heads TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )


def _tiny_causal_lm(seed=0, vocab_size=50, eos_token_id=1, pad_token_id=1):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=0,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def test_new_tokens_are_correctly_sliced_from_full_generate_output():
    """This is exactly what the lab's generate_response() must do in
    section 4: take model.generate()'s full output (prompt + new tokens)
    and keep only the suffix -- never the whole thing, and never leaking
    the prompt itself as part of "the response"."""
    model = _tiny_causal_lm(seed=0, eos_token_id=None, pad_token_id=0)
    input_ids = torch.randint(2, 50, (1, 9))
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=6, min_new_tokens=6, do_sample=False,
            eos_token_id=None, pad_token_id=0,
        )
    new_tokens = out[:, prompt_len:]
    assert new_tokens.shape[1] == 6
    assert torch.equal(out[:, :prompt_len], input_ids)
    assert not torch.equal(new_tokens, input_ids[:, :6])


def test_greedy_generation_is_deterministic_for_response_reproducibility_check():
    """Backs the lab's section-4 assertion: two generate_response() calls
    with do_sample=False and the same prompt must give an identical
    result."""
    model = _tiny_causal_lm(seed=1)
    input_ids = torch.randint(2, 50, (1, 5))
    with torch.no_grad():
        out1 = model.generate(input_ids, max_new_tokens=8, do_sample=False, pad_token_id=1)
        out2 = model.generate(input_ids, max_new_tokens=8, do_sample=False, pad_token_id=1)
    assert torch.equal(out1, out2)


def test_sampling_with_fixed_seed_is_reproducible_across_calls():
    model = _tiny_causal_lm(seed=2)
    input_ids = torch.randint(2, 50, (1, 5))
    outs = []
    for _ in range(2):
        torch.manual_seed(42)
        with torch.no_grad():
            outs.append(
                model.generate(
                    input_ids, max_new_tokens=8, do_sample=True,
                    temperature=0.8, top_k=10, pad_token_id=1,
                )
            )
    assert torch.equal(outs[0], outs[1])


def test_max_length_vs_max_new_tokens_diagnostic_bug_is_reproducible():
    """The lab's diagnostic task (section 10): confusing max_length (total
    sequence length) with max_new_tokens (only the new part) silently
    truncates generation when prompt_len < max_length, exactly as
    reproduced here on a tiny local model."""
    model = _tiny_causal_lm(seed=3, eos_token_id=None, pad_token_id=0)
    input_ids = torch.randint(2, 50, (1, 10))
    prompt_len = input_ids.shape[1]
    desired_new_tokens = 20

    with torch.no_grad():
        buggy_out = model.generate(
            input_ids, max_length=desired_new_tokens, do_sample=False,
            eos_token_id=None, pad_token_id=0,
        )
    actual_new_buggy = buggy_out.shape[1] - prompt_len
    assert actual_new_buggy == desired_new_tokens - prompt_len
    assert actual_new_buggy != desired_new_tokens

    with torch.no_grad():
        fixed_out = model.generate(
            input_ids, max_new_tokens=desired_new_tokens, min_new_tokens=desired_new_tokens,
            do_sample=False, eos_token_id=None, pad_token_id=0,
        )
    actual_new_fixed = fixed_out.shape[1] - prompt_len
    assert actual_new_fixed == desired_new_tokens


def test_max_length_raises_when_prompt_already_at_or_past_limit():
    """The "louder" failure mode of the same bug: once the prompt itself is
    at least as long as max_length, transformers refuses outright rather
    than silently generating zero/negative new tokens."""
    model = _tiny_causal_lm(seed=4, eos_token_id=None, pad_token_id=0)
    input_ids = torch.randint(2, 50, (1, 20))
    with torch.no_grad(), pytest.raises(ValueError):
        model.generate(
            input_ids, max_length=20, do_sample=False, eos_token_id=None, pad_token_id=0,
        )


def test_automatic_correctness_check_is_case_insensitive_substring_match():
    """Mirrors the lab's section-9 automatic check: expected text should be
    found in the response regardless of case."""
    expected = "34"
    response_correct = "Name: John Smith\nAge: 34"
    response_wrong = "Name: John Smith, age unspecified"
    assert expected.lower() in response_correct.lower()
    assert expected.lower() not in response_wrong.lower()
