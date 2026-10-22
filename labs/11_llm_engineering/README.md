Language: **English** | [Polski](README.pl.md)

# Lab 11. An inference pipeline for a small language model

The estimated time is about 2 hours. The required core takes 105 minutes, with a 15-minute buffer. This lab uses tokenization, decoding strategies, chat templates, and the KV cache from [`lecture.en.md`](../../lectures/11_llm_engineering/lecture.en.md).

Sections 0-11 form the required core. Extensions R1 and R2 are optional.

The selected model is `HuggingFaceTB/SmolLM2-135M-Instruct`, available under Apache 2.0 according to [`data/registry.yml`](../../data/registry.yml). It has 135 million parameters. It should handle simple English instructions but does not guarantee multi-step reasoning or factual correctness.

The pipeline contains a prompt, tokenization, the model, generation, decoding, measurement, and evaluation.

## 0. Setup

Imports, device selection, the random seed, and `torch.set_num_threads(4)` are provided.

## 1. Model and tokenizer

The model and tokenizer are loaded with `revision` pinned to a specific commit.

**Task.** Print `model.config.num_attention_heads` and `model.config.num_key_value_heads`, compare them, and explain the effect on KV cache memory.

## 2. Preparing prompts

Prepare exactly three required prompts.

1. An unambiguous information-extraction task over a short text.
2. A different task with one clear result, such as date-format conversion.
3. A qualitative task such as paraphrasing or continuation, with `expected=None`.

For the two unambiguous tasks, the expected answer must be written down before running the model. This prevents fitting the criterion to the observed output.

The required structure is a list of dictionaries `{"id", "type", "prompt", "expected"}`.

Optional R1 adds two prompts evaluated in the same way.

## 3. Chat template

`format_prompt(prompt_text)` should return input ready for `model.generate`. It must use `tokenizer.apply_chat_template()` with `add_generation_prompt=True`.

Special tokens must not be added manually. The call should explicitly set `return_dict=True`, because without it the return type may depend on the `transformers` version. Identifiers are available through `inputs["input_ids"]`.

**Code task.** The decoded prompt should begin with BOS or `<|im_start|>`, not the user content.

## 4. Generation and decoding

`generate_response(prompt_text, **gen_kwargs)` should:

1. format the prompt,
2. call `model.generate(**inputs, **gen_kwargs, pad_token_id=...)`,
3. select only newly generated tokens,
4. decode them with `skip_special_tokens=True`,
5. return the text and the number of new tokens.

A provided assertion requires identical results from two calls with the same prompt and `do_sample=False`.

## 5. Greedy decoding and sampling

Generate a greedy response with `do_sample=False` for all three prompts. Sampling at temperatures 0.7 and 1.3 is required for one selected unambiguous prompt and the qualitative prompt.

Set the seed with `torch.manual_seed(...)` before every sampling call. The complete cross-product of prompts and strategies is not required.

Optional R2 extends sampling to the second unambiguous prompt and any prompts from R1.

## 6. Runtime and throughput

For every executed prompt-strategy pair, measure the actual `generate_response()` runtime and calculate new tokens per second. Collect the results in one table.

## 7. Prompt length

Prepare short and long versions of one prompt. Both must generate the same number of new tokens, enforced by `min_new_tokens == max_new_tokens`. Otherwise, EOS could distort the measurement of prompt-length effects.

Relate the result to the distinction between prefill and decode.

## 8. KV cache

For one prompt and a shared generation length, compare `use_cache=True` and `use_cache=False`. Determine how the cache advantage changes with response length.

## 9. Quality evaluation

For both unambiguous prompts, check the greedy answer automatically:

```python
p["expected"].lower() in response.lower()
```

For the qualitative prompt, every executed strategy requires an assessment of:

- instruction following,
- completeness,
- compliance with the requested format,
- naturalness and coherence,
- reproducibility under sampling.

The justification should identify a specific response fragment. A general statement that the text looks good is not a sufficient criterion.

## 10. Diagnostic task

The notebook code confuses `max_length` with `max_new_tokens`. For a short prompt, it generates text but does not provide the intended budget of new tokens.

The task requires:

1. comparing the actual number of new tokens with `DESIRED_NEW_TOKENS`,
2. explaining the scopes of `max_length` and `max_new_tokens`,
3. repeating the test with `prompt_len >= max_length` and observing the `ValueError`,
4. replacing `max_length` with `max_new_tokens`,
5. confirming the correct token count for both prompt lengths.

## 11. Conclusions

Answers should refer to results from sections 5-10.

1. Which strategy was most useful for the unambiguous tasks, and what was the effect of sampling?
2. Did higher temperature improve the qualitative task, and was diversity useful?
3. What were the runtime and throughput, and how did prompt length and cache affect them?
4. Which specific errors, omissions, or format violations appeared?
5. What trade-off emerged between quality, cost, and configuration? A larger budget or higher temperature need not help every task.

## Optional extensions

- **R1.** Two additional prompts with complete measurement and evaluation.
- **R2.** Both sampling temperatures for the second unambiguous prompt and prompts from R1.
- A sixth prompt in a different style.
- A measurement of how `repetition_penalty` affects repetition in a longer answer.

## Estimated time allocation

| Block | Time |
|---|---:|
| Sections 0-1, setup and model | 7 min |
| Section 2, three prompts | 8 min |
| Section 3, chat template | 10 min |
| Section 4, `generate_response` and assertion | 15 min |
| Section 5, greedy and sampling | 12 min |
| Section 6, runtime and throughput | 8 min |
| Sections 7-8, length and cache | 10 min |
| Section 9, quality evaluation | 15 min |
| Section 10, diagnostics | 12 min |
| Section 11, conclusions | 8 min |
| **Core** | **105 min** |
| Buffer | 15 min |
| **Total** | **about 120 min** |

If time is limited, sections 2-5 and 9-10 have priority. Short responses up to 40 tokens take a few seconds on a CPU, and the complete reference solution runs in a few dozen seconds.

## Grading criteria

- There are exactly three required prompts, and `expected` for both unambiguous tasks is recorded before generation.
- `format_prompt` and `generate_response` separate the prompt from the new response, and the assertion passes.
- The chat template is used without manually adding special tokens.
- Greedy covers three prompts, and both temperatures cover at least two prompts, including the qualitative one.
- Measurements compare a short and long prompt and enabled and disabled cache.
- Evaluation separates automatic checks from a qualitative justification based on a response fragment.
- The diagnosis correctly distinguishes total `max_length` from new `max_new_tokens`.
- Conclusions use specific results from the run.

A complete reference solution is stored locally by the instructor in `decisions/solutions/11_llm_engineering/` and is not part of this repository.
