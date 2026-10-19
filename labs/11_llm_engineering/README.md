Language: **English (default)** | [Polski](README.pl.md)

# Lab 11: designing and evaluating a small LLM inference pipeline

Time: about 2 hours (**105 minutes mandatory core + 15 minutes buffer**).
Builds on
[`lectures/11_llm_engineering/lecture.en.md`](../../lectures/11_llm_engineering/lecture.en.md)
(tokenization, next-token prediction, decoding strategies, chat templates,
KV cache), but this is an **engineering experiment**, not another
classification task: you build and evaluate a small, reproducible
inference pipeline.

**Sections 0-11 are the mandatory core. Sections R1-R2 (at the end of the
notebook) are optional extensions** -- do them only if time remains after
finishing the core. No core grading criterion requires completing the
extensions.

**Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (the same as in the
lecture, Apache 2.0 -- see [`data/registry.yml`](../../data/registry.yml)).
This is a **135 million** parameter model -- expect coherent, grammatical
answers to simple tasks, **not** multi-step reasoning or reliable factual
correctness. The prompts in this lab are in English (the model works
clearly better in English -- verified in the lecture), but all instructions
and your own answers are in English too.

**The pipeline you build**: prompt -> tokenization -> model -> generation
-> decoding -> measurement -> evaluation.

## 0. Setup (provided)

Library imports, device, random seed, `torch.set_num_threads(4)`.

## 1. Model and tokenizer (provided + short task)

We load the model and tokenizer (`revision=` pinned to a specific commit --
reproducibility, not "the latest version").

**Task:** print `model.config.num_attention_heads` and
`model.config.num_key_value_heads`. Are they equal? What does this mean
for KV cache memory cost (see the lecture)?

## 2. Preparing prompts -- your task

Choose **exactly 3 required** prompts, covering:

1. **one** unambiguous prompt of the **information extraction** type from a
   short text (has a clearly defined expected answer -- automatically
   checkable, e.g. via `in`),
2. **a second** unambiguous prompt of a different type (e.g. **format
   transformation** -- a date in a different format),
3. **one qualitative** prompt (a paraphrase or continuation --
   `expected=None`, there is NOT one "correct" answer).

For **every** prompt with an unambiguous answer, write that answer down
**in advance**, before you see what the model generates -- otherwise the
evaluation is not trustworthy (it is easy to "fit" criteria after the
fact).

Data structure: a list of dicts `{"id", "type", "prompt", "expected"}`
(`expected=None` for the qualitative task).

**Extension R1 (optional):** add 2 more prompts (e.g. a question with
context, a second continuation style) with full treatment as in the core.

## 3. Chat template -- your task

Implement `format_prompt(prompt_text)`, which returns `inputs` ready for
`model.generate(**inputs, ...)`, using `tokenizer.apply_chat_template()`
with `add_generation_prompt=True`. **Do not** manually add any special
tokens -- `apply_chat_template` already inserts them (see the lecture,
"Chat templates" section).

**API note (verified in the lecture):** `apply_chat_template(...,
tokenize=True, return_tensors="pt")` **without** `return_dict=True` may
return a different type than you expect in your `transformers` version --
use `return_dict=True` explicitly and access it via `inputs["input_ids"]`,
do not assume a bare tensor.

**Task (code):** verify that your function actually adds the special
tokens (check that the decoded prompt starts with the BOS token/
`<|im_start|>`, not with the prompt's content).

## 4. Generation and correctly decoding the result -- your task

Implement `generate_response(prompt_text, **gen_kwargs)`:
1. format the prompt (section 3),
2. call `model.generate(**inputs, **gen_kwargs, pad_token_id=...)`,
3. **extract ONLY the newly generated tokens** (not the whole `output`,
   which also contains the prompt -- see the lecture, Experiment B/E),
4. decode (`skip_special_tokens=True`),
5. return the text **and** the number of newly generated tokens (needed
   for the throughput measurement in section 6).

**Verification (a provided assertion is waiting in the notebook):** for
`do_sample=False`, two calls to `generate_response()` with the same
prompt must give an **identical** result -- if not, you have a bug in
state/randomness handling, not in the model itself.

## 5. Deterministic generation and sampling -- your task

For **all 3 required** prompts (section 2), generate a **greedy**
(`do_sample=False`) response via `generate_response()`. Additionally, for
a **subset** -- the unambiguous prompt of your choice (e.g. extraction)
**and** the qualitative prompt -- also run **sampling** with two
temperatures (0.7 and 1.3). We do **not** require the full cross-product
"3 prompts x all strategies" -- that would be disproportionate to the
available time.

Use a **fixed seed** (`torch.manual_seed(...)`) before every sampling
call, so results are reproducible across your runs.

**Extension R2 (optional):** also run sampling for the second unambiguous
prompt (and for the R1 prompts, if you added them) -- does sampling help,
hurt, or not matter on tasks the core did not give it to?

## 6. Measuring time and throughput -- your task

For every (prompt, strategy) combination you actually ran in section 5,
measure the **actual** time of the `generate_response()` call
(`time.time()` before/after) and compute **tokens per second** = number
of newly generated tokens / time. Collect the results into one table.

## 7. Effect of prompt length -- your task

Choose **one** prompt and generate a longer version of it (e.g. add
context/history). Measure the generation time for **the same number** of
new tokens for the short and long versions of the prompt -- use
`min_new_tokens == max_new_tokens` to force the same number of new tokens
for both (otherwise differences in generation time, e.g. stopping early
via EOS, would contaminate the measurement of the effect of prompt length
alone -- this pitfall is real, confirmed while preparing this lab). Is
the difference large or small relative to the total time? (See the
lecture: prefill vs. decode.)

## 8. KV cache -- your task

For one prompt and one generation length, compare the time with
`use_cache=True` and `use_cache=False`. Does the cache's advantage grow,
shrink, or stay constant with generation length? (Compare with the
lecture -- is your result consistent?)

## 9. Evaluating quality -- your task

1. **Automatically**, for **both** prompts with an unambiguous answer
   (section 2), on their **greedy** response: check
   `p["expected"].lower() in response.lower()`.
2. **Qualitatively**, for the **qualitative** prompt: evaluate **every**
   strategy run in section 5 (greedy + any sampling) for:
   - **correct instruction-following** -- did the model do what we asked?
   - **completeness** -- is the answer cut off mid-thought?
   - **compliance with the required format** (if the prompt specified
     one),
   - **naturalness/coherence** of the answer,
   - **reproducibility** (for prompts with `do_sample=True`, would
     calling it multiple times with different seeds give a similar
     result?).

**For the qualitative evaluation**: write a **short justification based on
a specific fragment of the answer** -- not "looks decent." For example,
"the paraphrase kept the meaning ('fox'/'dog' are still present), but used
exactly the same 3 words as the original in the key spot -- a weak
paraphrase" is a specific evaluation; "sounds good" is not.

❌ "This answer looks better, so it is better."

✅ We separate the criteria above precisely so we do **not** evaluate with
one, subjective impression (see the lecture).

**Extension R1/R2 (optional):** if you did R1 or R2, evaluate them with
the same scheme.

## 10. Diagnostic task

The following code (in the notebook) **looks** correct -- it raises no
exception (for a short prompt), it generates some text -- but it has a
bug: confusing `max_length` with `max_new_tokens`.

**Task:**
1. Run the code with a short prompt and observe: how many **new** tokens
   were actually produced, relative to how many the code's author
   **wanted** (the `DESIRED_NEW_TOKENS` variable)?
2. Diagnose the cause: what exactly does `max_length` count, and what does
   `max_new_tokens` count? (See the lecture, "The context window and
   truncation" section.)
3. Run the **same** code with a **longer** prompt (long enough that
   `prompt_len >= max_length`) -- what happens now? (Hint: different,
   louder behavior than with the short prompt -- specifically,
   `transformers` raises a `ValueError` directly, instead of silently
   returning too few tokens.)
4. Fix the code (`max_new_tokens=` instead of `max_length=`).
5. Verify the fix: for **both** prompt lengths, the number of newly
   generated tokens must match what was intended.

## 11. Conclusions -- your task

Answer, justified with actual numbers/examples from sections 5-10:

1. Which decoding strategy gave the most **useful** results for your two
   unambiguous-answer tasks? Did sampling (where you tested it) help,
   hurt, or not matter?
2. For the qualitative task -- did a higher temperature give
   **subjectively** better results? Did diversity matter for **this
   specific** task?
3. What is the measured cost (time, tokens/s) of your pipeline? How does
   it change with prompt length and with `use_cache`?
4. What specific limitations of the model/pipeline did you observe
   (section 9)? Give a **specific** example of a wrong/incomplete/
   non-compliant answer, not a generality.
5. What is your conclusion about the quality/cost/configuration trade-off
   -- does a "better" configuration (e.g. higher temperature, a larger
   token budget) always mean a better result for **every** task type from
   section 2?

## Extensions (optional -- R1-R2, if time remains)

- **R1.** Add 2 more prompts (e.g. a question with context, a different
  continuation style) to the list from section 2, and give them the same
  treatment (sections 5, 6, 9) as the required prompts.
- **R2.** Also run sampling (both temperatures) for the second unambiguous
  prompt (and for the R1 prompts) -- does sampling help, hurt, or not
  matter for tasks the core did not give it to?
- Add a **sixth** prompt of your own design, in a different style from
  the suggestions above.
- Measure the effect of `repetition_penalty` on text continuation (does it
  reduce phrase repetition for longer generation?).

## Time budget (approximate, not verified in class)

| Block | Time |
|---|---:|
| Sections 0-1 (setup, model) | 7 min |
| Section 2 (preparing prompts, 3 required) | 8 min |
| Section 3 (chat template) | 10 min |
| Section 4 (generate_response + verification) | 15 min |
| Section 5 (greedy + sampling, reduced scope) | 12 min |
| Section 6 (time/throughput measurement) | 8 min |
| Sections 7-8 (prompt length, KV cache) | 10 min |
| Section 9 (quality evaluation, reduced scope) | 15 min |
| Section 10 (diagnostic task) | 12 min |
| Section 11 (conclusions) | 8 min |
| **Core (sections 0-11)** | **105 min** |
| Buffer | 15 min |
| **Total** | **about 120 min** |

Priority, if time runs short even within the core: sections 2-5 (the
pipeline core) and 9-10 (evaluation and diagnostics). Generating short
answers (<=40 tokens) with this model takes a few seconds on CPU -- the
time budget should go into design and analysis, not waiting for the
model (verified: the full reference solution, including loading the
model, 7 `generate()` calls in section 5, and all of the diagnostics,
runs in a few dozen seconds).

**Extensions R1-R2 are optional** and are not counted toward the core.

## Grading criteria

All criteria below apply to the **core (sections 0-11)** -- none requires
completing extensions R1-R2.

- Exactly 3 required prompts, with `expected` explicitly recorded
  **before** generating the response, for both unambiguous-answer tasks.
- `format_prompt`/`generate_response` correctly separate the prompt from
  the newly generated response -- the assertion from section 4 passes.
- The chat template is used correctly (`apply_chat_template`, with no
  manually added special tokens).
- Greedy run for all 3 prompts; sampling (both temperatures) run for at
  least 2 of them, including the qualitative prompt.
- The time measurement separates at least: short vs. long prompt (section
  7), `use_cache` True vs. False (section 8).
- The quality evaluation (section 9) separates automatic checks (for
  unambiguous-answer tasks) from the qualitative evaluation with a
  specific justification -- not one subjective label for everything.
- The diagnostic task points to the specific cause (`max_length` counts
  the **total** length, `max_new_tokens` counts only new tokens), not a
  generality.
- The conclusions refer to specific numbers/examples from your own run.

A full reference solution exists locally with the instructor
(`decisions/solutions/11_llm_engineering/`) and is not part of this
repository.
