Language: **English (default)** | [Polski](lecture.md)

# Lecture 11 — From the Transformer model to a working text-generation system

## Learning goals

1. Understand autoregressive language modeling as a factorization of the probability of an entire sequence into a product of "next token" conditional distributions.
2. Be able to compute by hand how logits turn into a probability distribution (softmax), and how temperature/top-k/top-p modify that distribution **before** a token is chosen.
3. Understand subword (BPE) tokenization as a compromise between a character vocabulary and a word vocabulary -- and know that the number of tokens is **not** proportional to the number of words, especially across languages.
4. Know the difference between prefill and decode, the role of the KV cache, and be able to measure the actual cost of generation (not just state it).
5. Be able to correctly use `tokenizer.apply_chat_template()` and `model.generate()` through Hugging Face Transformers, understanding exactly what they do.
6. Critically evaluate generated text: distinguish correct instruction-following from factual correctness, and understand the limitations of small language models.

## Why this matters

Modules 8-10 built every piece: embeddings (module 8), sequence modeling (module 9), self-attention and the Transformer block (module 10). This lecture **adds no new Transformer mathematics** -- it asks a different question: given an already-trained model, how does its raw output (logits) turn into the text we actually see on screen? This is an engineering question, not an architectural one -- and it has its own non-trivial decisions (decoding strategy, prompt formatting, memory budget, time cost) that affect the result just as much as the model's architecture does.

## Intuition

A language model, at every step, does not "know" in advance what whole answer it will give -- it computes **one** number per token in the vocabulary: "how likely is it that *this* token comes next, given everything so far?". Generating text is **repeating** this step: compute the distribution, choose a token (somehow), append it to the context, compute the distribution again, repeat.

❌ "An LLM generates the whole answer at once, the way a classifier produces one label."

✅ A typical autoregressive model generates **one token at a time**, each next token depending on all previous ones (including the ones the model itself generated a moment earlier). We will see this literally in Experiment B -- a manual, step-by-step generation loop, compared against `model.generate()`.

## Formal model: autoregressive factorization

Let $x_1,\ldots,x_T$ be a sequence of tokens (identifiers from a vocabulary of size $V$ -- the same $V$ as in module 8's `nn.Embedding(V, d)`; $T$ = sequence length, as in modules 9-10). The chain rule of probability (exact, with no assumption about the model) gives:

$$P(x_1,\ldots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_1,\ldots,x_{t-1})$$

(for $t=1$: $P(x_1 \mid x_1,\ldots,x_0) := P(x_1)$, i.e. the distribution with no context at all -- an empty condition). This is **not a model assumption** -- it is a mathematical identity true for **every** joint distribution. The model's assumption lies elsewhere: that $P(x_t \mid x_1,\ldots,x_{t-1})$ can be **well approximated** by a neural network $\theta$ -- hence the notation $P_\theta(x_t \mid x_{<t})$ ($x_{<t} := x_1,\ldots,x_{t-1}$, a standard shorthand, used from here on).

**A note on notation.** Here $x_t$ denotes a **token identifier** (an integer $\in\{1,\ldots,V\}$), not a vector -- unlike $x_t\in\mathbb R^d$ in module 9 (there: a continuous embedding/feature). This is not a new meaning replacing the old one -- it is the first step of the pipeline: $x_t$ (id) -> $\mathbf e_t$ (embedding, bold, module 8) -> ... -> $z_t$ (logits, below). $V$ = vocabulary size, extending the notation from module 8 (there: an `nn.Embedding` constructor parameter, here: the formal dimension of the logits and the output distribution).

## How the model turns context into a distribution: logits and softmax

For context $x_{<t}$, the model (an encoder/decoder-only Transformer from module 10, **not re-derived here**) produces a vector of **logits** for position $t$:

$$z_t = f_\theta(x_{<t}) \in \mathbb R^V$$

($z$ = logits, module 2 onward -- here extended with the time index $t$, the same pattern as $h_t$, $c_t$, $\delta_t$ in module 9). $z_t$ has **one entry per vocabulary token** -- not directly interpretable as a probability (it can be negative, it does not sum to 1). Softmax turns it into a distribution:

$$\hat y_t = \text{softmax}(z_t), \qquad \hat y_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^V e^{z_{t,j}}}$$

(an extension of $\hat y=\text{softmax}(z)$ from module 8, there with no time index -- one-shot classification; here $\hat y_t$ is a distribution over the **entire vocabulary** for **this specific generation position**, not over a task's set of classes). $\hat y_{t,i} = P_\theta(x_t = i \mid x_{<t})$ -- the $i$-th component of $\hat y_t$ is the probability that the next token is specifically token $i$.

❌ "The highest probability for one token guarantees the best whole answer."

✅ $\hat y_t$ only tells you about **one** step. Choosing the most probable token at every step (greedy decoding, below) does **not** guarantee the most probable **entire** sequence -- these are two different optimization problems. A verified numerical counterexample is in the "Greedy decoding" section below.

## Temperature: a transformation of the distribution, not a choice of token

**Temperature** scales the logits **before** softmax:

$$\hat y_t^{(\tau)} = \text{softmax}\!\left(\frac{z_t}{\tau}\right), \qquad \tau > 0$$

**A note on notation.** We denote temperature $\tau$, **not** $T$ -- in this course, since module 9, $T$ has meant **sequence length**, and both concepts appear in the same paragraph (temperature while generating a sequence of length $T$) -- the collision would be real, not merely technical, so here we genuinely introduce a new symbol instead of adding a disambiguation note to the existing one. $\tau$ is unused earlier in the course.

- $\tau < 1$: dividing by a number $<1$ **increases** the differences between logits before softmax -> a **sharper** distribution (more concentrated on the already-most-probable tokens).
- $\tau > 1$: the logits get "flattened" -> a **flatter** distribution (closer to uniform).
- $\tau \to 0^+$: $\hat y_t^{(\tau)} \to$ a distribution concentrated 100% on $\arg\max_i z_{t,i}$ -- formally equivalent to greedy decoding in the limit.

❌ "Temperature increases the model's creativity."

✅ This is an imprecise anthropomorphism. Temperature is a **purely mathematical transformation of the probability distribution** -- it does not change *what* the model "knows," only how **concentrated** or **flat** the distribution is from which something is **later** chosen. Temperature itself **does not choose a token** -- it needs a selection procedure for that (sampling, below). What's more: **if `do_sample=False` (greedy), temperature has no effect on the result at all** -- greedy takes the $\arg\max$ regardless of how the distribution was rescaled (a monotonic transformation of logits does not change which one is largest). Attributing an effect to temperature in a mode that does not use it is a configuration error, not a subtlety.

## Token-selection procedures: deterministic vs. random

**Greedy decoding**: $x_t = \arg\max_i \hat y_{t,i}$ -- deterministic, always the same result for the same context and model.

❌ "Greedy decoding chooses the globally most probable whole sequence."

✅ Greedy optimizes **locally, step by step**, not globally. **A verified counterexample** (2 steps, vocabulary $\{A,B\}\to\{X,Y\}$): step 1, $P(A)=0.5498 > P(B)=0.4502$ -- greedy chooses $A$. Step 2 given $A$: $P(X|A)=P(Y|A)=0.5$. Step 2 given $B$: $P(X|B)=0.99$. Greedy produces the sequence $(A,X)$ with $P=0.2749$. The **globally best** sequence is $(B,X)$ with $P=0.4457$ -- larger, but greedy never finds it, because it never "looks ahead" when choosing step 1. Verified by enumerating all $4$ sequences in the notebook, not declared.

**Sampling**: we draw $x_t \sim \hat y_t^{(\tau)}$ -- non-deterministic (with no fixed seed, a different result on different runs; **with** a fixed PyTorch seed, reproducible -- verified in the notebook).

### Top-k

We restrict sampling to the $k$ tokens with the highest $z_t$, zero out the probability of the rest, and **renormalize** so the retained probabilities sum to $1$.

**A note on notation.** $k$ here is the number of retained tokens -- the technique's name ("top-k") is standardized in the literature; the symbol $k$ collides with the filter size in module 6 and the number of neighbors in module 8's k-NN -- the context (decoding vs. convolution vs. search) unambiguously distinguishes them, following the same pattern as the earlier $k$ collisions in this course.

### Top-p (nucleus sampling)

We sort tokens by probability in decreasing order, take the **smallest** leading set whose **cumulative** probability exceeds a threshold $p$, discard the rest, and renormalize.

**A note on notation.** We denote the top-p threshold $p$ -- a different meaning than the dropout probability $p$ in module 5 (there: the probability of zeroing a neuron). Different topics (decoding vs. regularization), the same symbol, following this course's established rule: context disambiguates, a note names it explicitly.

**The key difference between top-k and top-p**: top-k keeps a **fixed number** of tokens regardless of what the distribution looks like (even if one token has 99% of the mass, we still keep $k$ tokens); top-p keeps a **variable number** of tokens, matched to the distribution's actual concentration (for a very concentrated distribution, top-p may keep only $1$-$2$ tokens; for a flat one -- far more).

## Worked example: a 5-token vocabulary

Vocabulary: `cat, dog, house, sea, xyz123`. Logits: $z=[2.0,\ 1.5,\ 0.5,\ 0.1,\ -1.0]$.

**Softmax, $\tau=1$:** $\hat y = [0.4928,\ 0.2989,\ 0.1100,\ 0.0737,\ 0.0245]$ (sum $=1$, verified).

**Effect of temperature** (same $z$, different $\tau$):

| $\tau$ | cat | dog | house | sea | xyz123 |
|---|---|---|---|---|---|
| $0.5$ (sharper) | $0.6932$ | $0.2550$ | $0.0345$ | $0.0155$ | $0.0017$ |
| $1.0$ | $0.4928$ | $0.2989$ | $0.1100$ | $0.0737$ | $0.0245$ |
| $2.0$ (flatter) | $0.3495$ | $0.2722$ | $0.1651$ | $0.1352$ | $0.0780$ |

A smaller $\tau$ -> a distribution more concentrated on `cat`; a larger $\tau$ -> more uniform -- **the same** output $z$, just a different transformation.

**Top-k=3** (on the $\tau=1$ distribution): keeps $\{$`cat`,`dog`,`house`$\}$, renormalization -> $[0.5465,\ 0.3315,\ 0.1220,\ 0,\ 0]$.

**Top-p**: the cumulative probabilities (decreasing) are $0.4928,\ 0.7918,\ 0.9017,\ 0.9755,\ 1.0$. At $p=0.7$: we keep $\{$`cat`,`dog`$\}$ (cumulative $0.7918 \geq 0.7$, `cat` alone is not enough: $0.4928 < 0.7$). At $p=0.9$: we keep $\{$`cat`,`dog`,`house`$\}$ (cumulative $0.9017 \geq 0.9$).

All numbers verified programmatically in the notebook (Experiment B/C), not just computed on paper.

## Tokenization: from characters to subwords

Three levels of input granularity, with real trade-offs:

- **Characters**: a small vocabulary (for the Latin alphabet: dozens of symbols), but **very long** sequences (every character is a step) -- computationally expensive, hard to learn long-range dependencies at the character level.
- **Words**: shorter sequences, but the vocabulary would have to be **enormous** (hundreds of thousands++ of unique forms) and **never complete** -- every new word, proper name, or typo is an "unknown" token.
- **Subword (BPE and similar)**: a compromise -- frequent words become **one** token, rare/unknown words split into **known pieces** (subwords), with no need for an explicit "`<UNK>`" token in most cases.

**BPE (Byte-Pair Encoding), the intuition without implementing it from scratch**: we start with individual characters (bytes), iteratively **merge the most frequent adjacent pair** into one new token, and repeat up to a fixed vocabulary size. Frequent sequences (e.g. "ing", "the") become a single token early; rare ones stay split longer. We do not implement this algorithm -- we use a ready-made tokenizer trained together with the model (Experiment A).

**Token IDs and special tokens**: every token has a unique integer identifier (a row in the embedding table, module 8). Models add **special** tokens, beyond the "ordinary" text vocabulary:

- **BOS** (*beginning of sequence*) -- marks the start of a sequence.
- **EOS** (*end of sequence*) -- signals that the model has "decided" generation is done; `generate()` **stops** when it generates this (verified in Experiment B).
- **PAD** -- padding to a common length within a batch (the same principle as in module 9: padding is not content).

❌ "Every model has a different, separate token for BOS/EOS/PAD."

✅ True for some, but **they are not always three different tokens** -- in the model used in this module, `pad_token` and `eos_token` are **the same** token (verified: `tokenizer.pad_token_id == tokenizer.eos_token_id`). Consequence: you cannot assume that checking "is this pad" and "is this eos" is always two different conditions -- it depends on the specific tokenizer, and must be verified, not guessed.

❌ "A token is a synonym for a word."

✅ Verified in Experiment A: an English sentence of 9 words -> 10 tokens (close to 1:1), but a Polish sentence of 7 words -> **25** tokens (about 3.6 tokens per word) -- the same tokenizer, dramatically different granularity, because it was trained mostly on English text. **You cannot assume a fixed word:token ratio, especially across languages** -- this is a measured fact, not a guess.

## Prefill and decode: two different stages of inference

Generating a response to a prompt involves **two qualitatively different** stages:

1. **Prefill**: the model processes the **entire prompt at once** (one forward pass, parallel across positions -- the same property as in module 10: self-attention has no sequential dependency between positions **within** a single forward pass). Result: logits for the token *after* the prompt, plus (if `use_cache=True`) the saved K/V representations for every prompt position.
2. **Decode**: the model generates **one token at a time**, each step being a **separate** forward pass (this time only for **one new** position, reusing the saved K/V from previous positions -- hence the KV cache, below). Decode is **sequential by definition** -- step $t+1$ needs the token generated at step $t$.

Engineering consequence: prefill is (for the same number of tokens) usually **much faster per token** than decode, because it is **one** parallel pass, not $T$ **separate** passes.

## KV cache: what it is, and what it is not

Without a cache, generating token $t{+}1$ would require **recomputing** self-attention for **all** positions $1,\ldots,t$ from scratch -- even though their K/V representations **do not change** between steps (self-attention is causal, module 10: position $i$ does not depend on $j>i$). The KV cache **remembers** the computed $K,V$ for already-processed positions, so they are not recomputed.

❌ "The KV cache is a persistent store of the model's knowledge."

✅ The KV cache stores **only** the computed $K,V$ representations for positions **of this specific, ongoing inference run** -- reset at the start of a new conversation, containing none of the knowledge learned during training (that lives in the parameters $\theta$, not in the cache). It also does not increase the model's reasoning ability -- it is purely a **computational optimization**, mathematically equivalent to computing without a cache (differing only in speed).

**Cache memory cost** (an exact formula, not an approximation):

$$\text{bytes} = 2 \times N \times H_{kv} \times d_{\text{head}} \times T \times \text{bytes}_{\text{dtype}}$$

where $N$ = number of layers (Transformer blocks) -- the same $N$ as "$N$ stacked Transformer blocks" in module 10, not the general "number of layers" symbol from module 2/6 ($K$), because $K$ (the key matrix) and $V$ (the value matrix) appear a moment later in **this same** formula/paragraph -- using $K$ for "number of layers" here would collide **locally**, not just at the course scale. $H_{kv}$ = the number of **KV heads** (see below -- **not** always equal to the number of query heads!), $d_{\text{head}}$ = the dimension of one head, $T$ = sequence length (the same $T$ as always), the factor $2$ = $K$ and $V$ separately.

❌ "Every model has an identical number of query heads (Q) and KV heads."

✅ **Not true, and this is not a theoretical remark** -- the model used in this module has **9 Q heads, but only 3 KV heads** (*grouped-query attention*, GQA: several query heads **share** the same KV head). Verified directly from the model's configuration and from the actual shape of the cache tensor: `keys.shape == (batch, 3, T, 64)`, not `(batch, 9, T, 64)`. GQA is a **deliberate** design decision that **reduces** the cache's memory cost (fewer heads to store), not a bug or an oversight -- and it means the formula above **must** use $H_{kv}$, not "the number of heads" in general.

Verified programmatically (Experiment E): the formula gave $737{,}280$ bytes for $T=16$ positions of this specific model -- **bit-exact** agreement with the actual size of the `past_key_values` tensors (not an estimate).

## Chat templates and message roles

**Instruct** models are trained on conversations in a specific, fixed textual format -- not on "plain" text. This format (the roles `system`/`user`/`assistant`, special tokens delimiting messages) is stored in `tokenizer.chat_template` and applied via `apply_chat_template()`.

❌ "A chat template is cosmetic text formatting, it does not affect the result."

✅ The model **did not see**, during training, text in any format other than the one it was fine-tuned on (instruction tuning, below) -- feeding a prompt **without** the correct format is feeding an input from a **distribution the model has never seen**, which can dramatically lower answer quality, not just "look inelegant." This is part of the **contract** with the model, not styling.

A concrete example (the model used in this module, ChatML format):
```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```
`add_generation_prompt=True` adds the **last** line (`<|im_start|>assistant\n`) -- a signal to the model that "it's your turn now," without which the model "does not know" it should start generating a reply rather than continuing the user's message.

❌ "You have to add the special tokens (BOS/EOS) to the prompt yourself."

✅ `apply_chat_template()` **already** inserts them (verified: the text after applying the template starts with `<|im_start|>`, not with the content). Adding them manually **on top of** what the template already inserted produces duplicated/incorrectly formatted tokens -- a classic, real configuration bug (the topic of the lab's diagnostic task).

### Conversation history is not the model's memory

❌ "The model 'remembers' previous messages of the conversation."

✅ The model is **stateless** between calls -- every `generate()` call only receives **what is in its input** for that specific call. If we want the model to "remember" earlier messages, we **ourselves** must append the entire conversation history so far (all `user`/`assistant` messages) to the input of the next call -- which means **token cost grows with conversation length** (every subsequent turn pays for **all** the history so far, not just the new message), up to the context window limit.

## The context window and truncation

A model has a **maximum** number of positions it can accept as input (`max_position_embeddings` in the configuration -- $8192$ for the model used here). This is a **hard** architectural limit (positional encoding/cache are undefined beyond it), not a generation parameter.

❌ "`max_length` and `max_new_tokens` are the same thing."

✅ **No** -- `max_length` is a limit on the **total** length (prompt + generated), `max_new_tokens` is a limit **only** on the newly generated part. Confusing them is a real, common bug: setting `max_length` too small relative to an **already long** prompt can **immediately** truncate the input or generate nothing new at all -- the topic of the lab's diagnostic task.

A long prompt **eats into** the context window's budget -- if the prompt plus the desired response length exceeds the limit, something has to be **truncated** (usually the oldest part of the conversation history) -- an engineering decision, not automatic "magic."

## Base vs. instruct, and where they come from (briefly)

- **Base model**: trained via **pretraining** -- next-token prediction on a huge text corpus, with no special conversation format. Good at continuing text, not at "following instructions."
- **Supervised fine-tuning (SFT)**: additional training on (instruction, desired response) pairs in a fixed format (chat template, above) -- the model learns the conversation **format** and how to **follow** instructions.
- **Preference tuning** (e.g. DPO -- *Direct Preference Optimization*, used for the model in this module): an additional step in which the model learns to **prefer** one of two responses over the other (judged by humans or another model) -- further style/quality alignment, not a new architecture.

❌ "An instruct model is a different architecture from a base model, designed to follow instructions better."

✅ **The same architecture** (the same Transformer) -- the difference is **solely** in what and how it was further trained on (SFT + preferences) **after** pretraining. "SmolLM2-135M" and "SmolLM2-135M-Instruct" differ in their weights (the effect of additional training), not in the number of layers or the attention mechanism.

We do not develop the training process (neither SFT nor DPO) as a main topic here -- that is deliberately out of scope for this module (modules 12+ return to training in other contexts).

## The Hugging Face Transformers workflow

Three classes that cover almost everything needed here:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION)

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
```

`AutoTokenizer`/`AutoModelForCausalLM` select the correct tokenizer/architecture class based on the configuration saved alongside the model -- you do not need to know the architecture type in advance. `revision=` **pins** an exact weight commit (a verified, reproducible version -- not "whatever is currently under this name"). `pipeline("text-generation", ...)` exists as a convenient high-level abstraction -- we show it briefly in the notebook, but the main experiment **deliberately** uses the lower level (`generate`, not `pipeline`), so the inference stages stay visible, not hidden.

## Latency, throughput, and cost

- **Latency**: the time to get the (first/whole) result.
- **Throughput**: tokens per second -- how much "work" the system does over time.
- These two measures are **not** interchangeable: high throughput (lots of tokens/s in total, e.g. batching many requests at once) does not mean low latency for **one, specific** request -- and vice versa.

❌ "Shorter inference time automatically means a better model."

✅ Inference time depends on model size, prompt/response length, hardware, and configuration (cache, precision) -- **not** on answer quality. A smaller, faster model can give worse answers; a larger, slower one -- better ones. These are two **independent** axes of evaluation, not one.

Measured directly (Experiment E, CPU, this machine -- **not** a universal benchmark): `use_cache=True` dramatically faster than `use_cache=False`, with the advantage **growing** with generation length (because without a cache, the cost of **every** step grows with position -- $O(T)$ per step, $O(T^2)$ total; with a cache -- $O(1)$ per step, $O(T)$ total, for a fixed model size).

## Evaluating quality and limitations

❌ "This answer looks better, so it is better."

✅ Subjective "looks better" is not an evaluation procedure. We separate **at least**: (1) correct instruction-following (did the model do what we asked -- often automatically checkable), (2) factual correctness (where it can be verified against a source), (3) completeness, (4) compliance with the required format (also often automatically checkable), (5) reproducibility (is the result stable, or random under sampling). We develop this in the lab as a systematic procedure, not an impression.

❌ "A small local model offers the quality of a large commercial model."

✅ The model used in this module has $135$ million parameters -- an order of magnitude (or more) fewer than contemporary large commercial models. We expect coherent, grammatical answers to simple tasks (verified), **not** multi-step reasoning, deep specialized knowledge, or reliable factual correctness on complex questions.

## Hallucinations and basic risks (briefly -- not a separate lecture)

- **Hallucinations**: the model generates text that **sounds** credible and grammatical, **regardless** of whether it is actually true -- the model has no built-in mechanism for distinguishing "I know this for certain" from "this sounds probable." Correct syntax and grammar **do not guarantee** factual correctness -- these are two entirely different criteria.
- **Instability under sampling**: the same prompt, a different result on a different run (with no fixed seed) -- relevant when evaluating ("was that bad luck, or a systematic problem?").
- **Sensitivity to prompt phrasing**: a different wording of the same question can give a different quality of answer -- not just different text.
- **Prompt injection**: if the model processes text from an **untrusted** source (e.g. webpage content, an email, a user document) **together** with a system instruction, that untrusted text may contain wording that **looks like instructions** ("ignore the above and do X") -- treating data content as a command is a security risk, not just a theoretical curiosity.
- **Confidential data**: sending prompts to an external service (a commercial model API) means handing the prompt's content to a third party -- a consequence to consider **before**, not after.
- **Licenses**: a model's weights and training data have their own licenses/terms (this module: Apache 2.0, verified -- see `data/registry.yml`) -- not every model is "free to use for any purpose" just because it is publicly available.
- **Generated content as a "verified fact"**: treating the model's answer as automatically true (without verification) is a user/system error, not the model's -- the model never promised correctness.

**Three different, independent criteria -- none guarantees the others:**

1. **Correct code execution** -- if the model generates code, does that code **actually run** without error (a question of programming-language syntax, not natural language).
2. **Correctness of the generated text's syntax** -- is the text grammatically/structurally well formed (e.g. in this module: is the answer fluent, grammatical English).
3. **Factual correctness** -- is the content of the answer **actually true/accurate**.

Code can run without error and give a **wrong** answer (a logic bug, not a syntax one). Text can be grammatically flawless and **factually false** (a hallucination, above). None of these three criteria automatically implies the others -- when evaluating generated content (the "Evaluating quality" section above, and the lab), each must be checked **separately**.

This is an **introduction** to these topics (awareness, not a deep analysis) -- full LLM safety is a separate field, out of scope for this module.

## What we observe

All the numbers below come from an actual execution of `examples.en.ipynb` (CPU, this machine -- not a universal benchmark) -- see the notebook for the full output.

- **Experiment A (tokenization).** As in the table above: the English sentence $0.90$ words/token, the Polish one $0.28$ (about $3.6\times$ more tokens per word), numbers split into individual digits (`1234.56` -> `1`,`2`,`3`,`4`,`.`,`5`,`6`), the `§` character (outside ASCII) gives a byte-level BPE artifact (`ĠÂ§`), rare/long words are split into sensible subwords, never into a single `<UNK>` token.
- **Experiment B.** For the prompt "The capital of France is" (after the chat template, $35$ tokens): the top-5 next tokens are `The` ($p=0.3113$), `B` ($0.0674$), `"` ($0.0616$), `Bon` ($0.0353$), `Paris` ($0.0344$) -- the model is **not** immediately certain of one token. Temperature on the same distribution: $\tau=0.5$ -> `The` jumps to $0.866$; $\tau=2.0$ -> `The` drops to $0.026$ (the same $z_t$, a different concentration). The manual generation loop (8 steps, stopped by EOS) gave, token by token, **exactly** the same result as `model.generate(do_sample=False)`: `The capital of France is Paris.` -- verified programmatically (`==` on the id list, not a declaration).
- **Experiment C.** On an open-ended, creative prompt (a robot discovers music): greedy, sampling ($\tau=0.7$/$1.5$), top-k=10, top-p=0.9 gave **five clearly different** continuations. On a concentrated, factual prompt ("capital of France"): all five strategies gave **the same** answer ("Paris") with only minor differences in the continuation **after** it -- the same configuration, radically different observed diversity, because it depends on the **shape of the distribution** for a given prompt, not only on the sampling parameters. With a fixed seed (`torch.manual_seed(42)`): two runs gave **identical** text.
- **Experiment D.** Plain text (no chat template, 7 tokens) gave **empty** generated text (`''`) -- the model immediately generated EOS, because it never saw this format during fine-tuning; with the chat template (37 tokens, with an automatically added system prompt) it gave the correct answer "The capital of France is Paris." Conversation history: after the first turn $36$ tokens, after the second (with the full history appended) $72$ tokens -- the model answered "What is my name?" correctly **only** because the first turn had been explicitly appended to the input.
- **Experiment E.** `use_cache=True` clearly faster than `False`, with the advantage growing with length: at $16$ new tokens $0.18$s vs $0.44$s ($2.5\times$); at $128$ tokens $1.29$s vs $5.64$s ($4.4\times$) -- a growing advantage, consistent with theory ($O(T)$ per step without cache vs. $O(1)$ per step with cache). A longer prompt ($437$ vs. $31$ tokens, the same generation=$32$ tokens): $0.47$s vs. $0.32$s -- a longer prefill, but decode still dominates at this scale. The cache memory formula ($2NH_{kv}d_{\text{head}}T\cdot\text{bytes}$) gave $737{,}280$ bytes for $T=16$ -- **exactly** (bit-exact, not an approximation) matching the actual size of the `past_key_values.layers[i].keys/values` tensors. Model: $9$ Q heads, **$3$ KV heads** -- GQA, not a bug.

## Common misconceptions

(A collected summary -- each has a full ❌/✅ block above, in the relevant section)

1. An LLM does not generate the whole answer at once in typical autoregressive decoding.
2. A token is not a synonym for a word.
3. A higher temperature does not guarantee a better answer (and has no effect at all when `do_sample=False`).
4. Greedy decoding need not choose the globally most probable complete sequence.
5. A longer prompt need not improve quality.
6. The model's context is not persistent memory -- conversation history must be explicitly appended to every call.
7. The KV cache does not store the model's entire learned knowledge -- only the K/V of the current run.
8. An instruct model is not a different architecture just because it follows instructions better -- it differs by further training (SFT + preferences).
9. A chat template is not purely cosmetic text formatting.
10. Lower inference time does not automatically mean a better model.
11. Correctly generated (grammatical) text can contain factual errors.
12. A local model need not offer the quality of a commercially hosted large model.

## Engineering implications

- **Cost grows with generation length, not just model size.** Decode is sequential by definition -- doubling `max_new_tokens` roughly doubles decode time (measured: throughput in tok/s is relatively stable with a cache, but the **total** time grows linearly with the number of generated tokens).
- **The KV cache is a memory-for-time trade-off**, not a "free" speedup -- the cache size grows **linearly** with sequence length and **linearly** with the number of layers/KV heads (formula above) -- for very long contexts/many parallel conversations, cache memory can become a real constraint.
- **GQA (fewer KV heads than Q heads) is a deliberate optimization** reducing cache memory cost -- not every model from the same architecture "family" has an identical configuration, you have to check `config.json`, not guess.
- **Prefill and decode have different cost characteristics** -- a long prompt costs mainly in prefill (one, parallel pass), a long answer costs mainly in decode (sequential steps) -- optimizing one does not automatically optimize the other.
- **Prompt format (chat template) is part of the contract with the model**, not a stylistic detail -- a wrong format is a silent quality drop, not always an exception/error.
- **Confusing `max_length` with `max_new_tokens`** is a real, common configuration bug with specific, bad consequences (a truncated input, or no new content at all) -- not a theoretical mistake.

## What to remember

- $P(x_1,\ldots,x_T) = \prod_{t=1}^T P_\theta(x_t\mid x_{<t})$ -- an exact identity, not an assumption; the assumption is that the neural network approximates each factor well.
- Logits -> softmax -> distribution; temperature scales the logits **before** softmax and **has no effect** when `do_sample=False`.
- Top-k keeps a fixed number of tokens, top-p keeps a variable number matched to the distribution's concentration.
- Subword tokenization is a characters/words compromise -- the number of tokens per word is **not** fixed, especially across languages (measured: 1.1 for English, 3.6 for Polish, the same tokenizer).
- Prefill (parallel, the whole prompt) and decode (sequential, token by token) are different stages with different cost characteristics; the KV cache speeds up decode by avoiding recomputing unchanged K/V.
- The chat template and BOS/EOS/PAD are part of the contract with the model, not cosmetics -- and they are not always three different tokens.
- An instruct model = the same Transformer + SFT + preference tuning, not a different architecture.

## Questions for students

1. You have a distribution $\hat y_t$ that is very concentrated (one token has $99\%$ of the mass). Will top-k=$50$ and top-p=$0.9$ give the same effective set of tokens to sample from? Justify your answer.
2. Why do `do_sample=False` and `temperature=5.0` give an identical result to `do_sample=False` and `temperature=1.0`?
3. You have a model with $40$ layers, $8$ KV heads, head dimension $128$, fp16 precision ($2$ bytes). What is the KV cache memory cost for a sequence of length $4096$? How does this cost change if the number of KV heads drops to $2$ (GQA)?
4. You have a multi-turn conversation with the model. Why does the cost (in tokens) of **every subsequent** turn grow, even if the new user message is very short?

## References

- Ben Allal, L., Lozhkov, A., Bakouch, E., et al. (2025). *SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model*. arXiv:2502.02737.
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL (introduced BPE to NLP).
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR (nucleus sampling / top-p).
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 30 (see also module 10 -- the Transformer architecture, not re-derived here).
- Pope, R., et al. (2022). *Efficiently Scaling Transformer Inference*. arXiv:2211.05102 (KV cache, inference cost).
- Hugging Face Transformers documentation: *Generation strategies*, `https://huggingface.co/docs/transformers/generation_strategies`.
- Hugging Face Transformers documentation: *Chat templates*, `https://huggingface.co/docs/transformers/chat_templating`.
- Model card: *HuggingFaceTB/SmolLM2-135M-Instruct*, `https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct`.
