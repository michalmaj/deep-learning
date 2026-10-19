Language: **English (default)** | [Polski](lecture.md)

# Lecture 10 — Attention and the Transformer architecture

## Learning goals

1. Understand attention as a weighted aggregation of information from different
   sequence positions — direct access, without pushing information through a chain
   of hidden states.
2. Be able to compute scaled dot-product attention by hand and explain the roles of
   $Q$, $K$, $V$.
3. Understand multi-head attention as several independent projections operating in
   parallel, not as one bigger mechanism.
4. Correctly apply masking (padding and causal) — including understanding that
   different PyTorch APIs interpret boolean masks **differently**, not
   interchangeably.
5. Understand why self-attention without positional information cannot distinguish
   token order, and how positional encoding fixes this.
6. Know the structure of a Transformer block (attention + FFN + residual + LayerNorm)
   and distinguish: a single head, a multi-head layer, a block, the whole model.

## Why this matters

In module 9, the hidden state $h_t$ had to compress the entire sequence seen so far
into a fixed-size vector — information from the distant past only reached the
current step after passing through many intermediate updates (and, as we measured,
could vanish along the way). Attention inverts this principle: instead of relying
solely on a compressed state, every position can **directly** "look up" the
representation of any other position, weighting it by how relevant it is. We will see
this literally in the comparative experiment with a GRU: on a task where two relevant
tokens are $64$ positions apart, the GRU loses most of its accuracy (dropping close
to chance level), while a small Transformer does noticeably better (though not
perfectly either).

## Intuition

Imagine that for every position in the sequence you ask: "which other positions
matter to me, and how much information should I take from them?". Attention
formalizes this question: every position emits a **query**, every position
(including itself) provides a **key** ("what I am about") and a **value** ("what
information I carry"). The similarity between a query and a key determines the
weight with which the corresponding value enters the result.

❌ "Attention and the Transformer are the same thing."

✅ Attention is a **mechanism** (a way of weighting and aggregating information) —
the Transformer is an **architecture** built from this mechanism plus a feed-forward
network, residual connections, and normalization, arranged into repeatable blocks.
The attention mechanism itself existed and was used (e.g. in RNN encoder-decoder
models) before the Transformer was invented — we will see below that the Transformer
is a specific way of combining attention with other, equally important components.

## Formal model: Query, Key, Value

For a sequence of input representations $x_1,\ldots,x_T \in \mathbb R^{d_{\text{model}}}$
(e.g. token embeddings from module 8, possibly plus positional encoding — below), we
build three matrices via linear projections:

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

where $X\in\mathbb R^{T\times d_{\text{model}}}$ (rows = positions), $W^Q, W^K \in
\mathbb R^{d_{\text{model}}\times d_k}$, $W^V\in\mathbb R^{d_{\text{model}}\times d_v}$
(in the basic version $d_k=d_v=d_{\text{model}}$; smaller in multi-head — below).

**A note on notation.** $Q$, $K$, $V$ is **standardized notation across the entire
literature** (query/key/value) — we keep it, even though $K$ in this course also
means "number of layers" (module 2) and "kernel size" (module 6). This is **a
different entity**: $K$ here is always a **matrix** (appearing together with $Q$ and
$V$), never a single number — context unambiguously distinguishes them, just as in
module 8 we distinguished two meanings of $k$ (k-NN vs. kernel size). We denote
attention weights $\alpha$ (unused earlier in the course) — **not** $A$, which in
module 3 denotes a layer's batched activations ($A^{(l)}$).

## Mathematics: scaled dot-product attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Let's break this down step by step:

1. **$QK^\top$** $\in\mathbb R^{T\times T}$ — the dot product of every query with
   every key: row $i$, column $j$ is $q_i\cdot k_j$, a measure of "how well query $i$
   matches key $j$" (a larger dot product = more aligned directions, for comparable
   norms — exactly the intuition from module 8, cosine similarity, just without
   explicit L2 normalization here).
2. **Scaling by $\sqrt{d_k}$**: for random $q,k$ with independent coordinates, the
   variance of $q\cdot k$ grows **linearly** with $d_k$ (a sum of $d_k$ independent
   products) — without scaling, for large $d_k$, the values of $QK^\top$ have high
   variance, which pushes softmax into a region of very small gradients (saturation,
   module 2). Dividing by $\sqrt{d_k}$ brings the variance back to order $1$,
   regardless of $d_k$.
3. **Softmax along the last dimension** (over columns, separately for each
   row/query) — turns every row into a probability distribution over the available
   keys: $\alpha_{ij}\geq 0$, $\sum_j \alpha_{ij}=1$. These are exactly the
   **attention weights**: how much attention query $i$ pays to key (and hence value)
   $j$.
4. **$\cdot V$**: every output position is a weighted (weights $\alpha_{i\cdot}$) sum
   of all values — aggregation of information from the whole sequence, in one step,
   without passing through intermediate states.

**Dimensions — spelled out.** For a single sequence: $Q,K\in\mathbb R^{T\times d_k}$,
$V\in\mathbb R^{T\times d_v}$, $QK^\top\in\mathbb R^{T\times T}$, output
$\in\mathbb R^{T\times d_v}$. For a batch of size $B$: we add a leading dimension,
$Q,K\in\mathbb R^{B\times T\times d_k}$ etc., $QK^\top\in\mathbb R^{B\times T\times T}$
— the attention-weight matrix is **separate for every example in the batch**.

**A note on notation.** We denote batch size here as $B$, not $N$ as in modules
1-3/6-7 — deliberately, because in this module $N$ is already taken (below: the
number of Transformer blocks stacked one after another). $B$ is the only symbol here
for "number of examples at once", with no collision within this document.

## Worked example: 3 positions, $d_k=4$

$$Q=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&0&0\end{pmatrix}\quad K=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&1&1\end{pmatrix}\quad V=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$$

**1. $QK^\top$:**
$$QK^\top=\begin{pmatrix}2&0&2\\0&2&2\\1&1&2\end{pmatrix}$$

**2. After scaling ($\div\sqrt4=\div2$):**
$$\begin{pmatrix}1.0&0.0&1.0\\0.0&1.0&1.0\\0.5&0.5&1.0\end{pmatrix}$$

**3. Weights after softmax (every row sums to $1$):**
$$\begin{pmatrix}0.4223&0.1554&0.4223\\0.1554&0.4223&0.4223\\0.2741&0.2741&0.4519\end{pmatrix}$$

**4-5. Weighted combination of $V$ = final representation:**
$$\begin{pmatrix}3.000&4.000\\3.534&4.534\\3.356&4.356\end{pmatrix}$$

Verified within numerical tolerance (`atol=1e-6`) against
`F.scaled_dot_product_attention` in the notebook.

## Multi-head attention

Instead of a single $Q,K,V$ projection at full dimension $d_{\text{model}}$,
multi-head attention splits into $H$ **independent** heads, each of smaller
dimension $d_k = d_{\text{model}}/H$. **A note on notation:** $H$ here is the number
of attention heads — **a different meaning** than $H$ in module 6 (image height in
the NCHW convention). The same symbol, a completely different domain (sequences vs.
images) — context disambiguates it, just like $k$ in module 6 (kernel size) and $k$
in module 8 (k-NN):

$$\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i), \qquad i=1,\ldots,H$$
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)\,W^O$$

Every head has **its own** $W^Q_i, W^K_i, W^V_i$ (different matrices, not the same
projection replicated $H$ times) — the results are concatenated along the feature
dimension and passed through one shared output projection $W^O\in\mathbb
R^{d_{\text{model}}\times d_{\text{model}}}$. In practice (and in
`nn.MultiheadAttention`), the $Q,K,V$ projections for all heads are computed with a
**single**, **concatenated** matrix $W^{QKV}\in\mathbb R^{d_{\text{model}}\times 3d_{\text{model}}}$,
and only then is the result split into $H$ heads — mathematically
equivalent to $H$ separate projections, verified within numerical tolerance
(`atol=1e-5`) in the notebook by manually slicing `in_proj_weight`.

❌ "Each head learns a different, recognizable relation (e.g. a 'syntactic head', a
'semantic head')."

✅ Different heads **can** specialize in different patterns — this is a real,
documented observation in large, trained models — but it is **not a guarantee** of
the architecture, only a possible (and ambiguous to interpret) effect of training.
Randomly initialized heads have no pre-determined specialization at all — in
Experiment B we will see that the shapes and independent weight matrices alone do not
settle anything about content.

## Masking

**Padding mask**: queries should not "look at" padding positions (module 9: padding
is not content). **Causal mask**: in autoregressive models (generating text token by
token), position $i$ must not use position $j>i$ — otherwise the model would "see the
future" during training, which it does not have during actual generation.

❌ "The causal mask also handles padding, or vice versa — they're basically the same
thing."

✅ These are **two independent** constraints that **combine** (logical AND), not
substitute for each other: the causal mask restricts based on **position relative to
the query** (always the same triangular pattern, independent of the specific
sequence), the padding mask restricts based on **the content of a given example**
(which positions are real depends on the actual length of a specific sequence in the
batch). A decoder model with variable sequence length in a batch needs **both at
once** — verified in the notebook: the combined mask is the (logical AND) product of
the two masks taken separately.

### Padding in an RNN vs. padding in attention — not the same mechanism

Module 9 teaches `pack_padded_sequence` as a way to ignore padding in an
RNN/LSTM/GRU — it is tempting to conclude from this that the padding mask in
attention "does the same thing". **It does not.** These are two different
mechanisms, not two names for one thing:

| | `pack_padded_sequence` (RNN, module 9) | `key_padding_mask` (attention) |
|---|---|---|
| What happens to a padding position | **Is not processed at all** — the recurrent layer physically skips this step | **Is processed normally** by the embedding, FFN, LayerNorm — only its **weight in the attention sum** is zeroed out |
| A padding position as a query | Does not exist (the sequence has already ended) | **Still produces some output** — it is not skipped as a computational step, only ignored afterward |

Consequence: masking attention alone **is not enough** to make padding fully
without effect on the result — if the representation of the whole sequence is built
by averaging *all* positions (not only the real ones), padding will affect the
result even when attention correctly ignored it when weighting. **Also** needed is
correct aggregation at the end — see the diagnostic task in this module's lab.

### Boolean mask vs. additive mask

A mask can be encoded two ways: as a **boolean** (where one value means "allowed",
the other "blocked" — but **which is which depends on the API!**, see below) or as an
**additive** mask (a tensor of real numbers added to the $QK^\top$ result before
softmax: $0$ for allowed positions, $-\infty$ for blocked ones — after adding
$-\infty$, `softmax` assigns that position a weight of $0$).

### Rows with all keys masked

If, for some query, **no** key is allowed (e.g. a bug in mask construction), the row
of $QK^\top$ after masking is all $-\infty$ — `softmax` computes $e^{-\infty}/\sum e^{-\infty} = 0/0$.
**Verified in the notebook:** a naive, manual implementation
(`masked_fill(-\infty)` + `softmax`) gives `NaN` in this situation. The optimized
`F.scaled_dot_product_attention` kernel has a special case and returns $0$ instead of
`NaN` — **but do not rely on this in your own "from scratch" implementation**: make
sure every query position whose result you actually use has at least one allowed key
(e.g. queries corresponding to real, non-padding tokens always have at least
themselves available under a causal mask).

**An additional pitfall, specific to `nn.MultiheadAttention`** (verified
empirically): whether a fully masked row gives `NaN` or a safe zero depends on the
`need_weights` parameter. **The default value is `need_weights=True`** (needed when
we want to **inspect** the attention weights themselves — as in the notebook and the
lab) — this forces the explicit path with manual `softmax`, and in that case a fully
masked row gives `NaN`, exactly like the naive "from scratch" implementation above.
You must **explicitly set** `need_weights=False` so that `nn.MultiheadAttention` can
use the same optimized path as `F.scaled_dot_product_attention` — a safe zero.
Conclusion: even within **one** API, numerical behavior can depend on an argument
unrelated to masking itself — another reason not to assume behavior without testing
it.

### An important implementation note: masks are NOT portable between APIs

**Verified experimentally, not from documentation** (direct inspection of attention
weights on a controlled example):

| API | parameter | `True` means |
|---|---|---|
| `F.scaled_dot_product_attention` | `attn_mask` (bool) | **allowed** (attend) |
| `nn.MultiheadAttention` | `attn_mask` (bool) | **blocked** (ignore) |
| `nn.MultiheadAttention` | `key_padding_mask` (bool) | **this is padding** (ignore) |

`nn.MultiheadAttention` is **internally consistent** (`attn_mask` and
`key_padding_mask` both: `True`=block), but **opposite** to
`F.scaled_dot_product_attention`. Copying a boolean mask from one API to the other
without inverting it gives a **silently wrong result** — no exception, the model will
still train, but it will "see" exactly the positions it was supposed to ignore (and
vice versa). A verified test in the notebook checks this directly for both APIs,
rather than trusting the documentation alone.

## Position in the sequence: why self-attention needs it

The formula $\text{Attention}(Q,K,V)$ **never refers anywhere to the position index**
$i$ or $j$, beyond which specific rows of $X$ it takes — if we permute the rows of
input $X$, the output permutes in **exactly the same way** (a property called
**permutation equivariance**). Verified in the notebook: `output(permute(X))` matches
`permute(output(X))` within numerical tolerance (`atol=1e-6`).

Consequence: if we build a model that classifies a sequence via self-attention +
**order-insensitive** aggregation (e.g. an average over positions), the whole thing
is **permutation invariant** — permuting the input **does not change** the
prediction. Verified: for the task from module 9 (the label depends solely on the
order of two tokens), a self-attention + mean-pooling model **without** positional
encoding reaches exactly chance level — an identical result on each of the 3 tested
seeds (so this is not a matter of "unlucky" initialization — the model
mathematically **cannot** represent a function that depends on order).

❌ "Self-attention automatically 'knows' which position is which, because it
processes the sequence."

✅ The attention mechanism itself is blind to position — it "knows" only as much as
it **explicitly** receives in the input representation. Hence **positional
encoding**: we add to each token's embedding an extra vector that depends on its
position, **before** it enters attention.

**Positional encoding — sinusoidal version** (the original Transformer):

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Every pair of coordinates encodes position with a sinusoidal wave of a different
frequency — deterministic, with no additional parameters to train, and it works (to
a limited extent) for sequences longer than those seen during training. Alternative:
a **learned positional embedding** (`nn.Embedding(max_len, d_model)`, exactly like
the token embedding from module 8, just indexed by position instead of token
identity) — simpler, but it does not extrapolate beyond the `max_len` seen during
training.

❌ "Either of these two position-encoding methods works equally well in every
situation."

✅ Sinusoidal encoding is deterministic and (partially) generalizes to longer
sequences; learned positional embeddings can fit a specific task better, but only
within the range of lengths seen during training — the choice is a trade-off, not an
obvious default.

## Transformer block architecture

$$x' = \text{LayerNorm}\big(x + \text{MultiHead}(x)\big), \qquad \text{out} = \text{LayerNorm}\big(x' + \text{FFN}(x')\big)$$

(the **Post-LN** variant, as in the original Transformer — normalization **after**
adding the residual connection). The FFN is a simple network **applied independently
to every position** (*position-wise*): $\text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$,
usually with an inner dimension $4\times$ larger than $d_{\text{model}}$.

- **Residual connections** ($x + \text{sublayer}(x)$): exactly the same mechanism as
  in ResNet (module 7) — they ease gradient propagation through deep stacks of
  blocks, for the same reasons (the gradient has a "shortcut" bypassing the
  sublayer).
- **LayerNorm**: normalizes every position **independently**, over the feature
  dimension (not over the batch, like BatchNorm from modules 6-7) — verified in the
  notebook: mean $\approx0$, standard deviation $\approx1$ for every position
  separately.
- **Dropout**: typically applied after attention and after the FFN, before adding to
  the residual connection — as in module 5, but watch for the pitfall below.

**Pre-LN** (the variant used in many newer models): normalization **before** the
sublayer, not after: $x' = x + \text{MultiHead}(\text{LayerNorm}(x))$. The difference
affects training stability for very deep stacks — we do not develop a full analysis
here, only note that this is a real, deliberate design choice, not an insignificant
detail.

**Four different levels — do not confuse them:**

| Level | What it is |
|---|---|
| A single head | one $\text{Attention}(XW^Q_i,XW^K_i,XW^V_i)$ |
| A multi-head layer | $H$ heads + concatenation + projection $W^O$ |
| A Transformer block | multi-head + residual + LN + FFN + residual + LN |
| A model | $N$ blocks stacked one after another (plus an embedding + head at input/output) |

❌ "Building a Transformer is basically one call to `nn.Transformer`."

✅ `nn.Transformer`/`nn.TransformerEncoder` **hide** all of the above layers behind a
single call — convenient in practice, but not useful pedagogically: in this module we
build the block explicitly, layer by layer, so as to see what happens inside.

## Encoder-only, decoder-only, encoder-decoder

The original Transformer (Vaswani et al., 2017) is an **encoder-decoder**
architecture: the encoder processes the entire input sequence at once (self-attention
with no causal mask — every position sees the rest), the decoder generates the
output sequence token by token (self-attention **with** a causal mask, plus
**cross-attention** to the encoder's output — $Q$ from the decoder, $K,V$ from the
encoder).

- **Cross-attention vs. self-attention**: self-attention computes $Q,K,V$ from the
  **same** sequence; cross-attention computes $Q$ from one sequence (e.g. the
  decoder), and $K,V$ from **another** (e.g. the encoder) — the mathematical
  mechanism is identical, only **where** the inputs come from differs.
- **Encoder-only** (e.g. BERT-type architectures): encoder only, self-attention with
  no causal mask — good for tasks where the whole sequence is available at once
  (classification, as in our lab).
- **Decoder-only** (e.g. GPT-type architectures): decoder only, self-attention with a
  causal mask, no cross-attention (nothing to refer to) — good for generating text
  token by token.

❌ "Encoder-only and decoder-only are basically the same architecture under a
different name."

✅ They differ in **whether and how** they mask self-attention — encoder-only does
not mask (bidirectional access to the whole sequence), decoder-only masks causally
(unidirectional, consistent with generation). This is a real functional difference,
not just a naming one — an encoder-only model is not directly suited to generating
text token by token, because during training it "saw" the whole sequence at once.

## What we observe

All numbers below come from actually running `examples.ipynb` (not from estimates) —
see the notebook for the full output.

- **Experiment A.** The manual implementation of scaled dot-product attention agrees,
  within numerical tolerance (`atol=1e-6`), with `F.scaled_dot_product_attention` on
  the 3-position numeric example. Changing one query changes **only** the
  corresponding row of the result; changing one key/value changes **all** rows (every
  query can "look at" it). Verified directly: `nn.MultiheadAttention` interprets the
  boolean mask (`attn_mask`, `key_padding_mask`) **oppositely** to
  `F.scaled_dot_product_attention` — the same logical intent written with the wrong
  convention gives a reversed, silent result. A row with all keys masked: the naive
  implementation gives `NaN`, `F.scaled_dot_product_attention` gives zero.
  `model.eval()` does not disable dropout passed as a constant function argument
  (confirmed: two calls to `NaiveAttention.eval()` give different results).
- **Experiment B.** The manual construction of multi-head attention (slicing
  `in_proj_weight`, splitting into heads, concatenation, `out_proj`) agrees, within
  numerical tolerance (`atol=1e-5`), with `nn.MultiheadAttention`, both for the output
  and the per-head weights. Parameter count for $d_{\text{model}}=8$, $H=2$: $288$,
  matching the formula
  $3d_{\text{model}}^2+3d_{\text{model}}+d_{\text{model}}^2+d_{\text{model}}$.
- **Experiment C.** Permutation equivariance of self-attention and invariance of the
  mean-pooled representation (as well as invariance of a CLS-style representation to
  permutation of the *remaining* tokens) — confirmed within numerical tolerance
  (`atol=1e-6`). On an order-dependent task (an 8-element vocabulary, label = order of
  tokens 0 and 1): **with** positional encoding, `val_acc=test_acc=1.000`; **without**
  PE, `val_acc=0.475`, `test_acc=0.485` — **identical on each of the 3 tested seeds**
  (0, 1, 2), confirming that this is a representational limitation, not unlucky
  initialization. Permuting the input **with** PE changes the prediction (logits
  $[5.84,-4.49]\to[6.01,-4.66]$ after permutation in the checked example) — as
  expected, since the order actually changes.
- **Experiment D.** A small Transformer (2514 parameters at $T=8$) solves the order
  task with no error (`val_acc=1.000`, 0/200 errors). Comparison with a GRU (1922
  parameters) on a harder variant of the task (distractors appended after a fixed,
  early decisive pair of tokens — forcing memory across the whole "tail" length):
  $T=8,16,32$ — both models `val_acc=1.000`; **$T=64$ — Transformer $0.860$, GRU
  $0.480$** (close to chance level) — an **observed accuracy difference**, measured
  from a single run (unlike the with/without-PE comparison above in the same
  experiment, this specific comparison is not checked across multiple seeds). Module
  9 (Experiment B) **independently measured**, on a different task, that the GRU's
  gradient norm with respect to a distant input vanishes exponentially with sequence
  length. The GRU's accuracy drop at $T=64$ is **consistent with the hypothesis**
  that the same mechanism contributed to this result — the GRU must carry
  information about the positions of tokens 0/1 through $56$ steps of pure noise, the
  Transformer has direct access to them regardless of distance — but in **this**
  experiment we did not measure the GRU's gradient norm on this specific task: this
  is a plausible, but not directly confirmed, hypothesis, not a proven cause. The
  Transformer **also** degrades at $T=64$ (it does not "always win") — under the
  same, unchanged training budget. Training time: the Transformer is faster than the
  GRU at larger $T$ (e.g. $1.22$s vs. $2.41$s at $T=64$), but this is a
  microbenchmark on very short sequences — we do not draw far-reaching conclusions
  about real-world performance from it.

## Common misconceptions

(Additional ones, not covered above)

❌ "A larger attention weight is proof that a given token causally influenced the
prediction."

✅ An attention weight says how much a given value **contributed** to the weighted
sum at that step — this is a **descriptive** property of one computational step, not
proof of causality with respect to the model's final prediction (which still passes
through the FFN, further blocks, the output head). A high weight does not guarantee
that removing that token would change the result — conversely, a low weight does not
guarantee no influence (the information could have arrived another way, e.g. through
an earlier block).

❌ "Attention eliminates all problems related to long sequences."

✅ Attention gives **direct** access to every position (instead of through a chain of
states), which genuinely helps with long-range dependencies (see the comparison with
a GRU below) — but memory and compute cost grow **quadratically** with sequence
length (a $T\times T$ weight matrix), so very long sequences are expensive for a
completely different reason than vanishing gradients.

❌ "A small Transformer will always solve a sequential task better than an
RNN/LSTM/GRU."

✅ Measured directly (comparative experiment, "What we observe" section above): on
short sequences ($T=8$) a GRU does equally well; at $T=64$, under the same, unchanged
training budget, **our small Transformer is also not perfect** (not 100%) — model
size and training budget matter, the architecture alone does not guarantee success.

❌ "The mere presence of an attention layer means the model correctly understands the
sequence's content."

✅ Attention is a computational mechanism — whether a trained model actually
"understands" content in any meaningful sense is a separate question, not settled by
the architecture alone. We distinguish: the mathematical operation of the mechanism
(what exactly it computes), the architecture's capacity (what it **could**
represent), and a specific trained model's behavior (what it **actually** learned) —
three different levels, none automatically implying the others.

## Engineering implications

- **The $T\times T$ cost.** The attention-weight matrix has size $T\times T$ (per
  head, per example in the batch) — the number of position comparisons grows
  **quadratically** with sequence length, unlike an RNN, where the cost grows
  **linearly**. For short sequences (as in our experiments) this is irrelevant; for
  very long ones (thousands of tokens) it becomes the dominant memory and time cost.
- **Memory.** A simple implementation (like ours, in the notebook) **materializes**
  the entire $T\times T\times H$ matrix (for $H$ heads) in memory — optimized kernels
  (e.g. FlashAttention, mentioned only, not implemented) avoid explicitly
  materializing the full matrix, giving a **different** memory-cost profile than our
  teaching version, despite an identical mathematical result.
- **Parallelism vs. actual wall-clock time.** Self-attention can process **all**
  sequence positions in parallel during training (there is no dependency of $h_t$ on
  $h_{t-1}$, as in an RNN) — but this **does not automatically mean** lower execution
  time in every setting: for short sequences, small models, and on CPU (where
  "parallelism" is limited by core count anyway), the difference can be small or even
  reversed, measured directly in the comparative experiment.
- **Number of heads and representation dimension.** More heads at the same
  $d_{\text{model}}$ means a **smaller** $d_k$ per head (since $d_k=d_{\text{model}}/H$)
  — not more parameters, just a different split of the same pool.
- **Training vs. inference.** During training, an encoder/self-attention with no
  causal mask processes the whole sequence at once. A decoder-only model, during
  **generation** (inference), produces one token at a time — in practice it caches
  the already-computed $K,V$ for earlier positions (*KV cache*), so as not to
  recompute them for every new token — a topic developed further in module 11.

## What to remember

- $\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$ — a weighted
  aggregation of values, weights from query-key similarity, scaling prevents softmax
  saturation.
- Multi-head = $H$ independent projections in parallel, not a guarantee of
  interpretable specialization for each head.
- Boolean masks are **not portable** between `F.scaled_dot_product_attention` and
  `nn.MultiheadAttention` — check the semantics every time.
- Self-attention without positional encoding is permutation equivariant — it does not
  distinguish token order without explicitly added positional information.
- A Transformer block = multi-head attention + FFN + two residual connections + two
  normalizations — four distinct levels (head/layer/block/model), not one entity.
- A Transformer does not have to beat an RNN/LSTM/GRU on every task, especially a
  small, short one — the advantage shows up for long-range dependencies, measured.

## Questions for students

1. You have a query $q$ identical to one of the keys $k_j$ and very different from
   the rest. What will the attention-weight row for this query (approximately) look
   like after softmax?
2. Why do we divide by $\sqrt{d_k}$, rather than, say, $d_k$ or not at all? What would
   happen to the attention weights at very large $d_k$ without this scaling?
3. You have an encoder-only model (no causal mask) trained to classify a sequence by
   averaging attention outputs over positions. Will permuting the order of the input
   tokens change the prediction? Justify your answer using permutation equivariance.
4. You have a batch with sequences of different lengths and want to build an
   autoregressive (decoder-only) model. Which **two** masks do you need at the same
   time, and how do they combine?

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in
  Neural Information Processing Systems 30 (NeurIPS 2017). arXiv:1706.03762.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*.
  arXiv:1607.06450.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image
  Recognition*. CVPR. (residual connections — see also module 7)
- PyTorch documentation: *torch.nn.functional.scaled_dot_product_attention*,
  `https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html`.
- PyTorch documentation: *torch.nn.MultiheadAttention*,
  `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
