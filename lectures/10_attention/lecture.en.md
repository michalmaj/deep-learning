Language: **English** | [Polski](lecture.md)

# Lecture 10. Attention and the Transformer architecture

## Learning goals

1. describe attention as weighted aggregation across sequence positions,
2. calculate scaled dot-product attention and explain the roles of $Q$, $K$, and $V$,
3. distinguish a single head, multi-head attention, a Transformer block, and a complete model,
4. apply padding and causal masks correctly,
5. explain why position information is required,
6. present attention, FFN, residual connections, and LayerNorm as parts of a Transformer block.

## Why this matters

In module 9, the state $h_t$ compressed the sequence history into a fixed-size vector. Information from a distant position had to cross many updates and could vanish along the way.

Attention gives every position direct access to representations at other positions. Their contribution depends on learned weights. One experiment compares this mechanism with GRU on a task where a long sequence of distractors separates the relevant information from the output.

## Intuition

Every position asks which elements of the sequence matter. A query describes the information being sought. A key allows the model to assess how well a position matches the query. A value contains the information passed to the result.

The similarity between a query and a key determines the weight of the corresponding value.

❌ Attention and Transformer mean the same thing.

✅ Attention is an aggregation mechanism. Transformer is an architecture that combines attention with an FFN, residual connections, and normalization.

## Query, Key, and Value

For $X\in\mathbb R^{T\times d_{\text{model}}}$, whose rows correspond to positions, three linear projections are created.

$$Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V$$

The matrices $W^Q,W^K\in\mathbb R^{d_{\text{model}}\times d_k}$ and $W^V\in\mathbb R^{d_{\text{model}}\times d_v}$. Inputs may be token embeddings from module 8 with positional encoding added.

$Q$, $K$, and $V$ are the standard symbols for query, key, and value. Here, $K$ is a matrix rather than the number of layers or kernel size. Attention weights are denoted by $\alpha$ to avoid a collision with the activations $A^{(l)}$ from module 3.

## Scaled dot-product attention

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The calculation has four stages.

1. $QK^\top$ contains dot products between every query and key. Entry $(i,j)$ measures the match between $q_i$ and $k_j$.
2. Division by $\sqrt{d_k}$ stabilizes the variance of the scores. Without scaling, larger $d_k$ encourages softmax saturation and small gradients.
3. Softmax is applied independently to each row. The resulting $\alpha_{ij}$ are nonnegative and satisfy $\sum_j\alpha_{ij}=1$.
4. Multiplication by $V$ forms a weighted sum of values for each query.

For one sequence, $Q,K\in\mathbb R^{T\times d_k}$, $V\in\mathbb R^{T\times d_v}$, $QK^\top\in\mathbb R^{T\times T}$, and the output belongs to $\mathbb R^{T\times d_v}$. For a batch of size $B$, the leading dimension has size $B$, and every example has its own $T\times T$ weight matrix.

The symbol $B$ denotes batch size. $N$ is reserved for the number of Transformer blocks.

## Three-position example

For $d_k=4$, let

$$Q=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&0&0\end{pmatrix},\quad K=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&1&1\end{pmatrix},\quad V=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}.$$

The query-key products are

$$QK^\top=\begin{pmatrix}2&0&2\\0&2&2\\1&1&2\end{pmatrix}.$$

Division by $\sqrt4=2$ gives

$$\begin{pmatrix}1.0&0.0&1.0\\0.0&1.0&1.0\\0.5&0.5&1.0\end{pmatrix}.$$

Softmax produces the weights

$$\begin{pmatrix}0.4223&0.1554&0.4223\\0.1554&0.4223&0.4223\\0.2741&0.2741&0.4519\end{pmatrix}.$$

The final weighted combination of $V$ is

$$\begin{pmatrix}3.000&4.000\\3.534&4.534\\3.356&4.356\end{pmatrix}.$$

The notebook confirms agreement with `F.scaled_dot_product_attention` at `atol=1e-6`.

## Multi-head attention

Multi-head attention uses $H$ independent heads. With $d_k=d_{\text{model}}/H$, each head operates in a smaller space.

$$\operatorname{head}_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V),\qquad i=1,\ldots,H$$

$$\operatorname{MultiHead}(X)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O$$

Every head has its own projections. Their results are concatenated and passed through a shared matrix $W^O\in\mathbb R^{d_{\text{model}}\times d_{\text{model}}}$. Here, $H$ denotes the number of heads rather than image height from module 6.

`nn.MultiheadAttention` combines the projections for all heads in one matrix $W^{QKV}\in\mathbb R^{d_{\text{model}}\times3d_{\text{model}}}$. This is equivalent to separate projections. Manually splitting `in_proj_weight` matches the PyTorch implementation at `atol=1e-5`.

❌ Every head must learn a separate relationship with an obvious interpretation.

✅ Specialization may emerge during training, but the architecture does not guarantee it. Random heads have no predefined roles.

## Masking

A padding mask blocks access to filler positions. A causal mask blocks positions $j>i$ so that an autoregressive model cannot use future tokens.

❌ One mask can replace the other.

✅ They impose independent constraints. A causal mask depends on relative position, while a padding mask depends on the real length of a particular sequence. A decoder processing variable-length batches needs both.

### Padding in RNN and attention

`pack_padded_sequence` from module 9 and `key_padding_mask` serve a similar purpose but operate differently.

| Mechanism | Treatment of a padding position | Padding as a query |
|---|---|---|
| `pack_padded_sequence` | the RNN does not process the step | it does not exist after the sequence ends |
| `key_padding_mask` | the position still passes through the embedding, FFN, and LayerNorm; only its attention weight is blocked | it still produces an output |

An attention mask alone does not remove every effect of padding. The final aggregation must also exclude filler positions. Otherwise, a mean representation still depends on the number of `PAD` positions.

### Boolean and additive masks

A Boolean mask uses two logical values, but the meaning of `True` depends on the API. An additive mask is added to $QK^\top$ before softmax. Allowed positions receive $0$ and blocked positions receive $-\infty$, giving them a softmax weight of $0$.

### All keys masked

If every key is blocked for a query, naive softmax receives only $-\infty$ values and calculates $0/0$. The manual notebook implementation returns `NaN`. `F.scaled_dot_product_attention` handles this case and returns zero.

In `nn.MultiheadAttention`, the result also depends on `need_weights`. The default `need_weights=True` follows an explicit softmax path and may produce `NaN`. Setting `need_weights=False` allows an optimized path that returns zero. A custom implementation should ensure at least one allowed key for every query whose output is used.

### Mask semantics in PyTorch APIs

Boolean masks are not portable between interfaces.

| API | Parameter | Meaning of `True` |
|---|---|---|
| `F.scaled_dot_product_attention` | `attn_mask` | the position is allowed |
| `nn.MultiheadAttention` | `attn_mask` | the position is blocked |
| `nn.MultiheadAttention` | `key_padding_mask` | the position is padding and blocked |

`nn.MultiheadAttention` is internally consistent but uses the opposite convention from `F.scaled_dot_product_attention`. Moving a mask incorrectly may not raise an exception. It can silently invert the set of visible positions.

## Position information

The attention equation does not directly use the indices $i$ and $j$. Permuting the rows of $X$ causes the same permutation of the output. This property is called permutation equivariance. The notebook confirms it at `atol=1e-6`.

Combining self-attention with an order-insensitive aggregation such as a mean is permutation invariant. Such a model cannot solve a task whose label depends only on token order.

❌ Self-attention automatically knows the positions of sequence elements.

✅ Position information must be added to the input. Positional encoding augments a token embedding before the attention layer.

The sinusoidal encoding from the original Transformer is

$$PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

Each coordinate pair uses a different frequency. The encoding is deterministic, adds no parameters, and can partly support lengths unseen during training.

A learned position embedding such as `nn.Embedding(max_len, d_model)` is an alternative. It may adapt more closely to the task but cannot handle indices beyond `max_len`.

## Transformer block

The Post-LN variant used in the original Transformer is

$$x'=\operatorname{LayerNorm}\big(x+\operatorname{MultiHead}(x)\big),$$

$$\operatorname{out}=\operatorname{LayerNorm}\big(x'+\operatorname{FFN}(x')\big).$$

The FFN operates independently at every position.

$$\operatorname{FFN}(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2$$

Its inner dimension is commonly about four times larger than $d_{\text{model}}$.

- Residual connections provide a shorter gradient path, as in ResNet from module 7.
- LayerNorm normalizes every position independently over features rather than over the batch.
- Dropout is usually applied after attention and FFN, before the residual addition.

In the Pre-LN variant, normalization precedes the sublayer, for example $x'=x+\operatorname{MultiHead}(\operatorname{LayerNorm}(x))$. This choice affects the stability of deep models.

| Level | Contents |
|---|---|
| single head | one $\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V)$ |
| multi-head layer | $H$ heads, concatenation, and $W^O$ |
| Transformer block | multi-head, FFN, two residual connections, and two normalizations |
| model | $N$ blocks, embeddings, and an input or output head |

High-level PyTorch classes hide these elements. This module builds the block explicitly to expose the data flow.

## Encoder, decoder, and cross-attention

The original Transformer is an encoder-decoder architecture. The encoder processes the complete input without a causal mask. The decoder generates successive tokens with a causal mask and uses cross-attention over encoder outputs.

Self-attention derives $Q$, $K$, and $V$ from one sequence. In cross-attention, queries come from one sequence while keys and values come from another.

- **Encoder-only** uses bidirectional self-attention and fits tasks such as classification.
- **Decoder-only** uses causal self-attention and fits token-by-token generation.
- **Encoder-decoder** connects both components through cross-attention.

❌ Encoder-only and decoder-only differ only in name.

✅ A causal mask changes access to information. An encoder sees the full sequence, while a decoder sees only the current and preceding positions.

## Experiments and observations

The results come from the executed [`examples.en.ipynb`](examples.en.ipynb).

### A. Scaled dot-product attention and masks

The manual implementation agrees with `F.scaled_dot_product_attention` at `atol=1e-6`. Changing one query affects only its corresponding row. Changing a key or value may affect every query.

The notebook confirms the opposite meanings of Boolean masks in the two APIs. It also shows `NaN` from a naive implementation for a fully masked row and zero from the optimized function.

Dropout passed as a constant function argument is not disabled by `model.eval()`. Two calls to `NaiveAttention.eval()` may therefore produce different outputs.

### B. Multi-head attention

Manual projections, head splitting, concatenation, and `out_proj` agree with `nn.MultiheadAttention` at `atol=1e-5` for both the output and per-head weights.

For $d_{\text{model}}=8$ and $H=2$, the model has 288 parameters. This matches $3d_{\text{model}}^2+3d_{\text{model}}+d_{\text{model}}^2+d_{\text{model}}$.

### C. Order and positional encoding

Self-attention is permutation equivariant, and a mean representation is permutation invariant. A CLS-style representation is also invariant when the remaining tokens are permuted. Numerical agreement is checked at `atol=1e-6`.

In an eight-token task, the label depends on the order of tokens 0 and 1. The model with positional encoding reaches `val_acc=test_acc=1.000`. Without PE, it reaches `val_acc=0.475` and `test_acc=0.485`. The result repeats for seeds 0, 1, and 2.

After adding PE, an input permutation changes the prediction. In the checked example, logits change from $[5.84,-4.49]$ to $[6.01,-4.66]$.

### D. Transformer and GRU

The small Transformer has 2514 parameters at $T=8$ and reaches `val_acc=1.000`, with no errors among 200 examples.

In a harder task variant, the decisive tokens occur early and are followed by distractors. The GRU has 1922 parameters.

| $T$ | Transformer | GRU |
|---:|---:|---:|
| 8 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 |
| 32 | 1.000 | 1.000 |
| 64 | 0.860 | 0.480 |

The $T=64$ result comes from one run. Accuracy was measured, not gradient norm. The GRU decline is consistent with a hypothesis based on the independent experiment from module 9, but it does not prove that vanishing gradients were the only cause. A fixed 30-epoch budget and a single seed remain confounding factors.

The Transformer also becomes less accurate at $T=64$. Training takes about 1.22 s for the Transformer and 2.41 s for the GRU, but this is a microbenchmark on short sequences rather than a general performance comparison.

## Common misconceptions

❌ A large attention weight proves that a token caused the prediction.

✅ A weight describes the value's contribution to one weighted sum. The prediction still passes through FFNs, later blocks, and a head. Causality requires a separate experiment.

❌ Attention removes every problem associated with long sequences.

✅ It shortens the information path, but the $T\times T$ matrix creates quadratic memory and compute costs.

❌ A small Transformer always outperforms RNN, LSTM, and GRU.

✅ At $T=8$, $16$, and $32$, both models reach the same result. Architecture alone does not guarantee an advantage for every task and budget.

❌ The presence of an attention layer proves that content is understood.

✅ A computation mechanism, the capacity of an architecture, and the behavior of a trained model are separate levels of description.

## Engineering implications

- **Cost.** The number of position comparisons grows quadratically with $T$, while recurrent cost grows linearly.
- **Memory.** A simple implementation materializes a $T\times T\times H$ matrix. Optimized kernels such as FlashAttention can avoid storing the complete matrix without changing the mathematical result.
- **Parallelism.** Positions can be processed in parallel during training, but this does not guarantee lower runtime for small models on a CPU.
- **Head count.** With fixed $d_{\text{model}}$, more heads mean smaller $d_k$ per head rather than automatically more parameters.
- **Generation.** A decoder produces tokens sequentially. A KV cache stores earlier $K$ and $V$ values to avoid recomputation. Module 11 develops this mechanism further.

## Key points

- Attention calculates $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$.
- Multi-head attention uses independent projections but does not guarantee interpretable specialization.
- Boolean mask semantics depend on the API.
- Self-attention without position information is permutation equivariant.
- A Transformer block combines multi-head attention, an FFN, residual connections, and LayerNorm.
- Direct position access helps with long-range dependencies, but attention cost grows quadratically.

## Questions

1. What weight distribution results when a query closely matches one key and differs strongly from the others?
2. Why is scaling by $\sqrt{d_k}$ used? What happens without it for large $d_k$?
3. Does permuting tokens change the prediction of a model without positional encoding if self-attention outputs are averaged?
4. Which two masks are required for an autoregressive batch of variable-length sequences?

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- PyTorch documentation. *torch.nn.functional.scaled_dot_product_attention*. `https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html`.
- PyTorch documentation. *torch.nn.MultiheadAttention*. `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
