Language: **English (default)** | [Polski](lecture.md)

# Lecture 09 — Modeling sequences: from a vanilla RNN to LSTM and GRU

## Learning goals

1. Understand how sequential data differs from the data we have seen so far, and why the order of observations can be decisive.
2. Understand recurrence: the hidden state as a compressed representation of the sequence processed so far, updated by the same, shared function at every step.
3. Understand backpropagation through time (BPTT) as a consequence of unrolling the recurrence -- and why this leads to vanishing/exploding gradients.
4. Know the LSTM and GRU architectures as a mitigation (not a solution) of this problem -- and be able to measure that mitigation, not just declare it.
5. Correctly handle variable-length sequences (padding, masking) and understand how the many-to-one/many-to-many and unidirectional/bidirectional variants differ.

## Why this matters

Every model from modules 1-8 processed data in which the order of the input features had no fixed ordinal meaning (pixels do have 2D structure, but the CNN from module 6 uses that structure through the locality of the convolution, not through a sequential "step by step" process). Text, time series, audio signals, event logs -- in this data **order carries information that the feature set alone does not capture**: "dog bites man" and "man bites dog" have an identical set of words. A model that cannot make use of order cannot, by definition, distinguish such cases -- we will verify this literally in experiment C, not just state it.

## Intuition

Sequential data: text (a sequence of tokens), time series (a sequence of measurements), signals (a sequence of samples), events (a sequence of user actions) -- what connects them is that **element $t$ should be interpreted in the context of elements $1,\ldots,t-1$**, not in isolation. Sequences also usually have **variable length** -- different sentences have different numbers of words -- which requires a representation different from a fixed feature vector (we return to this in the padding section).

The **hidden state** $h_t$ is an attempt to compress "what matters from $x_1,\ldots,x_t$" into a vector of **fixed** size, independent of $t$. Intuition: instead of remembering the whole history, the model continuously **updates** a single summary, with the same function at every step.

❌ "The hidden state is a lossless memory of the entire sequence seen so far."

✅ $h_t$ has a **fixed, finite** size, regardless of whether $t=3$ or $t=300$ -- so it must *compress*, and compression by definition loses information. The longer the sequence, the more has to "fit" into the same vector -- we will see in experiment B that this has a direct, measurable effect on gradient propagation.

## Formal model: the vanilla RNN

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b), \qquad t = 1,\ldots,T$$

- $x_t \in \mathbb R^{d}$ -- the input at step $t$ (e.g. a token embedding from module 8, $d$ = embedding dimension).
- $h_t \in \mathbb R^{d_h}$ -- the hidden state at step $t$ ($d_h$ = hidden-state dimension, a hyperparameter independent of $d$).
- $W_x \in \mathbb R^{d_h\times d}$, $W_h\in\mathbb R^{d_h\times d_h}$ -- the recurrence matrices, $b\in\mathbb R^{d_h}$ -- the bias.
- $h_0$ -- the initial state, usually a zero vector (the default in `torch.nn.RNN`, unless given otherwise).

**A note on notation.** $h_t$ extends $h$ from module 2 (there: the output of a single MLP hidden layer) with a time index -- the same role (an intermediate representation), now updated over time instead of over depth. **Do not confuse** this with the unindexed $h$ from module 3 (the finite-difference step in `gradient checking`) -- there $h$ never carries a $t$ index, the context is unambiguous. $W_x, W_h$ carry a **subscript**, unlike $W^{(l)}$ from the MLP (a **superscript** = layer number, modules 2-3): here there are no "successive layers" in the sense of depth -- $W_x$ and $W_h$ are **the same two matrices, used repeatedly**, once per time step.

**A key property: parameter sharing.** Unlike an MLP, where every layer has its own $W^{(l)}$, an RNN uses **exactly the same** $W_x, W_h, b$ at step $1$, $2$, …, $T$ -- this is not $T$ independent networks, just a single function applied $T$ times with an updated state. The number of RNN parameters **does not depend on the sequence length** $T$ -- a model with the same weights will process a sequence of length 5 and one of length 500.

❌ "An RNN has a separate set of weights for every time step, similar to how an MLP has separate weights for every layer."

✅ Exactly the opposite -- this sharing of weights over time is the defining feature of an RNN, verified directly in the worked example below.

## Mathematics: unrolling over time and BPTT

**Unrolling.** $h_t = f(x_t, h_{t-1};\theta)$ applied $T$ times is, written out, a long composition of functions -- exactly the structure we already have a tool for from module 3: the chain rule. An RNN "unrolled in time" looks structurally like a very deep feedforward network with $T$ "layers," with the difference that **all layers share the same weights** $W_x, W_h$.

**BPTT (Backpropagation Through Time)** is exactly the backpropagation from module 3, applied to this unrolling -- we do not repeat the full derivation here. Let $\delta_t := \partial L/\partial h_t$ (an extension of $\delta^{(l)}$ from module 3: there, a superscript = layer, here a subscript = time step -- the same concept of an "incoming gradient," a different propagation axis). By the chain rule, propagating $\delta$ backward through one recurrence step:

$$\delta_{t-1} = \frac{\partial L}{\partial h_{t-1}} = \Big(\frac{\partial h_t}{\partial h_{t-1}}\Big)^{\!\top} \delta_t = W_h^\top \,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\, \delta_t$$

(The Jacobian $\partial h_t/\partial h_{t-1}$ is the product of the recurrence matrix $W_h$ and the elementwise derivative of $\tanh$ -- exactly the same Jacobian pattern for a linear layer + elementwise activation as in module 3, now applied between time steps instead of between layers.) To go back $k$ steps, **we multiply by $k$ such Jacobians in a row**:

$$\delta_{t-k} = \Big(\prod_{i=0}^{k-1} W_h^\top\,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\Big)\,\delta_t$$

**This is the source of the vanishing/exploding gradient.** The product of $k$ matrices: if the typical "amplifying power" of each factor (roughly: the largest eigenvalue of $W_h$, rescaled by the derivative $\tanh\in(0,1]$) is $<1$, the product **vanishes** exponentially with $k$; if $>1$, it **explodes** exponentially. For long sequences ($T$ on the order of tens/hundreds) both situations are nearly unavoidable under random initialization -- we will verify this numerically in experiment B, for several sequence lengths and without assuming in advance which situation will occur.

## LSTM: cell memory and gates

LSTM introduces a **second** state, $c_t$ (the cell state), alongside $h_t$ (the hidden state) -- $c_t$ acts as a "memory tape," access to which is controlled by three **gates** (values in $(0,1)$, interpreted as "how much to let through"):

$$
\begin{aligned}
i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) &&\text{(input gate)}\\
f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) &&\text{(forget gate)}\\
g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) &&\text{(candidate new content)}\\
o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) &&\text{(output gate)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t &&\text{(memory update)}\\
h_t &= o_t \odot \tanh(c_t) &&\text{(hidden state = a "read" from memory)}
\end{aligned}
$$

($\odot$ = elementwise multiplication, as in module 3.) **The difference between $h_t$ and $c_t$:** $c_t$ is the internal memory, updated by a **sum** ($f_t\odot c_{t-1} + i_t\odot g_t$) -- a path along which the gradient can flow **without** a repeated multiplication by a weight matrix at every step (unlike $h_{t-1}\to h_t$ in a vanilla RNN). $h_t$ is the "externally visible" version of this memory, passed through the output gate -- it is $h_t$ (not $c_t$) that goes to the classification head or the next layer.

**Consistency with `torch.nn.LSTM`.** PyTorch uses **two separate biases** per gate ($b_{i\cdot}$ and $b_{h\cdot}$, summed) instead of one $b$ -- mathematically equivalent to a single bias (the sum of two constants is a constant), just a different split of the parameters. The weights `weight_ih_l0`/`weight_hh_l0` are **four matrices concatenated into one**, in a fixed order $[i,f,g,o]$ -- verified in the notebook by manually slicing the weight tensor and comparing with `nn.LSTM` within numerical tolerance (`atol=1e-6`).

❌ "LSTM completely eliminates the vanishing gradient and guarantees that arbitrarily distant information will be remembered."

✅ LSTM **mitigates** the problem (the $c_{t-1}\to c_t$ path is additive, not multiplicative by $W_h$ at every step), but the gates themselves are $\sigma$/$\tanh$ functions, which also saturate (module 2) -- with poor initialization or very long sequences, LSTM can also have a vanishing gradient, just more slowly. We measure this directly in experiment B -- without assuming LSTM's advantage in advance.

## GRU: a simplification with two gates

GRU merges $h_t$ and $c_t$ into one state and uses two gates instead of three:

$$
\begin{aligned}
r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) &&\text{(reset gate)}\\
z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) &&\text{(update gate)}\\
n_t &= \tanh\big(W_{in} x_t + b_{in} + r_t\odot(W_{hn} h_{t-1} + b_{hn})\big) &&\text{(candidate)}\\
h_t &= (1-z_t)\odot n_t + z_t\odot h_{t-1} &&\text{(interpolation)}
\end{aligned}
$$

Intuition: $z_t$ controls **how much** of the previous state is retained (when $z_t\to 1$, $h_t\approx h_{t-1}$ -- the state "passes through" nearly unchanged, analogous to the additive path in LSTM); $r_t$ controls how much of $h_{t-1}$ affects the **new** candidate $n_t$ (when $r_t\to 0$, the candidate ignores the past). GRU has no separate $c_t$ -- fewer parameters (3 gates instead of LSTM's 4 sets of weights -- verified: exactly, not approximately, $3\times$ the parameters of a vanilla RNN), a simpler mechanism, sometimes comparable quality -- **not always**, it depends on the task (experiment D).

**Consistency with `torch.nn.GRU`:** the order of the gates in the weight tensors is $[r,z,n]$ (PyTorch documentation) -- unlike the LSTM case above, the notebook does **not** manually reproduce the GRU gate equations against `nn.GRU` here; `nn.GRU` is used directly in experiments B and D.

## Worked example: 3 steps of a vanilla RNN

$d=2$ (input), $d_h=2$ (hidden state), $T=3$:

$$W_x=\begin{pmatrix}0.5&-0.3\\0.1&0.8\end{pmatrix}\quad W_h=\begin{pmatrix}0.2&0.0\\-0.1&0.4\end{pmatrix}\quad b=\begin{pmatrix}0.1\\-0.1\end{pmatrix}\quad h_0=\begin{pmatrix}0\\0\end{pmatrix}$$

Sequence: $x_1=(1,0)$, $x_2=(0,1)$, $x_3=(1,1)$.

$$h_1 = \tanh(W_x x_1 + W_h h_0 + b) = \tanh\begin{pmatrix}0.6\\0.0\end{pmatrix} = \begin{pmatrix}0.5370\\0.0000\end{pmatrix}$$

$$h_2 = \tanh(W_x x_2 + W_h h_1 + b) = \begin{pmatrix}-0.0923\\0.5692\end{pmatrix}, \qquad h_3 = \begin{pmatrix}0.2743\\0.7767\end{pmatrix}$$

**Verified within numerical tolerance (`atol=1e-6`) against `nn.RNN`** (notebook): the same $W_x,W_h,b$ inserted as `rnn.weight_ih_l0`/`weight_hh_l0`/`bias_ih_l0` (with `bias_hh_l0=0`, so that the combined bias matches the $b$ above -- see the note on PyTorch's two biases), `nn.RNN` gives the same three states to the sixth decimal place. **The same $W_x, W_h, b$** used three times -- not three different sets of weights.

## Many-to-one, many-to-many, and directionality

- **Many-to-one**: a whole sequence -> one label (e.g. text classification, this module's lab) -- we use $h_T$ (the last state) as the representation of the whole sequence.
- **Many-to-many**: a sequence -> a sequence of the same length (e.g. tagging every word) -- we use **all** of $h_1,\ldots,h_T$.
- **Unidirectional RNN**: $h_t$ depends only on $x_1,\ldots,x_t$ -- consistent with causal/streaming processing (at time $t$ we do not know the future).
- **Bidirectional RNN**: two independent RNNs, one reading the sequence from the left, the other from the right, with the states combined (e.g. concatenated) -- $h_t$ depends on the **entire** sequence, including $x_{t+1},\ldots,x_T$.

❌ "A bidirectional RNN is simply better, so it is always worth using."

✅ A bidirectional RNN requires access to the **entire** sequence upfront -- it cannot be used for streaming processing (e.g. live speech transcription in real time, where the future does not exist yet). The choice between unidirectional and bidirectional follows from the **requirements of the task**, not from pure quality.

## Variable-length sequences: padding

A batch with sequences of different lengths requires a common tensor shape -- we pad the shorter sequences with a special `PAD` token up to the length of the longest sequence in the batch.

❌ "Padding is just a neutral token, the model will learn to ignore it."

✅ Padding **carries no content** and, processed without special handling, distorts the hidden state: `tanh`/the LSTM-GRU gates **do not know on their own** that a given step is filler, not a real token -- they update $h_t$ the same way as for a real input. Consequence: `out[:, -1, :]` (the last position in the tensor) corresponds to **the last real token only when there is no padding** -- otherwise it is the state **after** several padding steps, not after the real content. The correct solution: `pack_padded_sequence` (tells PyTorch the real lengths, so the RNN/LSTM/GRU does not process the padding at all) or manually gathering $h_{\ell_i}$ for each sequence $i$ at its real length $\ell_i$. Verified in the notebook: the same text, batched with different-length "neighbors," gives a **different** representation under naive `out[:,-1,:]`, and an **identical** one under correct handling.

## Worked computational example -- the experiment

Notebook (`examples.en.ipynb`):

- **A**: unrolling over time -- a manual RNN cell (basic PyTorch operations) vs. `nn.RNN`, the effect of changing $x_1$ on later states, weight sharing.
- **B**: the vanishing gradient -- the norm of $\partial L/\partial x_1$ as a function of sequence length $T\in\{2,5,10,20,40,80\}$, RNN vs. LSTM vs. GRU, multiple seeds; separately, the norm of $\partial L/\partial W_h$ (a different quantity, different behavior with $T$).
- **C**: an order-dependent task -- sequences are permutations of the same set of tokens (identical bag-of-tokens), the label depends solely on the order of two specific tokens; a mean-pooling baseline vs. RNN vs. GRU.
- **D**: LSTM/GRU -- gates on a short sequence, tensor shapes, parameter count, and RNN/LSTM/GRU timing.

## What we observe

**A (unrolling over time).** The manual RNN cell agrees with `nn.RNN` to the fourth decimal place for all three steps. Changing only $x_1$ changes **every** later state ($h_1, h_2, h_3$) -- a direct consequence of the recurrence.

**B (vanishing gradient, $d_h=8$, loss = sum of squares of the last state).** The norm of $\partial L/\partial x_1$ as a function of $T$:

| $T$ | RNN | LSTM | GRU |
|---:|---:|---:|---:|
| 2 | $4.8\times10^{-1}$ | $6.1\times10^{-2}$ | $1.6\times10^{-1}$ |
| 10 | $9.8\times10^{-4}$ | $1.7\times10^{-3}$ | $2.0\times10^{-3}$ |
| 20 | $2.8\times10^{-7}$ | $2.5\times10^{-6}$ | $1.1\times10^{-5}$ |
| 40 | $2.7\times10^{-12}$ | $2.6\times10^{-10}$ | $3.0\times10^{-9}$ |
| 80 | $0$ | $1.5\times10^{-17}$ | $1.2\times10^{-17}$ |

All three architectures vanish to practically zero by $T=80$ (RNN literally underflows to $0$) -- LSTM/GRU **slow down** the vanishing, they do not eliminate it. Moreover, for **short** sequences ($T=2$) the RNN has a **larger** gradient than LSTM ($4.8\times10^{-1}$ vs $6.1\times10^{-2}$) -- gating's advantage only shows up for longer sequences, it is not universal. Checked across 5 seeds at $T=40$: the ordering GRU > LSTM > RNN holds in most, but not all, runs (e.g. seed 2: LSTM $1.9\times10^{-9}$ > GRU $2.8\times10^{-10}$) -- this is not a mathematical certainty for every initialization.

The gradient with respect to the shared matrix $W_h$ behaves **qualitatively differently**: in the RNN it stays on the order of $1$-$3$ for **every** tested $T$ (from $2$ to $80$), with no systematic decline -- because it sums contributions from all steps, and nearby steps do not have to "travel" far back. This is a different quantity from the gradient with respect to $x_1$, with different behavior -- the two must not be treated interchangeably.

**C (order-dependent task).** Mean-pooling baseline: val/test accuracy $0.45/0.51$ -- exactly chance level (2 classes). Vanilla RNN and GRU: **both** $1.00/1.00$ -- a perfect solution, even though the RNN has $498$ parameters versus $1330$ for GRU ($2.7\times$ fewer). The difference comes from **access to information** (averaging is permutation-invariant), not from model capacity.

**D (LSTM/GRU -- gates, parameters, time).** The manual LSTM gate equations (weight order $[i,f,g,o]$) agree with `nn.LSTM` within numerical tolerance (`atol=1e-6`), for both $h_t$ and $c_t$. Parameter count: LSTM is exactly $4\times$ an RNN, GRU exactly $3\times$ (for the same $d_{\text{in}}, d_h$) -- verified, not approximate. Forward-pass time ($d_{\text{in}}=32, d_h=64, T=50$, batch $64$): RNN $1.3$ms, GRU $2.0$ms ($1.6\times$), LSTM $3.4$ms ($2.6\times$) -- grows with the number of gates, but **not proportionally** to the parameter count ($3\times$/$4\times$) -- the same "parameters $\neq$ compute cost" pattern as in modules 5-7.

## Common misconceptions

❌ "An RNN has separate, independent weights for every time step."

✅ The same $W_x, W_h, b$ are used at every step -- verified directly in the worked example.

❌ "The hidden state is a perfect, lossless memory of the entire sequence."

✅ $h_t$ has a fixed size independent of $T$ -- it must compress, so it must lose information; the longer the sequence, the more "competes" for the same space.

❌ "LSTM and GRU solve the vanishing-gradient problem."

✅ They mitigate it (the additive $c_{t-1}\to c_t$ path), they do not eliminate it -- measured in experiment B: for sufficiently long sequences, the gradient also vanishes in LSTM/GRU, just more slowly than in a vanilla RNN.

❌ "A longer sequence always means a better prediction (more context)."

✅ A longer sequence is also a longer path for the vanishing gradient, and more opportunities to "forget" early information -- more input data is not automatically the same as a better training signal.

❌ "Padding is a harmless, neutral token."

✅ Padding actively affects the hidden state if it is not specifically handled -- see the padding section above.

❌ "The last element of the tensor after padding always corresponds to the last real token."

✅ Only when the sequence has no padding. Otherwise it is a step **after** the padding -- verified: the same representation of the same text changes depending on how long the sequences it was batched with were, if `pack_padded_sequence` or an analogous mechanism was not used.

❌ "`nn.Embedding` understands the meaning of text on its own."

✅ As in module 8: a freshly initialized `nn.Embedding` is random vectors -- structure (including any use of order by subsequent RNN layers) only emerges during training toward a specific objective.

❌ "Higher accuracy of a sequential model proves that it makes use of word order."

✅ Higher accuracy can come from other differences (capacity, parameter count, a different training budget) -- what proves the use of order is an experiment **specifically constructed** so that an order-blind baseline has no access to the decisive information (experiment C), not merely a difference in result on an arbitrary task.

❌ "A bidirectional model is an improved version of a unidirectional model, always better."

✅ A bidirectional RNN requires the entire sequence upfront -- unusable in a streaming/causal setting, regardless of whether it would give higher accuracy in an offline setting.

❌ "Recurrent networks are the best choice for every sequential problem."

✅ RNN/LSTM/GRU have real limitations (sequential processing = no parallelism over time during training, difficulty with very long context) -- the motivation for the attention mechanism (module 10), which gives the model more direct access to distant positions, instead of relying solely on the compressed $h_t$.

## Engineering implications

- **Cost vs. sequence length.** RNN/LSTM/GRU compute time grows **linearly** with $T$ -- but, unlike a CNN (module 6) or a linear layer, the steps **cannot be parallelized** during training within a single sequence: step $t$ needs $h_{t-1}$, so it must wait for step $t-1$. This is a fundamental limitation (not an implementation detail) -- one of the main motivations for architectures that can process positions in parallel (module 10).
- **Memory during BPTT.** Backpropagating through $T$ steps requires storing **all** the intermediate $h_t$ (and $c_t$ for LSTM) from the forward pass -- memory grows linearly with $T$, analogous to network depth in module 3, except here "depth" is sequence length.
- **Training vs. streaming inference.** During training we usually process whole sequences at once (we know $x_1,\ldots,x_T$ upfront). During streaming (e.g. live) inference, the model receives $x_t$ one at a time and must **maintain state** $h_{t-1}$ between calls -- an architecturally different mode of operation.
- **Risk of accidentally leaking state.** If the state $h_{t-1}$ from one example accidentally "leaks" into processing the next, independent example (a bug in the training loop -- no state reset between examples/batches), the model trains on a corrupted signal. Always explicitly initialize $h_0$ for every new, independent example.
- **Batching and padding for efficiency.** Batching speeds up training (using hardware parallelism along the batch dimension, not time), but requires padding sequences of different lengths to a common shape -- hence the need for the correct handling from the section above, since otherwise the efficiency gain comes at the cost of correctness.
- **Cost vs. quality of RNN/LSTM/GRU.** Measured in experiment D: LSTM has exactly $4\times$ the parameters of a vanilla RNN (4 sets of gates), GRU exactly $3\times$ -- this translates directly into compute time per step, measurable, not just theoretical.
- **Motivation for attention.** Instead of relying solely on the compressed $h_t$ as the only "window" into the past, the attention mechanism lets the model access the representation from any earlier position **directly**, weighted by how relevant it is to the current step -- without having to push that information through every intermediate recurrence step. This is the topic of module 10 -- here just the intuition, not the implementation.

## What to remember

- RNN: $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$, the same $W_x,W_h,b$ at every step.
- BPTT = backpropagation applied to the unrolling over time; multiplying $k$ Jacobians in a row is the source of the vanishing/exploding gradient.
- LSTM adds $c_t$ and three gates, GRU simplifies to one state and two gates -- both mitigate, not eliminate, the vanishing gradient.
- Padding requires explicit handling (`pack_padded_sequence` or manually gathering the last real step) -- otherwise a sequence's representation depends on what it happened to be batched with.
- RNN/LSTM/GRU have historical and pedagogical value and can still be a reasonable choice (small tasks, memory/compute constraints, streaming processing) -- they are not an architecture "to throw away," but they have real limitations that motivate the attention mechanism.

## Questions for students

1. You have an RNN with $d_h=16$ trained on sequences of length $10$. Can the same model (without retraining) process a sequence of length $100$? What changes, and what does not?
2. Why does the gradient with respect to the **shared matrix** $W_h$ not vanish the same way as the gradient with respect to the **first input** $x_1$, even though both "pass through" the same unrolled-in-time network?
3. You have a batch of two sequences: length 3 and length 8, padded to length 8. What exactly are the two errors that can occur if you simply take `out[:, -1, :]` for both?
4. Design (do not implement) a sequence-classification task in which a baseline based on averaging embeddings **theoretically cannot** reach an accuracy higher than chance. What condition must such a task satisfy?

## References

- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science, 14(2), 179-211.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP, 1724-1734.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. ICML, PMLR 28(3), 1310-1318.
- PyTorch documentation: *torch.nn.RNN*, *torch.nn.LSTM*, *torch.nn.GRU*, `https://pytorch.org/docs/stable/generated/torch.nn.RNN.html`.
- PyTorch documentation: *torch.nn.utils.rnn.pack_padded_sequence*, `https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html`.
