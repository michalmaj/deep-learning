Language: **English** | [Polski](lecture.md)

# Lecture 09. Sequence modeling with RNN, LSTM, and GRU

## Learning goals

1. explain why order matters in sequential data,
2. describe the hidden state and parameter sharing over time,
3. connect BPTT with vanishing and exploding gradients,
4. present LSTM and GRU as mechanisms that mitigate long-range dependency problems,
5. handle padding, variable lengths, and common output patterns correctly.

## Why this matters

Text, time series, audio, and event logs carry information in the order of their observations. “Dog bites man” and “man bites dog” contain the same words but have different meanings. A model that ignores order cannot distinguish between them.

A CNN uses the spatial structure of an image but does not process it one step at a time. An RNN introduces a state that is updated after every sequence element. One experiment will show a task where averaging token representations removes the decisive information while a recurrent model retains it.

## Intuition

An element $x_t$ should be interpreted in the context of $x_1,\ldots,x_{t-1}$. Sequences also have different lengths, so they cannot always be placed directly in a batch with one common shape.

The hidden state $h_t$ compresses the relevant history into a fixed-size vector. The same function updates it at every step.

❌ The hidden state is a lossless memory of the entire sequence.

✅ The size of $h_t$ does not depend on $t$. The same vector must summarize three or three hundred elements, so some information is lost during compression. A long recurrent path also makes it harder for the gradient to reach the start of the sequence.

## Vanilla RNN

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b), \qquad t = 1,\ldots,T$$

- $x_t\in\mathbb R^d$ is the input at step $t$,
- $h_t\in\mathbb R^{d_h}$ is the hidden state,
- $W_x\in\mathbb R^{d_h\times d}$ and $W_h\in\mathbb R^{d_h\times d_h}$ are weight matrices,
- $b\in\mathbb R^{d_h}$ is the bias,
- $h_0$ is usually a zero vector.

The symbol $h_t$ extends the hidden representation notation from module 2 with a time index. The subscripts in $W_x$ and $W_h$ are not layer numbers. These are the same matrices used repeatedly.

### Parameter sharing

An RNN uses one set of $W_x,W_h,b$ at every step. Its parameter count does not depend on the length $T$, so the same model can process sequences of length 5 and 500.

❌ Every time step has a separate set of weights, just as each MLP layer does.

✅ Weight sharing over time is a defining property of an RNN. The unrolled graph contains repeated uses of one cell, not a collection of independent networks.

## Unrolling and BPTT

The recurrence can be unrolled into $T$ applications of $f(x_t,h_{t-1};\theta)$. Its structure resembles a feedforward network of depth $T$, but all steps share their parameters.

BPTT, or *Backpropagation Through Time*, applies ordinary backpropagation to this unrolled graph. For $\delta_t=\partial L/\partial h_t$, one backward step is

$$\delta_{t-1}=W_h^\top\,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\,\delta_t.$$

Moving back by $k$ steps requires a product of $k$ Jacobians.

$$\delta_{t-k}=\Big(\prod_{i=0}^{k-1}W_h^\top\,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\Big)\delta_t$$

If the typical amplification of these factors is below 1, the gradient vanishes exponentially. If it is above 1, the gradient explodes. The derivative of $\tanh$ belongs to $(0,1]$, so activation saturation further encourages vanishing.

## LSTM

LSTM adds the cell state $c_t$ and three gates that control information flow.

$$
\begin{aligned}
i_t &= \sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi}) \\
f_t &= \sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf}) \\
g_t &= \tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg}) \\
o_t &= \sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho}) \\
c_t &= f_t\odot c_{t-1}+i_t\odot g_t \\
h_t &= o_t\odot\tanh(c_t)
\end{aligned}
$$

$i_t$ is the input gate, $f_t$ the forget gate, and $o_t$ the output gate. The candidate $g_t$ supplies new content. The state $c_t$ acts as memory, while $h_t$ is the visible output passed to the next component.

The update of $c_t$ contains an additive path. The gradient does not have to cross the full recurrent matrix at every step, which mitigates problems with long-range dependencies.

PyTorch stores two biases per gate and concatenates four matrices in the order $[i,f,g,o]$. The notebook manually slices `weight_ih_l0` and `weight_hh_l0` and matches `nn.LSTM` at `atol=1e-6`.

❌ LSTM completely eliminates the vanishing gradient.

✅ Its gates still use $\sigma$ and $\tanh$, which can saturate. LSTM usually slows down vanishing but does not guarantee that arbitrarily distant information will be preserved.

## GRU

GRU uses one state and two gates.

$$
\begin{aligned}
r_t &= \sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr}) \\
z_t &= \sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz}) \\
n_t &= \tanh\big(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})\big) \\
h_t &= (1-z_t)\odot n_t+z_t\odot h_{t-1}
\end{aligned}
$$

The gate $z_t$ determines how much of the previous state remains, while $r_t$ controls its influence on the new candidate. When $z_t\to1$, the state passes through almost unchanged.

GRU has three sets of weights and LSTM has four. With equal dimensions, they therefore have exactly $3\times$ and $4\times$ as many parameters as a vanilla RNN. PyTorch stores GRU gates in the order $[r,z,n]$.

## Three-step RNN example

For $d=d_h=2$ and $T=3$, let

$$W_x=\begin{pmatrix}0.5&-0.3\\0.1&0.8\end{pmatrix},\quad W_h=\begin{pmatrix}0.2&0.0\\-0.1&0.4\end{pmatrix},\quad b=\begin{pmatrix}0.1\\-0.1\end{pmatrix},\quad h_0=\begin{pmatrix}0\\0\end{pmatrix}.$$

The sequence contains $x_1=(1,0)$, $x_2=(0,1)$, and $x_3=(1,1)$.

$$h_1=\begin{pmatrix}0.5370\\0.0000\end{pmatrix},\qquad h_2=\begin{pmatrix}-0.0923\\0.5692\end{pmatrix},\qquad h_3=\begin{pmatrix}0.2743\\0.7767\end{pmatrix}.$$

Assigning the same weights to `nn.RNN` produces matching results at `atol=1e-6`. `bias_hh_l0` is set to zero so that PyTorch's combined bias equals $b$. One set of weights is used three times.

## Output patterns and directionality

- **Many-to-one** maps a complete sequence to one label and usually uses the final state.
- **Many-to-many** produces an output for every step, as in word tagging.
- **A unidirectional model** uses only $x_1,\ldots,x_t$ and supports streaming inference.
- **A bidirectional model** reads the sequence in both directions and combines two states.

❌ A bidirectional model is always better.

✅ It requires the complete sequence in advance, so it cannot operate causally on a live stream. Directionality follows from task requirements.

## Variable-length sequences

A batch requires a common shape, so shorter sequences are extended with a `PAD` token.

❌ Padding is neutral, and the model will learn to ignore it.

✅ A recurrent cell also updates its state for padding. `out[:, -1, :]` points to the last real token only in a sequence without padding. Correct handling uses `pack_padded_sequence` or selects $h_{\ell_i}$ at the real length $\ell_i$.

The notebook shows that the naive last state of the same text changes with the lengths of the other items in its batch. Correct length handling produces an identical representation.

## Experiments

The notebook [`examples.en.ipynb`](examples.en.ipynb) contains four parts.

1. A manual RNN cell is compared with `nn.RNN`.
2. The norm of $\partial L/\partial x_1$ is measured for $T\in\{2,5,10,20,40,80\}$ in RNN, LSTM, and GRU. The gradient with respect to $W_h$ is analyzed separately.
3. An order-only task compares mean pooling with RNN and GRU.
4. LSTM equations, parameter counts, and forward-pass times are compared across architectures.

## Observations

The manual cell and `nn.RNN` produce the same states. Changing $x_1$ affects every later $h_t$ as a direct consequence of recurrence.

### Gradient with respect to the first input

| $T$ | RNN | LSTM | GRU |
|---:|---:|---:|---:|
| 2 | $4.8\times10^{-1}$ | $6.1\times10^{-2}$ | $1.6\times10^{-1}$ |
| 10 | $9.8\times10^{-4}$ | $1.7\times10^{-3}$ | $2.0\times10^{-3}$ |
| 20 | $2.8\times10^{-7}$ | $2.5\times10^{-6}$ | $1.1\times10^{-5}$ |
| 40 | $2.7\times10^{-12}$ | $2.6\times10^{-10}$ | $3.0\times10^{-9}$ |
| 80 | $0$ | $1.5\times10^{-17}$ | $1.2\times10^{-17}$ |

At $T=80$, the gradient is practically zero in all three architectures. LSTM and GRU slow down the process but do not eliminate it. At $T=2$, the RNN gradient is even larger than the LSTM gradient. Across five seeds at $T=40$, the order GRU, LSTM, RNN appears in most runs but not all of them.

The gradient with respect to the shared $W_h$ behaves differently. In the RNN, it remains on the order of $1$ to $3$ for the tested lengths because it collects contributions from every step. Nearby steps do not have to propagate their signal through the full sequence. A parameter gradient must therefore not be treated as equivalent to the gradient with respect to $x_1$.

### Order-dependent task

Mean pooling reaches validation and test accuracy of $0.45/0.51$, which is chance level. RNN and GRU both reach $1.00/1.00$. The RNN has 498 parameters and the GRU has 1330. The advantage comes from access to order, not greater capacity.

### Parameters and runtime

The manual LSTM equations match `nn.LSTM` at `atol=1e-6`. LSTM has exactly $4\times$ and GRU $3\times$ as many parameters as RNN. For $d_{in}=32$, $d_h=64$, $T=50$, and a batch size of 64, the measured times are about 1.3 ms for RNN, 2.0 ms for GRU, and 3.4 ms for LSTM. Runtime does not grow in direct proportion to parameter count.

## Common misconceptions

❌ A longer sequence always improves a prediction.

✅ More context also creates a longer gradient path and more opportunities to lose early information.

❌ `nn.Embedding` understands text by itself.

✅ Initial embedding vectors are random. Structure emerges during training toward a specific objective.

❌ Higher accuracy from a sequential model proves that it uses order.

✅ A difference may come from capacity or training budget. A reliable experiment must make the decisive information unavailable to the order-insensitive model.

❌ RNN, LSTM, and GRU are the best choice for every sequential problem.

✅ Recurrence limits parallelism and access to very long context. These limitations motivate the attention mechanism in module 10.

## Engineering implications

- **Runtime.** Cost grows linearly with $T$, and consecutive steps within one sequence cannot run in parallel.
- **BPTT memory.** Training stores every $h_t$ and, for LSTM, every $c_t$. Memory use grows linearly with sequence length.
- **Streaming inference.** The model must preserve its state between successive calls.
- **State reset.** Carrying $h_t$ between independent examples leaks information. Every new example needs a correct $h_0$.
- **Batching.** Parallelism over the batch dimension requires padding and correct length handling.
- **Gate cost.** LSTM and GRU have more parameters and a higher per-step cost than RNN.
- **Attention.** Direct access to earlier positions shortens the information path and reduces reliance on one compressed state.

## Key points

- RNN shares $W_x,W_h,b$ across all steps.
- BPTT multiplies Jacobians over time, which causes vanishing or exploding gradients.
- LSTM and GRU mitigate the problem but do not eliminate it.
- Padding requires `pack_padded_sequence` or explicit selection of the last real step.
- Recurrence remains useful for small and streaming tasks, although its limitations motivate attention.

## Questions

1. Can an RNN with $d_h=16$, trained on sequences of length 10, process a sequence of length 100? What remains fixed, and what changes?
2. Why can the gradient with respect to $W_h$ behave differently from the gradient with respect to $x_1$?
3. What errors result from using `out[:, -1, :]` for sequences of length 3 and 8 padded to a common size?
4. How can a task be designed so that mean pooling cannot exceed chance accuracy but a sequential model can?

## References

- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science, 14(2), 179-211.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP, 1724-1734.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. ICML, PMLR 28(3), 1310-1318.
- PyTorch documentation. *torch.nn.RNN*, *torch.nn.LSTM*, *torch.nn.GRU*. `https://pytorch.org/docs/stable/generated/torch.nn.RNN.html`.
- PyTorch documentation. *torch.nn.utils.rnn.pack_padded_sequence*. `https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html`.
