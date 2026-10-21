Language: **English (default)** | [Polski](README.pl.md)

# Lab 03. Backpropagation from scratch in NumPy

The estimated completion time is about two hours. This lab extends the
derivation from
[`lectures/03_backprop/lecture.en.md`](../../lectures/03_backprop/lecture.en.md)
with an independent NumPy implementation that does not use `loss.backward()`.
The problem is single-variable regression rather than random data for formula
verification. The activation is tanh instead of sigmoid. Its derivative was
introduced in lecture 2 and is now used during the backward pass.

The goal is not to build a general automatic-differentiation engine. The
implementation covers forward and backward passes for one fixed architecture
with a single hidden layer. Understanding the mechanism matters more than
creating an alternative to PyTorch.

## 1. Problem and data

The task is single-variable regression for $x\in[-3,3]$ and
$y = \sin(2x) + 0.3x^2 + \varepsilon$. The function is clearly nonlinear, so a
linear model cannot represent it well. The data is generated locally with a
fixed seed. Code for data generation and visualization is provided.

## 2. Forward pass

The function `forward(X, W1, b1, W2, b2)` must be implemented for the network
`Linear(1, hidden) → tanh → Linear(hidden, 1)`. It should return all intermediate
values $Z^{(1)}, A^{(1)}$, and $Z^{(2)}$ because they are needed during the
backward pass. The returned array shapes must be checked before continuing.

## 3. Backward pass

The main task is to implement `backward(X, Y, Z1, A1, Z2, W2)`, which returns
`(dW1, db1, dW2, db2)`.

1. First, the derivative of MSE loss with respect to $Z^{(2)}$ must be derived
   for a batch. This is the formula from the lecture's *Batched version* section
   written with the notebook's notation.
2. The derivative $\tanh'(z) = 1 - \tanh^2(z)$ must then replace
   $\sigma(z)(1-\sigma(z))$. This step distinguishes the implementation from
   the example in lecture 3.
3. Propagation through both layers should use the matrix formulas, including
   transposes, $\odot$ multiplication, and sums over the batch for bias
   gradients.

The shapes of `dW1, db1, dW2, db2` must exactly match those of
`W1, b1, W2, b2` before section 4 begins.

## 4. Gradient verification

The `gradient_check(...)` function should build **the same** forward pass in
PyTorch with the same `W1, b1, W2, b2` values passed as
`torch.tensor(..., requires_grad=True)`. After calling `backward()`, the
gradients must be compared with the NumPy implementation, and the maximum
difference for each of the four arrays must be printed.

Training should begin only after this check passes. Expected differences are on
the order of $10^{-10}$ or smaller. NumPy uses `float64` by default, so the
precision limitations discussed in the lecture should not dominate here.

## 5. Training

The function `train(X, Y, hidden_size, lr, epochs)` must initialize weights
randomly, call `forward` and `backward`, update parameters manually with
`W -= lr * dW`, and record the loss history. `torch.optim` is not used because
optimizer mechanics are covered in lecture 4. The final loss must be compared
with the provided baseline that predicts the mean value of $y$.

## 6. Interpretation

The provided code plots the loss curve and the network prediction against the
data. The Markdown response should contain an analysis in 3–5 sentences. It
should assess the shape and rate of loss reduction as well as the fit across the
complete range of $x$. Any region with systematic deviation should be noted. If
there is none, the remaining scatter should be related to the noise used during
data generation.

## 7. A buggy gradient

The notebook contains a second, deliberately incorrect version of `backward`
and uses it to train an identical network on the same data. The type of error is
revealed only in the code. Its loss curve must be compared with the correct
implementation.

The Markdown analysis should contain 3–5 sentences. It must identify the omitted
multiplication or sum, refer to the relevant formula from the lecture, and
explain why the error changes the loss curve without causing a runtime error.

## Grading criteria

- `forward`, `backward`, `gradient_check`, and `train` work, and the notebook
  runs from the completed cell to the end without an error.
- The gradients from `backward` agree with PyTorch `autograd` within
  $10^{-8}$–$10^{-10}$.
- The model trained with manually calculated gradients achieves a clearly lower
  loss than the mean-prediction baseline.
- The interpretation in section 6 refers to specific values or plot shapes.
- The analysis in section 7 identifies the missing element of the backward-pass
  formula rather than only the symptom of higher loss.

A complete model solution is stored locally by the instructor in
`decisions/solutions/03_backprop/` and is not part of this repository.
