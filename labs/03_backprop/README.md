Language: **English (default)** | [Polski](README.pl.md)

# Lab 03: backpropagation from scratch in NumPy

Time: about 2 hours. Builds on the derivation from
[`lectures/03_backprop/lecture.en.md`](../../lectures/03_backprop/lecture.en.md),
but you implement it **on your own in NumPy**, without `loss.backward()`,
for a different problem than the lecture (single-variable regression, not
random data for verifying formulas) and with a different activation
(**tanh**, not the lecture's sigmoid -- you learned the formula for tanh's
derivative in lecture 2, now you will use it in backpropagation).

You are not building a general-purpose automatic-differentiation engine --
only the forward and backward pass for **this one, fixed architecture** (1
hidden layer). The goal is to understand the mechanism, not to write an
alternative to PyTorch.

## 1. Problem and data (given)

Single-variable regression: $x\in[-3,3]$, $y = \sin(2x) + 0.3x^2 + \varepsilon$
-- a clearly nonlinear function (a linear model has no chance), generated
locally with a fixed seed. The code that generates and visualizes the data
is given.

## 2. Forward pass -- your task

Implement `forward(X, W1, b1, W2, b2)` for the network: `Linear(1, hidden)
-> tanh -> Linear(hidden, 1)`. The function must return **all** the
intermediate values ($Z^{(1)}, A^{(1)}, Z^{(2)}$), not just the final
prediction -- they will be needed for backpropagation in the next step.
Check the shapes of what you return before moving on.

## 3. Backward pass -- your task

This is the lab's core. Implement `backward(X, Y, Z1, A1, Z2, W2)`, which
returns `(dW1, db1, dW2, db2)`:

1. Derive (on paper, before you start writing code) the derivative of the
   MSE loss with respect to $Z^{(2)}$ for a batch -- this is exactly the
   formula from the lecture (the *Batched version* section), just
   substituting your own notation.
2. Derive the derivative $\tanh'(z) = 1 - \tanh^2(z)$ (given in lecture 2)
   and use it instead of $\sigma(z)(1-\sigma(z))$ from lecture 3 -- this is
   the point where your implementation must differ from the lecture's
   formula, not copy it one to one.
3. Implement backpropagation through both layers, using the matrix
   formulas from the lecture (transposes, $\odot$, summing over the batch
   for the biases).

Do not move on to section 4 until the shapes of `dW1, db1, dW2, db2` match
the shapes of `W1, b1, W2, b2` exactly.

## 4. Gradient verification -- your task

Implement `gradient_check(...)`, which builds **the same** forward pass in
PyTorch (from the same `W1, b1, W2, b2` values, as `torch.tensor(...,
requires_grad=True)`), calls `backward()`, and compares the resulting
gradients with yours -- print the maximum difference for each of the four
gradients.

**Do not move on to training until this check passes** (differences on the
order of $10^{-10}$ or smaller -- NumPy computes in `float64` by default,
so you should not run into the precision limitations from the lecture).

## 5. Training -- your task

Implement `train(X, Y, hidden_size, lr, epochs)`: initializes weights
randomly, in a loop calls your `forward` and `backward`, updates the
parameters **manually** (`W -= lr * dW`, without `torch.optim` -- you will
learn that mechanism in lecture 4), and collects the loss history. Compare
the final loss against the "predict the mean of $y$" baseline (given).

## 6. Interpretation -- your task

The notebook plots (given): the loss curve and a fit plot (your network's
prediction against the real data). Answer (Markdown, 3-5 sentences): does
the loss curve look the way you would expect after lectures 1 and 3 (shape,
rate of decrease)? Does the fit correctly capture the function's shape
across the whole $x$ range, or do you see a region with a systematic (not
random) deviation from the data -- and if not, can the remaining scatter of
points around the model's curve be explained by the noise the data was
generated with alone (section 1)?

## 7. A buggy gradient -- your task

The notebook contains (given) a second version of `backward`, with a
**deliberately** inserted bug (you will only see which one in the code --
we do not give it away here), and trains an identical network on the same
data with it. Compare the loss curves of both versions.

**Task (Markdown, 3-5 sentences):** identify what is missing in the buggy
version of `backward`, referring to a specific formula from the lecture
(which multiplication/summation was skipped), and explain why exactly this
omission leads to the observed loss-curve behavior (and not, say, to a
runtime error).

## Grading criteria

- `forward`, `backward`, `gradient_check`, and `train` work (the notebook
  runs from the given cell to the end without error).
- The gradients from `backward` agree with PyTorch `autograd` within
  $10^{-8}$-$10^{-10}$ (not $10^{-3}$ -- that would mean something is
  wrong).
- The model trained with your own gradients beats the "mean" baseline by a
  clear margin, not just marginally.
- The interpretation in section 6 refers to the specific shape of the
  plots/numbers, not generalities.
- The error interpretation in section 7 correctly identifies the missing
  element of the backpropagation formula, not just describing the symptom
  ("the loss is worse").

A full model solution exists locally with the instructor
(`decisions/solutions/03_backprop/`) and is not part of this repository.
