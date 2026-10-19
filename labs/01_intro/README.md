Language: **English (default)** | [Polski](README.pl.md)

# Lab 01: PyTorch basics — tensors, devices, your first model

Time: about 2 hours. Builds on the mechanism from
[`lectures/01_intro/lecture.en.md`](../../lectures/01_intro/lecture.en.md)
(linear regression, gradient, training loop), but **is not the same
experiment** — here you generalize it to multiple features and run into a
situation where a linear model *isn't enough*.

## 1. The problem

In the lecture we trained a single-variable model: $\hat{y} = wx + b$. In
practice, data rarely has just one feature. In this lab you'll build
**multivariate** linear regression in matrix form:

$$\hat{y} = Xw + b$$

where $X$ has shape $(N, 3)$ (three features), and $w$ has shape $(3,)$.

Task: predict a continuous variable $y$ from three features $x_1, x_2,
x_3$. One of these features enters the true data-generating process in a
**nonlinear** way — your job is not only to train a model, but to
**detect and interpret** that the linear model has a systematic error at
exactly that point.

## 2. Dataset

The data is synthetic and generated locally in the notebook (`lab.en.ipynb`,
section 1) — you don't download anything. The data-generating code is
given and ready: it is not part of the task.

## 3. Metrics and data split — your task

Right after generating the data, the notebook asks you for two functions
that will be used throughout the rest of the lab (TODO, section 2):

- a function splitting example indices into a training and validation set
  (about 80/20),
- the **MSE** (mean squared error) and **$R^2$**
  ($1 - SS_{\text{res}}/SS_{\text{tot}}$) metrics.

We do this at the start because the baseline, training, and evaluation —
everything that follows — relies on these same two tools.

## 4. Baseline

Before training any model, we need a reference point. The notebook gives
you a ready-made baseline: predict the mean of $y$ from the training set.
This is a "model" that doesn't look at $x$ at all — any sensible model
must beat it. The notebook computes its MSE/$R^2$ using your functions
from section 3.

## 5. Model and training — your task

In the following notebook cells you fill in (`# TODO`, raises
`NotImplementedError` until you do):

1. **`forward(X, w, b)`** — the linear model's prediction in matrix form
   (generalizing `w * x + b` from the lecture to a matrix product).
2. **The closed-form solution** (*normal equation*) —
   $w^\* = (X^\top X)^{-1} X^\top y$ (with a column of ones for the bias)
   — as a gradient-free reference point for your gradient-trained model.
3. **A gradient-descent training loop** — generalizing the lecture's loop
   to a vector-valued $w$ (still without `nn.Module`/`optim`: you write
   `forward → loss → backward → update` yourself).

The provided code (data generation, device handling, baseline, plots)
works as-is — you don't need to (and shouldn't) change it.

## 6. Experiment

Train the model and compare three numbers: the baseline's loss, the
gradient-trained model's loss, and the closed-form solution's loss. The
gradient-based loss should be **close** to the closed-form solution (this
is a convex problem — both methods search for the same optimum). If the
difference is large, that's a signal that something in your training
loop implementation (learning rate, update step, number of epochs) needs
fixing — don't ignore that gap.

## 7. Evaluation

Compute the MSE and $R^2$ of your trained model on the **validation**
set, not the training set — reporting metrics on data the model saw
during training says nothing about how well it generalizes.

## 8. Error analysis — your task

Plot the residuals ($\hat{y} - y$) against each of the three features on
the validation set (the plotting code is given). One of these three
features will show a **systematic, non-random pattern** in the residual
distribution — not random noise around zero.

**Task (Markdown cell, 3-5 sentences):** identify which feature this is,
describe the shape of the pattern, and explain — referring to what a
linear model *structurally* cannot represent — why this pattern appears.
The point isn't to guess the "true formula" (you won't see it directly in
the task) but to correctly recognize the *kind* of problem (what the
linear model can't express) from the residual plot alone.

## 9. Conclusions — your task

The notebook ends with a short section where you answer (in Markdown) two
questions:

- Could this same problem be solved better by adding a new *feature*
  (e.g. the square of an existing one) without changing the model itself?
  Justify your answer.
- Why is comparing against the closed-form solution a useful correctness
  test for your training loop?

## Grading criteria

- `forward`, the data split, the metrics, the closed-form solution, and
  the training loop all work (the notebook runs from the given cell to
  the end without error).
- The gradient-trained model's loss is close to the closed-form
  solution's loss (within a few percent, not multiples).
- The MSE/$R^2$ metrics reported in section 7 are computed on the
  **validation** set, not the training set.
- The residual-plot interpretation correctly identifies which feature has
  a nonlinear effect, and ties it to the linear model's limitation (not
  to "an error in the data" or "too small an $N$").
- The conclusions are justified, not one-liners.

A full model solution exists locally with the instructor
(`decisions/solutions/01_intro/`) and is not part of this repository.
