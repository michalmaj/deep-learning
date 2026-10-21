Language: **English (default)** | [Polski](README.pl.md)

# Lab 01. PyTorch basics, tensors, devices, and a first model

**Time:** about 2 hours.

The lab extends the mechanism introduced in
[`lectures/01_intro/lecture.en.md`](../../lectures/01_intro/lecture.en.md)
through linear regression, gradients, and a training loop. The experiment
uses multiple features and includes a case where a linear model *is not
enough*.

## 1. The problem

The lecture introduced a single-variable model, $\hat{y} = wx + b$.
Real-world datasets rarely contain only one feature. This lab extends the
model to **multivariate** linear regression in matrix form.

$$\hat{y} = Xw + b$$

The matrix $X$ has shape $(N, 3)$ and contains three features. The vector
$w$ has shape $(3,)$.

The goal is to predict a continuous variable $y$ from the features $x_1$,
$x_2$, and $x_3$. One feature affects the data-generating process in a
**nonlinear** way. The work therefore includes both model training and the
**detection and interpretation** of the resulting systematic error.

## 2. Dataset

The data is synthetic and generated locally in section 1 of
`lab.en.ipynb`. The lab does not require any additional downloads. The
data-generation code is provided and is not part of the task.

## 3. Metrics and data split

After generating the data, two functions used throughout the lab must be
completed. The relevant places are marked as TODO in section 2.

- a function splitting example indices into a training and validation set
  (about 80/20),
- the **MSE** (mean squared error) and **$R^2$**
  ($1 - SS_{\text{res}}/SS_{\text{tot}}$) metrics.

The baseline, training, and evaluation all use these functions.

## 4. Baseline

A reference point is needed before training. The notebook provides a
baseline that predicts the mean value of $y$ in the training set. It does
not use $x$, so a useful model should perform better. The MSE and $R^2$
values are calculated with the functions from section 3.

## 5. Model and training

The following cells contain places marked as `# TODO`. Until they are
completed, the code raises `NotImplementedError`. The required work has
three parts.

1. **`forward(X, w, b)`**, the linear model prediction in matrix form. It
   generalizes `w * x + b` from the lecture to a matrix product.
2. **The closed-form solution** (*normal equation*)
   $w^\* = (X^\top X)^{-1} X^\top y$, with a column of ones for the bias.
   It provides a gradient-free reference for the gradient-trained model.
3. **A gradient-descent training loop** that extends the lecture's loop to
   a vector-valued $w$. It does not use `nn.Module` or `optim`, so the
   `forward → loss → backward → update` steps must be written directly.

The provided code handles data generation, device selection, the baseline,
and plotting. It does not require changes.

## 6. Experiment

The experiment trains the model and compares the baseline loss, the
gradient-trained model loss, and the closed-form solution loss. The last
two should be **close** because the problem is convex and both methods
seek the same optimum. A large difference indicates an issue in the
training loop, such as the learning rate, update step, or number of epochs.

## 7. Evaluation

The trained model's MSE and $R^2$ must be calculated on the **validation**
set. Metrics from the training set do not show how the model performs on
new data.

## 8. Error analysis

The provided code plots the residuals ($\hat{y} - y$) against each feature
in the validation set. One feature produces a **systematic, non-random
pattern** instead of random noise around zero.

The Markdown response should identify the feature, describe the pattern in
3–5 sentences, and connect it to the limitations of a linear model. The
underlying formula is not given and does not need to be guessed. The
analysis should infer from the residuals a type of relationship that the
linear model cannot represent.

## 9. Conclusions

The final notebook section contains two questions for a short Markdown
response.

- Could this same problem be solved better by adding a new *feature*
  (e.g. the square of an existing one) without changing the model itself?
  What supports the answer?
- Why is comparing against the closed-form solution a useful correctness
  test for the training loop?

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

A full model solution is stored locally by the instructor
(`decisions/solutions/01_intro/`) and is not part of this repository.
