Language: **English (default)** | [Polski](README.pl.md)

# Lab 04. An experimental comparison of optimization strategies

The expected working time is about 2 hours. This lab builds on the SGD,
momentum, and Adam update rules described in
[`lectures/04_optimization/lecture.en.md`](../../lectures/04_optimization/lecture.en.md),
but uses a different problem. Multivariate regression replaces the
two-dimensional quadratic function and half-moon classification from the
lecture. Adam is not implemented from scratch. The goal is to compare and
diagnose optimization experimentally.

## 1. Problem and data

The regression task has three input features with deliberately different
scales. The first lies in $[-1,1]$, the second in $[-40,40]$, and the third in
$[-3,3]$. The target depends on all three features, including a nonlinear
dependence on the first. The data is generated locally and split into training
(70%), validation (15%), and test (15%) sets.

Each split has a separate role.

- The **training set** is used to learn model parameters.
- The **validation set** is used to compare optimizers and select a learning
  rate. It may be evaluated multiple times.
- The **test set** is used once at the end of the lab to evaluate the selected
  configurations. Using it to make earlier decisions would prevent it from
  serving as an independent measure of performance on new data.

The different feature scales are a deliberate part of the experiment and
affect the optimization process.

## 2. Training loop

Implement
`train_model(opt_name, lr, epochs, batch_size, model_seed, data_seed, momentum)`.
The function creates a model with the provided `make_model(seed)` and selects
the appropriate optimizer. The available choices are `torch.optim.SGD` with
or without momentum and `torch.optim.Adam`. Adam should not be implemented
from scratch.

Training uses mini-batches returned by `iterate_minibatches` and follows the
order `zero_grad() -> forward -> loss (MSE) -> backward() -> step()`. The
training and validation losses are recorded after each epoch. Both values
should be computed inside `torch.no_grad()` to avoid building an unnecessary
graph. The function returns `(model, train_losses, val_losses)`.

The experiment remains controlled by using the same `model_seed` and
`data_seed` on every call. This preserves the initial weights and mini-batch
order, so differences between runs result from the optimizer and learning
rate rather than chance. The provided baseline, which always predicts the
training-set mean, serves as a reference point.

## 3. Comparison at a shared learning rate

Run `train_model` for `"sgd"`, `"momentum"` with the default `momentum=0.9`,
and `"adam"`. Every configuration uses `lr=0.01` and `epochs=20`. Plot the
training and validation loss curves for all three methods in one figure, with
a separate line for each method.

Before continuing, check whether each optimizer learned anything useful.

## 4. Learning-rate sweep

For SGD without momentum, test the values `[0.0001, 0.001, 0.01]` for 20
epochs each. For momentum with `momentum=0.9`, use `[0.00005, 0.0005, 0.005]`.
Collect the final validation losses in a simple table or readable printout.

## 5. Diagnosis

Write a diagnosis of 4–6 sentences in a Markdown cell. In section 3, at least
one optimizer other than Adam will probably fail to learn anything useful at
`lr=0.01`, and the loss may even reach `nan`.

The explanation should connect the sweep results from section 4 with the
feature scales and loss curvature discussed in the lecture. Stating only that
the learning rate was too large is not enough. The diagnosis needs to explain
why this dataset makes SGD and momentum highly sensitive to `lr`, while Adam
is much less sensitive.

## 6. Final evaluation

Select one `lr` for each optimizer using the validation loss from section 4
and the material about Adam from the lecture. Train each selected configuration
for 25 epochs.

Only after selection is complete, evaluate all three models on the test set
exactly once. The results table should contain the optimizer name, selected
`lr`, and test loss.

## 7. Conclusions

Record the answers in a Markdown cell and support them with values from
section 6.

1. Which optimizer achieved the lowest test loss? Does the result match the
   lecture's conclusions for a problem made ill-conditioned by different
   feature scales?
2. Would similarly large differences between the optimizers be expected after
   normalizing all features to comparable ranges? What supports this
   prediction?

## Required core and extension

Sections 1–6 take about 100 minutes and form the core of the lab. As a rough
guide, the introduction and section 2 require about 30 minutes, sections 3–4
about 35 minutes, section 5 about 15 minutes, and section 6 about 20 minutes.
Section 7 takes another 15–20 minutes and is a required interpretation of the
results.

Optional extensions include adding more values to the `lr` sweep or testing a
`momentum` value other than `0.9`.

## Grading criteria

- `train_model` follows the correct `zero_grad/backward/step` order, handles
  mini-batches correctly, and returns the loss history.
- The shared-`lr` comparison and the sweep from section 4 reveal differences
  in behavior rather than three identical flat curves.
- The diagnosis in section 5 identifies a specific cause related to feature
  scale and loss curvature, not only the symptom of an excessive `lr`.
- The test set is used exactly once, after selecting `lr` through validation.
- The conclusions in section 7 refer to the actual values from section 6.

A full model solution exists locally with the instructor in
`decisions/solutions/04_optimization/` and is not part of this repository.
