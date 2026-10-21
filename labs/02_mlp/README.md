Language: **English (default)** | [Polski](README.pl.md)

# Lab 02. Tabular classification with an MLP

The estimated completion time is about two hours. This lab builds on the
perceptron, MLP, and activation-function material in
[`lectures/02_mlp/lecture.en.md`](../../lectures/02_mlp/lecture.en.md). It uses a
different dataset and a custom `nn.Module` implementation instead of
`nn.Sequential`. It also introduces a comparison of architectures and activation
functions, followed by an analysis of misclassified points.

## 1. Problem and data

The task is binary classification of points from two **concentric circles** with
added Gaussian noise. The outer ring forms class 0, and the inner circle forms
class 1. This is a different kind of nonlinear separability from the half-moons
in the lecture. The required decision boundary is a closed curve rather than a
single wave.

Section 1 of the notebook generates the data locally and splits it into training
and test sets. This code is provided and is not part of the assignment.

## 2. Metrics

The function in section 2 must be completed to calculate **TP, FP, TN, and FN**
from predictions and labels, followed by **accuracy, precision, and recall**.
Class 1, the inner circle, is treated as the positive class. The rest of the
notebook also uses this function to evaluate the baseline model.

## 3. Baseline model

The provided baseline contains a single linear layer, `nn.Linear(2, 1)`. It
repeats the central observation from the lecture on a different dataset. A
linear model cannot solve a nonlinear problem. Its result on the circles is
noticeably worse than on the half-moons and approaches random guessing. This
follows from the geometry of the data because no line can separate the ring
from the center.

## 4. MLP implementation

Section 4 requires an implementation of the `MLP(nn.Module)` class. Unlike the
lecture example, it must not use `nn.Sequential`. Both the number of hidden
neurons and the activation function should be configurable so that the same
class can support the comparison in section 6.

Section 5 requires a training function that performs `forward`, computes the
loss with `BCEWithLogitsLoss`, calls `backward`, and finishes with
`optimizer.step()`. The function should accept any `nn.Module` model.

## 5. Experiment

The provided code trains the default MLP configuration with the completed class
and training function, then compares its result with the baseline.

## 6. Architecture and activation comparison

The comparison must include **at least three** MLP configurations that differ in
the number of hidden neurons or the activation function. At least one should be
deliberately too weak, for example because it contains very few neurons. This
makes it possible to examine more than a simple relationship between model size
and performance. Each configuration must be trained, and its test metrics must
be collected in one table or printout.

## 7. Evaluation

The Markdown cell should interpret the table from section 6. The analysis must
identify the best configuration and any model with insufficient capacity. The
explanation should refer to the functions that each architecture can represent
and connect the result with the proof from the lecture.

## 8. Misclassification analysis

The provided code plots the best model's decision boundary and marks the
misclassified test points. The Markdown response should describe their location
relative to the boundary between the ring and the circle in 3–5 sentences. It
should determine whether these are difficult points near the true boundary and
affected by noise, or evidence of a systematic model error. The plot should
support the conclusion.

## 9. Conclusions

The final Markdown cell should answer two questions.

- Is the result from section 6 consistent with the conclusion of the proof about
  composing affine transformations? How does the empirical observation differ
  from the proof itself?
- What single practical principle for designing an MLP follows from the results
  of this lab?

## Grading criteria

- The metrics, `MLP` class, and training function work, and the notebook runs
  from the completed cell to the end without an error.
- Section 6 contains at least three genuinely different configurations,
  including one clearly weaker model.
- The interpretation connects the numerical results with the architecture's
  representational capacity.
- The error analysis refers to the location of the misclassified points in the
  plot.
- The conclusions are concise but supported by reasoning.

A complete model solution is stored locally by the instructor in
`decisions/solutions/02_mlp/` and is not part of this repository.
