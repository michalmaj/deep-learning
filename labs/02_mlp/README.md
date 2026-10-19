Language: **English (default)** | [Polski](README.pl.md)

# Lab 02: tabular classification with an MLP

Time: about 2 hours. Builds on
[`lectures/02_mlp/lecture.en.md`](../../lectures/02_mlp/lecture.en.md)
(the perceptron, MLP, activation functions), but **does not repeat** the
lecture's experiment -- different data, your own `nn.Module` implementation
(not `nn.Sequential`), an architecture/activation comparison, and a
misclassification analysis the lecture did not do.

## 1. Problem and data

Task: binary classification of points on a plane that lie on two
**concentric circles** (outer ring -- class 0, inner circle -- class 1),
with Gaussian noise. This is a different kind of nonlinear separability
than the lecture's half-moons: here the decision boundary that is needed is
a closed curve (a circle), not a single "wave".

The data is generated locally in the notebook (section 1, code given --
not part of the task) and split into a training/test set (given,
section 1).

## 2. Metrics -- your task

Implement (TODO, section 2) a function that computes from predictions and
labels: **TP, FP, TN, FN**, and from them **accuracy, precision, recall**
(class 1 -- the inner circle -- as the positive class). These functions
are used by the rest of the notebook, including evaluating the baseline.

## 3. Baseline (given)

A single linear layer (`nn.Linear(2, 1)`) -- the same "linear model fails
on a nonlinear problem" as in the lecture, just on different data. You will
see that on the circles, the linear model performs **noticeably worse**
than on the lecture's half-moons (closer to random guessing) -- this is
not a bug, it is a consequence of the data's shape: no line can sensibly
separate the ring from the center.

## 4. Your MLP -- your task

Implement (TODO, section 4) **your own class** `MLP(nn.Module)` (not
`nn.Sequential`, unlike the lecture) with a configurable number of hidden
neurons and a configurable activation function, so it can be used for the
comparison in section 6. Also implement (TODO, section 5) a training
function (`forward → loss (BCEWithLogitsLoss) → backward →
optimizer.step()`), analogous to the lecture's, but working on any
`nn.Module` passed to it.

## 5. Experiment (given, uses your code)

The notebook trains one default MLP using your class/function and compares
it against the baseline.

## 6. Architecture/activation comparison -- your task

Choose **at least three** configurations of your MLP, differing in the
number of hidden neurons and/or the activation function (TODO, section 6).
Include in the comparison at least one configuration you suspect will be
**too weak** (e.g. a very small number of neurons) -- a comparison that
only shows "bigger is always better" teaches nothing. Train each
configuration and collect the test metrics in a single table/printout.

## 7. Evaluation

Interpret (Markdown cell, TODO) the table from section 6: which
configuration was best, was any clearly too weak, and -- if so -- why (in
terms of what that architecture can and cannot represent, referring back
to the lecture's proof).

## 8. Misclassification analysis -- your task

The notebook plots (given) your best model's decision boundary with the
misclassified test points marked. Answer (Markdown, TODO, 3-5 sentences):
where do the misclassified points lie relative to the boundary between the
ring and the circle? Are these points that are "inherently hard" (close to
the true boundary, victims of noise), or rather evidence that the model
systematically misunderstands something? How will you decide, looking at
the plot?

## 9. Conclusions -- your task

Answer (Markdown, TODO):

- Does the result of the experiment in section 6 match what the lecture's
  proof about composing affine transformations teaches? How does what this
  experiment shows differ from what the proof showed?
- If you had to give one sentence of advice to someone designing an MLP
  architecture for a new classification problem, what would it be --
  based on what you observed in this lab, not generalities?

## Grading criteria

- The metrics, the `MLP` class, and the training function work (the
  notebook runs from the given cell to the end without error).
- The comparison in section 6 covers genuinely different configurations
  (not the same thing three times under a different variable name) and
  includes at least one clearly weaker one.
- The interpretation in section 7 correctly connects the numeric result to
  the architecture's representational capacity, not just describing it.
- The error analysis in section 8 refers to the specific location of the
  misclassified points on the plot, not generalities like "the model is
  sometimes wrong".
- The conclusions are justified, not one-liners.

A full model solution exists locally with the instructor
(`decisions/solutions/02_mlp/`) and is not part of this repository.
