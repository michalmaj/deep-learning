Language: **English (default)** | [Polski](README.pl.md)

# Lab 05: recognizing overfitting and choosing regularization

Time: about 2 hours. Builds on
[`lectures/05_generalization/lecture.en.md`](../../lectures/05_generalization/lecture.en.md)
(empirical vs. expected risk, L2/weight decay, AdamW, dropout, early
stopping), but on a **different problem** than the lecture -- tabular
classification with informative and noise features, not 1D regression.

## 1. Data and split (given) + data leakage -- your task

Data: 14 features ($4$ informative, $10$ pure noise -- unrelated to the
target), $100$ training examples, $200$ each for validation and test. Two
features have deliberately much larger scale than the rest (a common
situation in real tabular data) -- which is why normalization matters
here.

The notebook contains (given, but **deliberately buggy**) code that
computes the mean and standard deviation for normalization **over the
entire dataset, before the split** into train/validation/test.

**Task (Markdown, 3-5 sentences, before moving on):** what exactly is the
bug in this code? Why is this data leakage, even if the model "works
well"? What data, at what point, should not be visible when computing
these statistics?

**Task (code):** write a correct version -- normalization statistics
computed **only** on `X_train`, applied (the same ones, not recomputed) to
`X_train`, `X_val`, and `X_test`. Compare the computed means/standard
deviations (buggy vs. correct) for the first feature -- the difference
should be clearly nonzero.

From this point in the notebook you work **exclusively** on correctly
normalized data.

## 2. Baseline model and an over-capacity model (given)

A baseline (majority class) and an MLP `[32,32]` with capacity sufficient
to overfit on $100$ examples are ready to run -- this is not part of the
task.

## 3. Training loop -- your task

Implement `train_model(model, epochs, lr, weight_decay=0.0,
use_adamw=False)`: a correct mini-batch loop **is not needed here** (the
whole training set, $100$ examples, fits in one batch) -- but the order
`zero_grad -> forward -> loss (BCEWithLogitsLoss) -> backward() ->
step()`, and switching `model.train()`/`model.eval()` when computing the
validation loss, must be correct. Returns `(train_losses, val_losses)`.

**Task (Markdown, 3-5 sentences):** run `train_model` on the `[32,32]`
model without regularization for $1000$ epochs and plot the curves (the
plotting code is given). What is the training vs. validation accuracy? At
which point of the curve do you see evidence of overfitting -- not just
"the losses are different", but the specific shape that shows it?

## 4. Weight decay and dropout -- your task

For the `[32,32]` model:

- test `AdamW` with `weight_decay` from the list `[0.0, 0.01, 0.1, 0.3]`,
- test `nn.Dropout` (added between the hidden layers -- given function
  `make_model(hidden_sizes, dropout_p=...)`) with `p` from the list
  `[0.0, 0.2, 0.4, 0.6]`.

For each configuration, collect the accuracy and validation loss. Present
the results in a table/printout.

## 5. Early stopping -- your task

You are given the skeleton of `train_with_early_stopping(model, epochs,
patience, min_delta)` with a working training loop, but **without** the
logic for remembering and restoring the model's best state -- that is your
task. The function must, when done, return the model **restored** to the
state from the best epoch (not the last one), and the number of that
epoch.

Verify your implementation: the validation loss computed on the model
*after* calling the function must exactly match the remembered best loss
(a given assertion will check this).

## 6. Controlled comparison and final evaluation -- your task

Based on the results from sections 4-5, choose **one** configuration
(which one: no regularization / weight decay / dropout / early stopping --
or a combination, if your results give you grounds for it) with the best
**validation** loss. Train it once more (to be sure, with the same seed)
and evaluate it -- **for the first time in this lab** -- on the **test**
set. Print the test accuracy and loss.

## 7. Conclusions -- your task

Answer (Markdown, justified with real numbers from sections 4-6):

1. Which regularization method gave the largest improvement over the
   unregularized baseline? Did it improve only accuracy, only loss, or
   both measures -- and does that distinction matter?
2. Go back to section 1: did the pre-split normalization bug have a
   large, clear effect on the final result in your run, or a small one?
   Regardless of the answer -- why is the procedure itself wrong (computing
   statistics on the full dataset), even if the effect turned out small
   in this specific run?
3. What is at least one limitation of this experiment (data size, the
   specific random seed, the specific architecture) that means you should
   not generalize the conclusions from section 6 to "method X is always
   best"?

## Time budget (indicative, not verified in a real class)

| Block | Time |
|---|---|
| Section 1 (leakage diagnosis + fix) | 20 min |
| Section 2 (orientation, given) | 5 min |
| Section 3 (training loop + curve interpretation) | 25 min |
| Section 4 (weight decay + dropout) | 25 min |
| Section 5 (early stopping) | 20 min |
| Section 6 (comparison + test) | 15 min |
| Section 7 (conclusions) | 10 min |

Sections 1-7 are the required core. If time remains: widen the sweep from
section 4 with additional `weight_decay`/`p` values, or try combining
dropout + weight decay at once -- this is already an extension, not the
core.

## Grading criteria

- The normalization fix in section 1 actually computes statistics only
  from `X_train` and applies them (without recomputing) to the other
  sets.
- `train_model` and `train_with_early_stopping` work correctly (the
  notebook runs from the given cell to the end without error).
- Early stopping actually restores the best state -- the assertion in the
  notebook passes, it is not skipped/removed.
- The test set appears in the notebook exactly once, after choosing the
  configuration based on validation.
- The interpretations and conclusions refer to the specific numbers/curve
  shapes from your run, not generalities.

A full model solution exists locally with the instructor
(`decisions/solutions/05_generalization/`) and is not part of this
repository.
