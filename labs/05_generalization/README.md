Language: **English (default)** | [Polski](README.pl.md)

# Lab 05. Recognizing overfitting and choosing regularization

The expected working time is about 2 hours. This lab uses concepts from
[`lectures/05_generalization/lecture.en.md`](../../lectures/05_generalization/lecture.en.md),
including empirical and expected risk, L2, weight decay, AdamW, dropout, and
early stopping. The problem differs from the lecture. Tabular classification
with informative and noise features replaces one-dimensional regression.

## 1. Data, splits, and leakage

The dataset has 14 features. Four contain information about the target, while
ten are pure noise. The training set contains 100 examples, and the validation
and test sets contain 200 examples each. Two features deliberately have a much
larger scale than the others. This reflects a common property of tabular data
and makes normalization important.

The notebook contains provided but **deliberately incorrect** code. It computes
the normalization mean and standard deviation over the full dataset before
splitting it into training, validation, and test sets.

The first task is a diagnosis of 3–5 sentences in a Markdown cell. It should
identify the error, explain why it constitutes data leakage, and state which
data must remain unavailable when fitting normalization statistics.

Next, implement the correct version. Compute statistics **only** on `X_train`,
then apply the same values to `X_train`, `X_val`, and `X_test` without fitting
them again. Compare the means and standard deviations from the incorrect and
correct versions for the first feature. The difference should be clearly
nonzero.

Only correctly normalized data is used in the remaining sections.

## 2. Baseline and an over-capacity model

The majority-class baseline and an MLP `[32,32]` with enough capacity to
overfit 100 examples are ready to run. They are not part of the implementation
task.

## 3. Training loop

Implement `train_model(model, epochs, lr, weight_decay=0.0,
use_adamw=False)`. The entire training set fits in one batch, so a mini-batch
loop is unnecessary. The implementation must follow `zero_grad() -> forward
-> loss (BCEWithLogitsLoss) -> backward() -> step()` and switch between
`model.train()` and `model.eval()` when computing validation loss. The function
returns `(train_losses, val_losses)`.

After training the `[32,32]` model without regularization for 1000 epochs,
analyze the provided plot. A 3–5 sentence Markdown response should report the
training and validation accuracy and identify the specific curve shape that
demonstrates overfitting. A difference between losses alone is insufficient.

## 4. Weight decay and dropout

For the `[32,32]` model, compare:

- `AdamW` with `weight_decay` values `0.0`, `0.01`, `0.1`, and `0.3`,
- `nn.Dropout` with `p` values `0.0`, `0.2`, `0.4`, and `0.6`. The provided
  `make_model(hidden_sizes, dropout_p=...)` adds dropout between hidden layers.

Collect validation accuracy and loss for every configuration. Present the
results in a table or readable printout.

## 5. Early stopping

The provided `train_with_early_stopping(model, epochs, patience, min_delta)`
skeleton contains a working training loop but does not save or restore the
best model state. The completed function returns a model **restored** to the
best epoch and the number of that epoch.

A provided assertion verifies the implementation. Validation loss computed
after the function returns must exactly match the best recorded loss.

## 6. Controlled comparison and final evaluation

Select one configuration with the lowest validation loss using the results
from sections 4–5. The choice may use no regularization, weight decay, dropout,
early stopping, or a combination supported by the results. Train the selected
configuration again with the same seed.

Only then, for the first time in the lab, evaluate the model on the test set.
Report test accuracy and loss.

## 7. Conclusions

The answers in a Markdown cell should refer to values from sections 4–6.

1. Which regularization method produced the largest improvement over the
   unregularized model? Did it change accuracy, loss, or both? Does this
   distinction matter?
2. Did normalization before splitting have a large or small effect on the
   final result? Why does the procedure remain incorrect regardless of the
   effect observed in one run?
3. Which limitation of the experiment, such as dataset size, random seed, or
   architecture, prevents one method from being considered universally best?

## Time budget

The values are indicative and have not been verified in a classroom setting.

| Block | Time |
|---|---|
| Section 1, leakage diagnosis and fix | 20 min |
| Section 2, orientation | 5 min |
| Section 3, training loop and curve interpretation | 25 min |
| Section 4, weight decay and dropout | 25 min |
| Section 5, early stopping | 20 min |
| Section 6, comparison and test | 15 min |
| Section 7, conclusions | 10 min |

Sections 1–7 form the required core. Optional extensions include adding values
to the `weight_decay` and `p` sweep or combining dropout with weight decay.

## Grading criteria

- Normalization statistics in section 1 are computed only from `X_train` and
  applied to the other sets without refitting.
- `train_model` and `train_with_early_stopping` work correctly, and the
  notebook runs from the relevant cell to the end without an error.
- Early stopping restores the best state, and the provided assertion is not
  skipped or removed.
- The test set appears exactly once, after configuration selection through
  validation.
- Interpretations refer to values and curve shapes from the executed run.

A full model solution exists locally with the instructor in
`decisions/solutions/05_generalization/` and is not part of this repository.
