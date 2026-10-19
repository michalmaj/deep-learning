Language: **English (default)** | [Polski](README.pl.md)

# Lab 04: an experimental comparison of optimization strategies

Time: about 2 hours. Builds on
[`lectures/04_optimization/lecture.en.md`](../../lectures/04_optimization/lecture.en.md)
(the SGD/momentum/Adam update rules), but on a **different problem** than
the lecture (multivariate regression, not the 2D bowl or half-moon
classification) and without implementing Adam from scratch -- you have the
algorithms' mathematics from the lecture, here you learn to
**experimentally compare and diagnose** optimization.

## 1. Problem and data (given)

Regression: three input features with **deliberately different scales**
(one in the range $[-1,1]$, one in $[-40,40]$, one in $[-3,3]$), a target
depending on all three (nonlinearly on the first). Data generated locally,
split **three ways**: train (70%) / validation (15%) / test (15%).

**Roles of the splits, which you must respect throughout the lab:**
- **train** -- for learning the model's parameters,
- **validation** -- for comparing optimizers and choosing a learning rate (you may "peek" at it as many times as you like),
- **test** -- you touch it **exactly once**, at the very end, for the already-chosen configuration. If you use the test set to make decisions along the way, it stops being an honest measure of how the model handles new data.

The different feature scales are not an accident -- you will see why they
matter.

## 2. Training loop -- your task

Implement `train_model(opt_name, lr, epochs, batch_size, model_seed,
data_seed, momentum)`: it creates a model (given function `make_model(seed)`),
the corresponding optimizer (`torch.optim.SGD` with/without `momentum`,
`torch.optim.Adam` -- you do **not** implement Adam from scratch), and
trains it, iterating over mini-batches (given function
`iterate_minibatches`) in the correct order: `zero_grad() -> forward ->
loss (MSE) -> backward() -> step()`. After every epoch, record the
training and validation loss (without `torch.no_grad()` here you would
needlessly spend memory on a graph you do not need). The function must
return `(model, train_losses, val_losses)`.

This is a **controlled experiment**: the same `model_seed` and the same
`data_seed` on every call mean the same weight initialization and the same
mini-batch order -- so differences in results between calls will come
solely from the choice of optimizer/learning rate, not chance. The
"predict the training mean" baseline is given -- this is your reference
point.

## 3. Comparison at a shared learning rate -- your task

Call `train_model` for `"sgd"`, `"momentum"` (default `momentum=0.9`), and
`"adam"`, all with `lr=0.01`, `epochs=20`. Plot the training and
validation loss curves for all three on one plot (each method its own
line).

**Before moving on:** look at the results. Did all three actually learn
anything?

## 4. Learning-rate sweep -- your task

For **SGD** (without momentum), test `lr` from the list `[0.0001, 0.001,
0.01]` (`epochs=20` each), collect the final validation loss for each. Do
the same for **momentum** (`momentum=0.9`) with the list `[0.00005,
0.0005, 0.005]`. Present the results in a simple table/printout.

## 5. Diagnosis -- your task

**Task (Markdown, 4-6 sentences):** in section 3, at least one of the
optimizers (other than Adam) probably learned nothing sensible at
`lr=0.01` -- the loss may even have exploded to `nan`. Based on the
results from section 4 (the `lr` sweep) and what you know from the lecture
about different feature scales and the loss's curvature: diagnose **why**
exactly `lr=0.01` failed for that optimizer, while Adam handled it without
trouble. It is not enough to write "lr was too large" -- explain **why**
this specific problem (this specific data) makes SGD/momentum so sensitive
to the choice of `lr`, while Adam is far less so.

## 6. Final evaluation -- your task

Based on the sweep from section 4 (and your knowledge of Adam from the
lecture), choose **one, best by validation loss** `lr` for each of the
three optimizers. Train each with the chosen `lr` (`epochs=25`), and then
-- **only now, once** -- evaluate each of the three trained models on the
**test** set. Present the result in a table: optimizer, chosen `lr`, test
loss.

## 7. Conclusions -- your task

Answer (Markdown, justified with real numbers from section 6, not
generalities):

1. Which optimizer achieved the lowest test loss? Does this result match
   what the lecture taught you about this specific type of problem
   (ill-conditioned by different feature scales)?
2. If you compared the same three optimizers on a problem **without** a
   difference in feature scales (all features normalized to a similar
   range) -- would you expect a similarly large difference between them?
   What is your prediction based on?

## Required core vs. extension (time guidance)

Sections 1-6 (about 100 min: the introduction and section 2 ~30 min,
sections 3-4 ~35 min, section 5 ~15 min, section 6 ~20 min) are the core --
they fully fulfill the lab's goal. Section 7 (~15-20 min) is a short, but
required, interpretation -- it is **not** an extension that can be
skipped, unlike further widening the `lr` sweep with more values or a
`momentum` other than `0.9`, which we leave as optional, if time remains.

## Grading criteria

- `train_model` works correctly (the correct
  `zero_grad/backward/step` order, correct mini-batches, returns the
  history).
- The comparison at a shared `lr` and the `lr` sweep from section 4
  actually show a difference in behavior (not three identical, flat
  curves).
- The diagnosis in section 5 refers to a specific cause (feature scale /
  the loss's curvature), not just the symptom ("it was too large").
- The final evaluation in section 6 touches the test set exactly once,
  after choosing `lr` based on validation -- not the other way around.
- The conclusions in section 7 refer to the actual numbers from section 6.

A full model solution exists locally with the instructor
(`decisions/solutions/04_optimization/`) and is not part of this
repository.
