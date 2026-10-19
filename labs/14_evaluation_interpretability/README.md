Language: **English (default)** | [Polski](README.pl.md)

# Lab 14: error auditing and interpreting an image classifier's decisions

Time: ca. 120 minutes. Builds on
[`lectures/14_evaluation_interpretability/lecture.en.md`](../../lectures/14_evaluation_interpretability/lecture.en.md)
(metrics, calibration, Grad-CAM), but on a **different problem** than the
lecture (there: Fashion-MNIST, a small CNN from scratch). Here:
**EuroSAT** (Sentinel-2 satellite images, modules 07-08) + **ResNet18**
`IMAGENET1K_V1` with a frozen backbone and a trained head (following
module 07).

**Goal**: not a single Grad-CAM plot -- a **classifier audit report**,
answering concrete questions about quality, errors, confidence, and the
interpretation of the model's decisions.

## 0. Setup and model (provided)

Library imports, a random seed, loading EuroSAT
(`torchvision.datasets.EuroSAT`), loading `resnet18(weights=
ResNet18_Weights.IMAGENET1K_V1)` with a frozen backbone and a new
`nn.Linear` head (10 classes). The `extract_features()` function
precomputes the frozen backbone's features (drastically speeds up
training the head -- no need to recompute the convolutions every epoch).

## 1. Data and the train/validation/test split -- your task

EuroSAT has 27000 images in 10 classes (`AnnualCrop`, `Forest`,
`HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`,
`PermanentCrop`, `Residential`, `River`, `SeaLake`). For CPU speed, use a
**subset** (e.g. 1800 train / 400 val / 400 test, with no overlap).

**Task:**
1. Sample indices and split into `train_idx`/`val_idx`/`test_idx` --
   **verify programmatically** that there is no overlap.
2. Print the class distribution in each subset -- is it clearly
   unbalanced?
3. Short comment: what would happen if you used the **same** split to
   choose hyperparameters **and** for the final evaluation?

## 2. Training the head (provided + a short task)

Feature extraction (`extract_features`) and training a linear head on the
precomputed features are **provided** -- this is not the central topic of
this lab (transfer learning was module 07). **Task:** run the training on
**`train_idx`** (not on `val_idx` or `test_idx`!) and print the training
time.

## 3. Evaluation on an independent set -- your task

On **`test_idx`** (never touched until now): compute accuracy, macro-F1,
per-class precision/recall/F1. Build and draw the confusion matrix
($10\times10$).

**Task (comment):** which class has the lowest recall? Does this match
your intuition about which landscape categories look visually similar
(e.g. `PermanentCrop` vs `AnnualCrop`, `Highway` vs `Residential`)?

## 4. The most frequent and most costly errors -- your task

From the confusion matrix, find the **two** most frequently confused
class pairs (see the lecture -- don't pick them "by eye," read them off
the matrix). For **one** of them, consider (comment): does confusing
these two specific classes have a different "cost" than confusing another
pair (e.g. is `River`→`SeaLake` less costly than `Industrial`→`Residential`
in some real application)?

## 5. Confidence of correct and wrong predictions -- your task

Compute confidence (= $\max_c \hat y_c$) for all test predictions. Compare
the distribution for correct and wrong ones (histogram). Find the **most
confident wrong** prediction -- record its index, true and predicted
class, confidence.

## 6. Choosing samples for detailed analysis -- your task

Define an **explicit procedure** (not "by eye") for choosing 3 examples
for Grad-CAM/occlusion: (a) a correct, high-confidence prediction; (b) a
low-confidence error (the model "knows" it's unsure); (c) a
high-confidence error (misleading). Record exactly how each was chosen
(e.g. "the first example in `test_idx` for which condition X holds").

## 7. Grad-CAM -- your task

Implement the `GradCAM` class (a forward hook on the chosen layer's
activation, `register_full_backward_hook` on the gradient, $\alpha_k^c$ =
global-average-pool of the gradient, weighted sum + ReLU, bilinear
upsampling, **handling the zero-spread case** for the map). Use
`backbone.layer4` as the chosen layer (ResNet18's last convolutional
block, before `avgpool`).

**Note, verified in practice**: the backbone is **frozen**
(`requires_grad=False`). This means `model.eval()` alone **is not
enough** -- you need `x.requires_grad_(True)` on the **input**, otherwise
the gradient graph never reaches `layer4` (see the lecture, the
`eval()`/`no_grad()`/`inference_mode()` section).

## 8. Comparing heatmaps and occlusion sensitivity -- your task

For the **three** examples from section 6: compute and draw Grad-CAM (for
the **predicted** class). For **one** of them (a correct, high-confidence
prediction), also compute occlusion sensitivity ($32\times32$ patch,
stride $32$, baseline = normalized mid-gray, **consistent** with
`ResNet18_Weights.IMAGENET1K_V1.transforms()` -- **not** raw zero, since
that is not zero in the normalized space).

**Task (comment):** do the Grad-CAM and occlusion regions **agree** or
**differ**? Does that make sense, given that these are **different
questions** (local gradient vs. the effect of physically removing a
region)?

## 9. Subgroup analysis OR robustness -- your task (choose ONE)

**Option A -- subgroups (brightness):** split `test_idx` into
"bright"/"dark" halves (median mean pixel value). Compute accuracy in
each group. **Verify the class distribution in each group** -- could the
accuracy difference come from differences in class distribution, not
brightness itself?

❌ "An accuracy difference between brightness groups proves the model is
unfair/biased toward bright or dark images."

✅ **This cannot be assumed without checking.** Brightness **correlates**
with class (e.g. forest is usually darker than industrial terrain) -- the
accuracy difference may be **largely** a consequence of one group having
more examples from harder classes, not an effect of brightness itself.
**Verify this explicitly**, don't assume it.

**Option B -- robustness (horizontal flip):** satellite images have no
"canonical" orientation (unlike, say, photos of people) -- a horizontal
flip **should** preserve the label. Compute accuracy/confidence on
`test_idx` and its flipped version. Find **specific** examples whose
prediction **changed**.

❌ "If the model is robust to a flip, it is robust to a real distribution
shift (a different sensor, a different season, a different region)."

✅ **Not true.** A single, controlled transformation tests **only** that
one transformation -- it does not automatically generalize to other, real
sources of distribution change.

## 10. Diagnostic task

The following, **separately provided** code tries to compute Grad-CAM,
but has **two related** problems: (a) the whole analysis is wrapped in a
`torch.no_grad()` block (someone "optimized" the code, assuming inference
doesn't need gradients), **and** (b) even after removing `no_grad()`, the
backbone is frozen and the input lacks `requires_grad=True`.

**Task:**
1. Run the provided code -- observe the **specific** error (not "something
   doesn't work" -- give the exception type and location).
2. Remove `torch.no_grad()`. Run again -- **still** an error. Diagnose
   **why** removing `no_grad()` wasn't enough.
3. Fix it **fully** (add `x.requires_grad_(True)` on the input).
4. Verify: the Grad-CAM map is finite (`torch.isfinite`) and has a nonzero
   spread for **at least one** of your examples from section 6.

❌ "This bug requires guessing how PyTorch hooks work internally."

✅ **No -- the error gives a clean, explainable signal** (a `None`
gradient / a specific `RuntimeError`/`AttributeError`), understandable by
referring to what you are learning in this module (`eval()` vs.
`no_grad()` vs. the gradient graph), not by guessing hook implementation
details.

## 11. Final report -- your task

Write a short report (markdown, in the notebook), answering **directly**:

1. How well does the model classify each class? (specific numbers)
2. Which errors are most frequent? (specific pairs, from the confusion
   matrix)
3. Does the model sometimes show high confidence in wrong decisions?
   (a specific example)
4. Which input regions does Grad-CAM point to? (for your 3 examples)
5. Does the perturbation-based analysis (occlusion) support this
   interpretation?
6. Does the model's behavior change after your chosen input modification
   (section 9)?
7. What should be checked before using this model in practice?

**Important**: the examples from sections 6/8 are **chosen illustrations**,
not a representative sample of the whole set -- do not draw general
conclusions about model quality from 3 heatmaps. Base general conclusions
(points 1-2) on the **entire** test set (sections 3-4), not on selected
examples.

## Time budget (realistic, verified against actual CPU cost)

| Block | Time |
|---|---|
| Section 0 (setup, model) | 5 min |
| Section 1 (data, split) | 10 min |
| Section 2 (training the head) | 5 min |
| Section 3 (evaluation, confusion matrix) | 10 min |
| Section 4 (most frequent errors) | 8 min |
| Section 5 (confidence) | 8 min |
| Section 6 (choosing samples) | 5 min |
| Section 7 (Grad-CAM) | 13 min |
| Section 8 (comparison + occlusion) | 12 min |
| Section 9 (subgroups OR robustness) | 8 min |
| Section 10 (diagnostic task) | 10 min |
| Section 11 (report) | 10 min |

Total: **104 min** core + a **16-minute** buffer (120 min total). CPU
cost: feature extraction (~2600 images) ca. 30s **one-time**; training
the head (30 epochs on precomputed features) <1s; Grad-CAM/occlusion for
a few images -- seconds. The time budget goes into implementing Grad-CAM,
error analysis, and writing the report, not into waiting for the model.

## Grading criteria

- The train/val/test split is verified to be disjoint; the head is
  trained **exclusively** on `train_idx`.
- Metrics (accuracy, macro-F1, per-class) are computed on the
  **independent** `test_idx`, not on data used for training.
- Grad-CAM is implemented independently (hook-based), works on the frozen
  backbone (with the correct `requires_grad` on the input), verified
  (finite values, nonzero spread).
- The choice of samples for analysis (section 6) is **explicit and
  reproducible** -- not "I picked what looked good."
- The subgroup/robustness analysis (section 9) explicitly checks
  alternative explanations (confounding by class), does not draw a
  conclusion of "unfairness" without grounds.
- The diagnostic task correctly identifies **both** causes of the error,
  not just the first obvious one.
- The report answers all 7 questions, referring to specific
  numbers/examples, with a clear distinction between general conclusions
  (the whole set) and illustrations (selected examples).

A full reference solution exists locally with the instructor
(`decisions/solutions/14_evaluation_interpretability/`) and is not part of
this repository.
