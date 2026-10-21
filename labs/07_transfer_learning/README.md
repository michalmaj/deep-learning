Language: **English (default)** | [Polski](README.pl.md)

# Lab 07: adapting a pretrained model to a new image-classification problem

Time: about 2 hours. Builds on [`lectures/07_transfer_learning/lecture.en.md`](../../lectures/07_transfer_learning/lecture.en.md)
(backbone/head, frozen backbone, partial fine-tuning, `requires_grad` vs.
`train()`/`eval()` vs. `no_grad()`), but uses **a different dataset and a
different question** than the lecture -- EuroSAT (satellite imagery, not
ground-level object photos like Caltech101 in the lecture) and the
question: **does transfer learning help when the new task's domain is
visually distant from ImageNet, and does partial fine-tuning give
anything beyond a frozen backbone alone here** (in the lecture -- with
strong ImageNet coverage -- it did not; you will see whether it is
different here).

## 0. Setup (provided)

Library imports, device, random seed.

## 1. Data: EuroSAT -- structure and exploration (provided + short task)

EuroSAT: 10 land-use classes from Sentinel-2 satellite imagery
(`AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake`), $64\times64$ RGB images.
Downloads automatically via `torchvision.datasets.EuroSAT` (never
committed to the repository -- see [`data/registry.yml`](../../data/registry.yml)).

**Task:** look at one sample image from each class (provided display
code). Do these images resemble anything from ImageNet (animals, objects,
"ground-level" scenes)? Write down, in a sentence, your initial intuition
-- you will come back to it in section 8.

## 2. Preprocessing consistent with the pretrained weights -- your task

ResNet18 requires specific preprocessing (input size, normalization) --
**the same one** used during training on ImageNet, otherwise the features
learned by the backbone will not make sense for the new data.

Implement `build_preprocess()`, which returns a transform consistent with
the `ResNet18_Weights.IMAGENET1K_V1` weights (hint: `weights.transforms()`
-- do not rewrite it by hand, use the ready-made transform from the
weights object, to be sure it is consistent with what the model was
trained on).

Verify: run one EuroSAT image through your function and check that the
resulting shape is `(3, 224, 224)` (a provided assertion checks this).

## 3. Split and subset extraction (provided)

500 images (50/class, balanced), split 30/10/10 into train/val/test.

## 4. Loading the model and replacing the head -- your task

Load `resnet18` with `ResNet18_Weights.IMAGENET1K_V1` weights (**pinned
explicitly**, not `DEFAULT`) and replace `model.fc` with a new `nn.Linear`
layer with the correct number of outputs (10 EuroSAT classes).

**Task (code, 2-3 lines):** verify programmatically (not just by reading
the code) that the new head has the correct number of output parameters,
and that it is **the same object** that `model.fc` refers to after the
assignment (not a separate, disconnected layer).

## 5. Freezing the backbone and training only the head -- your task

Freeze all backbone parameters (`requires_grad=False`), leave only the
head trainable. **Mind the order**: first replace the head (section 4),
only then freeze and build the optimizer -- you will see why in the
diagnostic section (section 7).

Write the training loop for the head alone (you may, but do not have to,
use the feature caching from the lecture -- it speeds up training, but is
not required). Set the model to the correct mode (`train()`/`eval()`) so
that **the backbone's BatchNorm statistics do not change** -- this is
exactly the bug discussed in the lecture (*Common misconceptions*
section); here you have to avoid it yourself, not just read about it.

**Verify programmatically**: after training, the backbone's frozen
parameters are identical to before (a provided assertion compares the
before/after state).

Record: validation accuracy and training time.

## 6. Partial fine-tuning -- your task

Unfreeze **`layer4`** (ResNet18's last convolutional block) together with
the head. Build an optimizer with **two parameter groups** at different
learning rates -- smaller for `layer4`, larger for the head (justification
in `lecture.en.md`, *Mathematics* section).

**Task (Markdown, 2-3 sentences):** what learning rate values did you
choose for the two groups, and why that specific ratio between them?

Verify programmatically: the frozen part of the backbone (`layer1`-`layer3`)
did not change, `layer4` and the head actually did change.

## 7. Diagnostic task -- your task

The following code (in the notebook) **does not work correctly** -- the
loss does not decrease at all despite many training steps, accuracy stays
at the level of random guessing. The code raises no exception -- it looks
correct.

**Task (Markdown, 4-6 sentences):** find the cause, referring to a
**specific line** of code and to what you actually observe (the loss,
`requires_grad`, the contents of `optimizer.param_groups`). It is not
enough to say "something is wrong with the optimizer" -- point out exactly
which object gets updates, and which one should but does not, and why this
is not obvious from checking `requires_grad` alone.

(A hint for verifying, not for solving right away: compare the `id()` of
the object in `optimizer.param_groups` with `id(model.fc.weight)` **after**
running the buggy cell.)

## 8. Comparing strategies -- your task

Compile the results of the frozen backbone (section 5) and partial
fine-tuning (section 6): validation and test accuracy, number of
trainable parameters, training time. You use the test set here **for the
first time**, after choosing the better configuration based on validation.

**Optional extension:** investigate the effect of **one** setting on the
result of partial fine-tuning -- choose either a different
$\eta_{\text{backbone}}/\eta_{\text{head}}$ ratio (e.g. $10\times$ vs.
$100\times$ smaller) **or** a different unfreezing range (`layer3`+`layer4`
instead of `layer4` alone). This additional training run is not part of
the required core.

## 9. Confusion matrix and error analysis -- your task

For the better (by validation) strategy: a confusion matrix on the test
set (a provided function is waiting for the data) and an analysis of 6-8
misclassified examples.

**Task (Markdown, 3-5 sentences):** which land-use classes are most often
confused? Looking at the images themselves -- do the mistakes look
sensible (similar bird's-eye appearance of terrain, e.g. different crop
types) or random?

## 10. Conclusions -- your task

Answer, justified with actual numbers from sections 5-9:

1. Go back to your intuition from section 1 -- did transfer learning from
   ImageNet actually help on data this visually distant (satellite
   imagery)? How large is the difference from what you expected?
2. Did partial fine-tuning give a **measurable** benefit over the frozen
   backbone alone, or did it only increase the cost? Compare with the
   lecture's result (Caltech101) -- is the answer the same, or different?
   What does that say about when it is worth investing in fine-tuning, and
   when it is not?
3. What is at least one limitation of this experiment (subset size, number
   of epochs, the specific unfreezing range) that prevents you from
   generalizing the conclusions to "strategy X is always better for
   satellite data"?

## Time budget (approximate, not verified in class)

| Block | Time |
|---|---|
| Sections 0-1 (setup, exploration) | 10 min |
| Section 2 (preprocessing + verification) | 10 min |
| Section 3 (split, provided) | 5 min |
| Section 4 (replacing the head + verification) | 10 min |
| Section 5 (frozen backbone + training the head) | 20 min |
| Section 6 (partial fine-tuning, 2 LR groups) | 20 min |
| Section 7 (diagnostic task) | 15 min |
| Section 8 (comparison of the two core strategies) | 10 min |
| Section 9 (confusion matrix + errors) | 10 min |
| Section 10 (conclusions) | 10 min |

Sections 0-10 are the mandatory core. Training on the reduced subset (500
images) takes a few to a few dozen seconds on CPU -- the time budget
should go into design decisions and analysis, not waiting for training.

**Extensions (optional, if time remains):** run the additional setting
experiment described in section 8; train from scratch (without pretrained
weights) as an additional reference point; or try full fine-tuning (all
layers unfrozen) and compare its benefit and cost with partial fine-tuning.

## Grading criteria

- Preprocessing consistent with the pretrained weights
  (`weights.transforms()`, not a hand-rolled "by eye" normalization); the
  shape assertion passes.
- Freezing the backbone in section 5 actually does not change its
  parameters (the assertion passes) and does not change BatchNorm
  statistics (correct model mode).
- Partial fine-tuning in section 6 uses two parameter groups with
  different LR; the frozen part of the backbone actually does not change.
- The diagnostic task (section 7) points to a specific cause in the code,
  not a generality.
- The test set appears only in section 8, after a selection based on
  validation.
- The conclusions in section 10 refer to specific numbers from your own
  run and explicitly compare with the lecture's result, not to
  generalities.

A full reference solution exists locally with the instructor
(`decisions/solutions/07_transfer_learning/`) and is not part of this
repository.
