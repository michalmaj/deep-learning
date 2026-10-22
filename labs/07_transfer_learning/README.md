Language: **English (default)** | [Polski](README.pl.md)

# Lab 07. Adapting a pretrained model to a new task

This lab takes about 2 hours and builds on
[`lectures/07_transfer_learning/lecture.en.md`](../../lectures/07_transfer_learning/lecture.en.md).
It covers a frozen backbone, partial fine-tuning, and the differences between
`requires_grad`, model modes, and `no_grad()`.

The experiment uses EuroSAT satellite images, a domain visually distant from the
everyday photographs in ImageNet and Caltech101. The goal is to determine whether
transfer learning still helps and whether partial fine-tuning improves on a frozen
backbone.

## 0. Setup (provided)

Library imports, device selection, and the random seed.

## 1. EuroSAT data (provided + short task)

EuroSAT contains 10 land-use classes from Sentinel-2 imagery: `AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River, SeaLake`. The RGB images have size $64\times64$.

The dataset is downloaded automatically through `torchvision.datasets.EuroSAT` and
is not committed to the repository. Details are available in
[`data/registry.yml`](../../data/registry.yml).

**Task.** After viewing one image from each class, record in one sentence whether the
data resembles typical ImageNet images. This initial intuition will be compared with
the result in section 10.

## 2. Preprocessing for the pretrained weights

ResNet18 requires the same input size and normalization used during ImageNet
training. `build_preprocess()` should return the transform associated with
`ResNet18_Weights.IMAGENET1K_V1` through `weights.transforms()`. The normalization
values do not need to be copied manually.

A provided assertion checks that one transformed EuroSAT image has shape
`(3, 224, 224)`.

## 3. Split and subset (provided)

The experiment uses 500 balanced images, 50 from each class. The training,
validation, and test split contains 30/10/10 images per class.

## 4. Model and new head

Load `resnet18` with explicitly selected `ResNet18_Weights.IMAGENET1K_V1` weights,
then replace `model.fc` with an `nn.Linear` layer containing 10 outputs. The `DEFAULT`
alias should not be used.

**Code task, 2-3 lines.** Confirm programmatically that the output count is correct
and that the new head is the same object referenced by `model.fc`, rather than a
disconnected layer.

## 5. Frozen backbone and head training

All backbone parameters should receive `requires_grad=False`, while the head remains
trainable. The order matters. Replace the head first, then freeze parameters and
construct the optimizer. Section 7 demonstrates the consequence of an incorrect
order.

Implement the training loop for the head. Feature caching is optional. Set `train()`
and `eval()` modes so that the backbone's BatchNorm statistics remain unchanged while
the head is trained.

A provided assertion compares the backbone parameters before and after training.
Record validation accuracy and training time.

## 6. Partial fine-tuning

Unfreeze `layer4` together with the head and create an optimizer with two parameter
groups. `layer4` should use a lower learning rate than the head.

**Markdown task, 2-3 sentences.** State the learning rates for both groups and justify
their ratio.

Verification should confirm that `layer1` through `layer3` remain unchanged while
`layer4` and the head are updated.

## 7. Diagnostic task

The notebook code appears correct and raises no exception, but the loss does not
decrease and accuracy remains near random guessing.

**Markdown task, 4-6 sentences.** Identify the specific line causing the problem and
describe the observed loss, `requires_grad` values, and contents of
`optimizer.param_groups`. The explanation should state which object receives updates,
which object should receive them, and why checking `requires_grad` alone does not
expose the error.

Comparing the `id()` of the object in `optimizer.param_groups` with
`id(model.fc.weight)` after the faulty cell can help verify the diagnosis.

## 8. Comparing strategies

Compare the frozen backbone and partial fine-tuning using validation and test
accuracy, trainable parameter count, and training time. The test set may be used for
the first time only here, after selecting a configuration through validation.

**Optional extension.** Change either the
$\eta_{\text{backbone}}/\eta_{\text{head}}$ ratio or the unfreezing range by adding
`layer3` to `layer4`. The extra run is not part of the required core.

## 9. Confusion matrix and error analysis

For the strategy selected through validation, draw a confusion matrix on the test set
and display 6-8 misclassified images.

**Markdown task, 3-5 sentences.** Identify the classes confused most often and assess
whether similar bird's-eye terrain appearance explains the mistakes.

## 10. Conclusions

Answers should use the measured results from sections 5-9.

1. Did the result support the initial intuition about transfer from ImageNet to
   satellite imagery? How large was the difference from the expectation?
2. Did partial fine-tuning measurably improve on the frozen backbone, or only increase
   cost? How does the result differ from the Caltech101 experiment in the lecture?
3. Which limitation, such as subset size, number of epochs, or unfreezing range,
   prevents generalizing the conclusion to all satellite data?

## Approximate time budget

| Block | Time |
|---|---|
| Sections 0-1 (setup, exploration) | 10 min |
| Section 2 (preprocessing + verification) | 10 min |
| Section 3 (split, provided) | 5 min |
| Section 4 (head replacement + verification) | 10 min |
| Section 5 (frozen backbone + head training) | 20 min |
| Section 6 (partial fine-tuning, 2 LR groups) | 20 min |
| Section 7 (diagnostic task) | 15 min |
| Section 8 (strategy comparison) | 10 min |
| Section 9 (confusion matrix + errors) | 10 min |
| Section 10 (conclusions) | 10 min |

Sections 0-10 form the required core. Training on 500 images takes from a few to a few
dozen seconds on CPU, leaving most of the time for decisions and analysis.

**Optional extensions.** Possible additions include the extra experiment from section
8, training a model from scratch, or full fine-tuning with a cost comparison.

## Grading criteria

- Preprocessing uses `weights.transforms()`, and the shape assertion passes.
- The frozen backbone and its BatchNorm statistics remain unchanged.
- Partial fine-tuning uses two parameter groups with different learning rates, and the
  frozen section remains unchanged.
- The diagnosis in section 7 identifies a specific cause in the code.
- The test set first appears in section 8 after validation-based selection.
- Conclusions cite measured values and compare the result with the lecture.

A full reference solution exists locally with the instructor in
`decisions/solutions/07_transfer_learning/` and is not part of this repository.
