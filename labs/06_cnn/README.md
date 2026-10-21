Language: **English (default)** | [Polski](README.pl.md)

# Lab 06. Designing, training, and analyzing a CNN

This lab takes about 2 hours and builds on
[`lectures/06_cnn/lecture.en.md`](../../lectures/06_cnn/lecture.en.md). It covers
convolution, `Conv2d`, pooling, parameter counts, and receptive fields. The
dataset is KMNIST, which contains handwritten kuzushiji characters: `o, ki, su,
tsu, na, ha, ma, ya, re, wo`. The goal is to examine how design choices within
a CNN affect quality, model size, and computational cost.

> **Why KMNIST instead of CIFAR-10?** Downloading CIFAR-10 took about 90 minutes
> in this environment, which is too long for a class. KMNIST downloads in about
> 75 seconds and has the same format as Fashion-MNIST ($28\times28$, grayscale).
> An optional CIFAR-10 extension requires changing the number of input channels
> from 1 to 3 and the image size from 28 to 32.

## 0. Setup (provided)

Library imports, device selection (`cpu` or `cuda`), and the random seed.

## 1. Data and class distribution (provided + short task)

KMNIST is downloaded automatically through `torchvision.datasets.KMNIST` and
is not committed to the repository. Details are available in
[`data/registry.yml`](../../data/registry.yml). The notebook displays the first
image's shape, the dataset sizes, and labeled examples.

**Code task, 2-3 lines.** Use `torch.bincount` to count and print the examples
from each of the 10 classes in the training subset. Based on these counts,
determine whether the dataset is approximately balanced and how that affects
the later interpretation of accuracy.

## 2. Train/val/test split and leak-free normalization

KMNIST provides a separate test set. A validation set must be taken from the
training data with non-overlapping indices. The prepared `make_split` function
is left for completion.

The `normalize_with_train_stats(X_train, X_val, X_test)` function should
calculate the pixel mean and standard deviation only on `X_train`. The same
statistics, without recomputation, must then be applied to `X_train`, `X_val`,
and `X_test`. This follows the same principle as lab 05, although pixel
brightness replaces the vector of tabular features.

A provided assertion checks that normalized `X_train` has a mean close to 0 and
a standard deviation close to 1.

## 3. Baseline model (provided)

The reference model contains `Flatten` and one `Linear` layer. It does not use
the image's two-dimensional structure. Its result provides a baseline for the
designed architectures.

## 4. CNN design, Variant 1

Implement `CNNVariant1(nn.Module)` with `nn.Conv2d`, an activation function,
and optionally `nn.MaxPool2d`. The number of layers and filters, the kernel
size, and the use of pooling are design decisions. The model must accept a
`(N, 1, 28, 28)` tensor and return `(N, 10)`.

**Required shape verification.** Pass one sample batch through the completed
model and print the tensor shape after every layer. This catches an incorrect
input size for `nn.Linear` after `flatten()` before training starts.

**Markdown task, 2-3 sentences.** Describe and briefly justify the chosen number
of layers and filters, kernel size, and use of pooling.

## 5. Training and validation of Variant 1

Call the provided `train_eval(model, epochs, lr, seed)` function for the
designed model. Record its validation accuracy, parameter count, and training
time.

## 6. CNN design, Variant 2

The second variant, `CNNVariant2`, must differ from the first in at least one
specific and justified aspect. Options include the number of filters, kernel
size, pooling, or depth. Before the code, state in 2-3 Markdown sentences which
hypothesis the change tests, such as the expected effect of additional filters
on accuracy, parameter count, and training time.

Repeat the shape verification from section 4 for the second variant.

## 7. Comparing the three models

Train the baseline and both CNN variants with at least two random seeds, such
as `seed=0` and `seed=1`. The table or printed summary should include validation
and test accuracy, parameter count, and measured training and inference time.
The test set may be used for the first time only in this section, after model
selection based on validation.

## 8. Confusion matrix and error analysis

Draw a confusion matrix on the test set for the variant selected by validation.
A provided function is ready for the data. Also display 6-8 misclassified
examples with their true and predicted labels.

**Markdown task, 3-5 sentences.** Identify the classes most often confused and
assess the visual similarity of the misclassified characters. Conclusions must
follow from the resulting matrix and examples rather than assumptions made
before the experiment.

## 9. Conclusions

Answers should refer to the actual results from sections 7 and 8.

1. Which variant performed best on validation, and did the same model perform
   best on the test set? What does any difference say about the reliability of
   a single comparison?
2. Did the variant with more parameters train faster or slower? Does the result
   support the intuition that fewer parameters mean fewer computations?
3. Does higher training accuracy for the best model demonstrate good
   generalization? The answer should cite specific results from section 7.
4. Which limitation, such as subset size, number of epochs, or selected random
   seeds, prevents treating one variant as universally better?

## Approximate time budget

| Block | Time |
|---|---|
| Sections 0-1 (setup, data exploration) | 10 min |
| Section 2 (split + leak-free normalization) | 15 min |
| Section 3 (baseline, provided, orientation) | 5 min |
| Section 4 (Variant 1 design + shape verification) | 30 min |
| Section 5 (training Variant 1) | 5 min |
| Section 6 (Variant 2 design + hypothesis) | 20 min |
| Section 7 (comparison, at least 2 seeds) | 10 min |
| Section 8 (confusion matrix + error analysis) | 15 min |
| Section 9 (conclusions) | 10 min |

Sections 0-9 form the required core. Training each variant on the reduced
subset takes a few seconds on CPU, leaving most of the time for design and
analysis.

**Optional extension.** Possible additions include a third architecture,
per-class precision and recall, or a repeated comparison with more data or
training epochs.

## Grading criteria

- Normalization calculates statistics only from `X_train` and applies them to
  `X_val` and `X_test` without recomputation. The provided assertion passes.
- Both variants accept `(N, 1, 28, 28)` and return `(N, 10)`. Shape verification
  in sections 4 and 6 is present and correct.
- Variant 2 differs from Variant 1 in a named and justified way.
- The comparison includes at least two random seeds and reports accuracy
  together with parameter count and time.
- The test set first appears in section 7, after selection based on validation.
- Conclusions use specific measured results rather than general statements or
  unverified assumptions.

A full reference solution exists locally with the instructor in
`decisions/solutions/06_cnn/` and is not part of this repository.
