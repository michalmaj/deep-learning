Language: **English (default)** | [Polski](README.pl.md)

# Lab 06: designing, training, and analyzing a CNN for image classification

Time: about 2 hours. Builds on [`lectures/06_cnn/lecture.en.md`](../../lectures/06_cnn/lecture.en.md)
(convolution, `Conv2d`, pooling, parameter count, receptive field), but uses
a **different dataset** than the lecture -- KMNIST (handwritten kuzushiji
characters: `o, ki, su, tsu, na, ha, ma, ya, re, wo`), not Fashion-MNIST, and
asks a different question: not "CNN vs. MLP", but "how do architectural
decisions inside a CNN itself affect quality, parameter count, and cost?".

> **Why KMNIST, not CIFAR-10?** CIFAR-10 was the first candidate, but in
> this environment downloading it was measured at about 90 minutes -- too
> long for a class. KMNIST downloads in about 75 seconds and has the same
> dimensions as the Fashion-MNIST from the lecture ($28\times28$,
> grayscale), so the shape-counting skills transfer 1:1. If your
> environment has a faster connection, you can try the same notebook on
> CIFAR-10 as an extension -- this requires changing a few constants
> (number of input channels from 1 to 3, image size from 28 to 32).

## 0. Setup (provided)

Library imports, device (`cpu`/`cuda`), random seed.

## 1. Data: structure, class distribution, tensor shapes (provided + short task)

KMNIST downloads automatically via `torchvision.datasets.KMNIST` (never
committed to the repository -- see [`data/registry.yml`](../../data/registry.yml)).
The notebook prints the shape of the first image, the number of examples in
the training/test set, and sample images with their class labels.

**Task (code, 2-3 lines):** count and print the number of examples of each
of the 10 classes in the training subset (`torch.bincount`). Is the set
roughly balanced or not? This matters when interpreting accuracy later.

## 2. Train/val/test split and leak-free normalization -- your task

KMNIST already has a separate test set (a different file from the training
one). It is missing a validation set -- carve one out **from the training
data** (non-overlapping indices, the `make_split` function is waiting to be
finished).

Implement `normalize_with_train_stats(X_train, X_val, X_test)`: the mean and
standard deviation of the pixels, computed **only** on `X_train`, applied
(without recomputing) to `X_train`, `X_val`, and `X_test`. This is the same
leakage bug from lab 05, now on images instead of tabular features -- you
compute the statistics of a single number (mean pixel brightness), not a
vector, but the principle is identical.

Verify: after normalization, `X_train` has a mean close to 0 and a standard
deviation close to 1 (a provided assertion checks this).

## 3. Baseline model (provided)

A simple linear classifier (`Flatten` + a single `Linear` layer) as a
reference point -- it is not a CNN and does not use the 2D structure of the
image in any way. Its accuracy is the lower bound your architectures are
compared against.

## 4. Designing your CNN -- Variant 1 -- your task

Design and implement a `CNNVariant1(nn.Module)` class using `nn.Conv2d`, an
activation function, and (optionally) `nn.MaxPool2d` -- you choose the
number of convolutional layers, the number of filters, the kernel size,
whether to use pooling. The only requirement: the model must accept a
`(N, 1, 28, 28)` tensor and return `(N, 10)`.

**Required step -- shape verification:** after defining the class, run one
sample batch through the model and print the tensor's shape **after every
layer** (not just at the output). This is exactly the step that, in
practice, prevents the most common bug when writing a CNN -- an incorrectly
computed input size for `nn.Linear` after `flatten()`. If it does not match
-- fix it before moving on.

**Task (Markdown, 2-3 sentences):** what architectural choices did you make
(number of layers, filters, kernel, pooling), and how do you justify them?

## 5. Training and validating Variant 1 (loop provided, your call)

The `train_eval(model, epochs, lr, seed)` function is provided (the same
pattern as in module 05/lecture 06) -- call it for your model and record
validation accuracy, parameter count, and training time.

## 6. Designing Variant 2 -- your task

Design a **second** variant (`CNNVariant2`), differing from the first in
**at least one specific, justified aspect** -- e.g. a different number of
filters, a different kernel size, adding/removing pooling, a different
depth. Before writing any code, write down (Markdown, 2-3 sentences) **what
specific hypothesis you are testing** with this change (e.g. "more filters
in the first layer should improve accuracy, but will increase parameter
count and training time").

Repeat the shape verification from section 4 for this variant.

## 7. Comparison: baseline, Variant 1, Variant 2 -- your task

Train all three models on **at least 2 random seeds** (e.g. `seed=0` and
`seed=1`) and compile the results into a table/printout: validation and
test accuracy, parameter count, measured training and inference time. You
use the test set here **for the first time** in this lab (only now, after
selecting a model based on validation -- just as in module 05).

## 8. Quality evaluation: confusion matrix and error analysis -- your task

For your best (by validation) variant: draw a confusion matrix on the test
set (a provided function is waiting for the data) and print 6-8 misclassified
examples with their true and predicted label.

**Task (Markdown, 3-5 sentences):** which classes are most often confused
with each other? Look at the misclassified examples (do the handwritten
characters visually resemble each other?) -- do the model's errors look
"sensible" (similar-looking characters) or random? Do not guess in advance --
describe what you actually see in your confusion matrix.

## 9. Conclusions -- your task

Answer (Markdown, justified with actual numbers from sections 7-8):

1. Which CNN variant won on validation? Is it the same one that won on the
   test set? If not -- what does that say about the reliability of a single
   comparison?
2. Did the variant with more parameters also train faster, or slower? Is
   what you observed consistent with the intuition "fewer parameters =
   fewer computations," or does it contradict it?
3. Does your best model's higher training accuracy mean it generalizes
   well? What do you base that judgment on (refer to specific numbers from
   section 7)?
4. What is at least one limitation of this experiment (subset size, number
   of epochs, specific random seeds) that prevents you from generalizing
   the conclusions to "Variant X is always better"?

## Time budget (approximate, not verified in class)

| Block | Time |
|---|---|
| Sections 0-1 (setup, data exploration) | 10 min |
| Section 2 (split + leak-free normalization) | 15 min |
| Section 3 (baseline, provided, orientation) | 5 min |
| Section 4 (Variant 1 design + shape verification) | 30 min |
| Section 5 (training Variant 1) | 5 min |
| Section 6 (Variant 2 design + hypothesis) | 20 min |
| Section 7 (comparison, >=2 seeds) | 10 min |
| Section 8 (confusion matrix + error analysis) | 15 min |
| Section 9 (conclusions) | 10 min |

Sections 0-8 are the mandatory core. Training each variant on the reduced
data subset takes a few seconds on CPU -- the time budget should go into
design and analysis, not waiting for training.

**Extension (optional, if time remains):** a third architecture variant;
per-class precision/recall analysis (not just overall accuracy); repeating
the comparison on a larger data subset or with more epochs, to check
whether the conclusions hold.

## Grading criteria

- The normalization in section 2 computes statistics only from `X_train`
  and applies them (without recomputing) to `X_val`/`X_test`; the assertion
  passes.
- Both CNN variants accept `(N, 1, 28, 28)` and return `(N, 10)`; the shape
  verification in sections 4 and 6 is present and correct.
- Variant 2 differs from Variant 1 in a specific, named way -- it is not a
  copy with a random single-digit change and no justification.
- The comparison in section 7 covers >=2 random seeds and reports accuracy
  **together with** parameter count and time, not accuracy alone.
- The test set appears only in section 7, after a selection based on
  validation.
- The conclusions in section 9 refer to specific numbers from your own run,
  not to generalities or unverified assumptions from the lecture.

A full reference solution exists locally with the instructor
(`decisions/solutions/06_cnn/`) and is not part of this repository.
