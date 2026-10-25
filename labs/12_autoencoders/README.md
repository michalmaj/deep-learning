Language: **English (default)** | [Polski](README.pl.md)

# Lab 12. Anomaly detection with an autoencoder

This lab takes about 120 minutes and uses the concepts introduced in
[`lecture.en.md`](../../lectures/12_autoencoders/lecture.en.md). Reconstruction
is used for anomaly detection rather than image generation. The analysis covers
both the method's performance and the cases where it fails.

The data is Fashion-MNIST under the MIT License, registered in
[`data/registry.yml`](../../data/registry.yml). *Sneaker* (class 7) is the
normal class, and the other nine classes form the anomalies. This resembles a
setting where normality is narrowly defined and anything outside that group is
an anomaly.

This is not ten-class classification. During training, the autoencoder receives
neither labels nor anomaly examples. It learns only to reconstruct images from
the normal class.

## 0. Setup

The provided section contains imports, device selection, a random seed, and
Fashion-MNIST loading through `torchvision.datasets.FashionMNIST`.

## 1. Data and split

Prepare disjoint subsets:

1. **`ae_train`** contains only Sneaker images from the original training set.
   This is the only subset used to train the autoencoder.
2. **`val_normal`** contains different Sneaker images from the original
   training set. `torch.randperm` and a split of the class 7 indices can be used.
3. **`val_anomaly`** contains a sample of the remaining classes from the
   original training set, preferably with the same count per class.
4. **`test_normal`** and **`test_anomaly`** come from the original test set.
   The first contains class 7, and the second contains a sample of other classes.

The `test_*` subsets are reserved for one final evaluation. They must not affect
architecture, hyperparameters, or the detection threshold. The notebook should
print every subset size and programmatically confirm that their indices do not
overlap.

## 2. Autoencoder

Implement an `AE(nn.Module)` class with an MLP encoder, a bottleneck, and a
decoder back to a $28\times28$ image. The training function uses MSE and only
`ae_train`. The main model for sections 3-7 has $d_e=16$.

Training takes from a fraction of a second to a few seconds on a CPU. Most of
the lab time is intended for data preparation and result analysis.

## 3. Reconstruction error

The `per_image_error(model, X)` function should return a separate error for
each image:

```python
((x_hat - x) ** 2).flatten(1).mean(dim=1)
```

Compute the error for `val_normal` and `val_anomaly`. Plot both distributions on
one histogram or density chart. Also report the mean, standard deviation, and
median of each distribution.

Partial overlap is expected. The method's usefulness depends on separation at a
chosen threshold and on the examples found in the overlap region.

## 4. Threshold selection

The threshold must come **only** from the `val_normal` distribution. One option
is the mean plus $k$ standard deviations, provided that the choice of $k$ is
justified. `val_anomaly` must not influence the specific threshold value.

This procedure is not fully unsupervised. Labels are used to create the normal
set and to make general validation decisions. The threshold itself uses only
normal examples, which matches a setting with no anomalies available during
system configuration.

## 5. Test metrics

First verify the metric implementation on a known example:

```text
scores = [0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.6, 0.7, 0.8, 1.0]
labels = [0,   0,   0,   0,   0,   1,   1,   1,   1,   1]
threshold = 0.45
```

Calculate TP, FP, TN, FN, precision, recall, and F1 by hand, then confirm the
result with code. The expected values are:

- TP = 5, FP = 1, TN = 4, FN = 0,
- precision $\approx0.833$,
- recall $=1.0$,
- F1 $\approx0.909$.

The repository does not include `scikit-learn`, so ROC-AUC and PR-AUC must be
implemented directly.

- **ROC-AUC** can be calculated as
  $P(\text{score}_{\text{anomaly}}>\text{score}_{\text{normal}})$ over
  anomaly-normal pairs. Two loops are acceptable for the small verification.
- **PR-AUC**, represented here by average precision, requires sorting scores in
  descending order and averaging precision at positions occupied by true
  positives.

On the test set, report precision, recall, F1, a $2\times2$ confusion matrix,
ROC-AUC, and PR-AUC. AUC metrics assess ranking independently of a threshold.
Precision, recall, and F1 describe behavior at the threshold selected in
section 4. Both groups are needed.

## 6. Baseline and error analysis

The training-free baseline uses distance from the mean normal image:

```python
mean_normal_image = ae_train.mean(dim=0)
((x - mean_normal_image) ** 2).mean()
```

Calculate the same test metrics for this baseline and compare them with the AE.
Weak autoencoder performance relative to a simple reference is useful evidence.
It may mean that the classes differ mainly through simple image properties that
do not require a learned representation.

The error analysis must include two concrete cases:

1. an anomaly with low reconstruction error, or a false negative, together with
   its class and a visual comment,
2. a Sneaker image with high error, or a false positive, also displayed and
   discussed.

An autoencoder may reconstruct an anomaly that resembles the normal class, such
as another kind of footwear. This is a limitation of reconstruction error, not
automatically an implementation bug.

## 7. Diagnosing data leakage

The provided code searches thresholds and selects the highest F1 on test-set
labels. It works technically but violates evaluation methodology.

Complete the following steps:

1. run the search on `test_normal` and `test_anomaly`,
2. compare its threshold with the one based on `val_normal`,
3. compare the leaked F1 with the honest test result,
4. explain why using the test set for a decision is wrong regardless of the
   size of the difference in this run,
5. identify the correction already applied in section 4.

A small difference does not remove the problem. The same leakage on another
dataset or model could strongly inflate the result. The test set must remain
independent of every modeling decision.

## 8. Conclusions

Answers should refer to numbers and examples from sections 3-7:

1. What precision, recall, F1, and ROC-AUC did the autoencoder achieve?
2. How does it compare with the baseline, and what does this indicate about the
   source of class separation?
3. Which specific false negative or false positive best demonstrates the
   method's limitation?
4. What caused the data leakage, and what general rule follows from it?
5. What is still needed before practical deployment, including acceptable FP
   and FN costs?

## Optional extension

Choose one of two tasks:

- repeat sections 2-5 for $d_e\in\{4,32\}$ and compare how the bottleneck
  affects class separation,
- build an analogous VAE and use reconstruction error or $-$ELBO as the anomaly
  score, without assuming that it must outperform the AE.

## Time budget

| Block | Time |
|---|---:|
| Setup | 5 min |
| Data and split | 15 min |
| AE architecture and training | 15 min |
| Reconstruction error and distributions | 10 min |
| Validation threshold | 10 min |
| Metrics | 15 min |
| Error analysis and baseline | 15 min |
| Leakage diagnosis | 20 min |
| Conclusions | 10 min |

The core takes 115 minutes, leaving a 5-minute buffer. The extension is not
included in this limit.

## Grading criteria

- Subset disjointness is confirmed programmatically.
- The autoencoder is trained only on normal data.
- The threshold depends only on `val_normal`.
- Metrics pass the known-example check before use on the dataset.
- The baseline comparison is completed regardless of the result.
- The analysis displays and discusses a specific false negative and false
  positive.
- The diagnosis identifies the leakage and explains its consequences.
- Conclusions use evidence from the actual run.

The complete reference solution is stored locally by the instructor in
`decisions/solutions/12_autoencoders/` and is not part of this repository.
