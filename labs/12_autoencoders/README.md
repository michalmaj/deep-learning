Language: **English (default)** | [Polski](README.pl.md)

# Lab 12: anomaly detection with an autoencoder, and analyzing reconstruction's limits

Time: about 120 minutes. Builds on
[`lectures/12_autoencoders/lecture.en.md`](../../lectures/12_autoencoders/lecture.en.md)
(AE, VAE, latent space), but this is **a different problem** than the
lecture's experiment: not generating new images, but **anomaly
detection** via reconstruction error -- with an emphasis on **when this
method fails**, not only on when it works.

**Data**: Fashion-MNIST (MIT License -- see
[`data/registry.yml`](../../data/registry.yml)). **Normal class**:
*Sneaker* (class 7) -- chosen because it is a coherent, unambiguous visual
category (a specific type of footwear). **Anomalies**: the remaining 9
Fashion-MNIST classes (T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Bag, Ankle boot) -- treated as "anything that is not a
sneaker," pedagogically justified: in practical applications (e.g.
quality control on a production line), "anomaly" often means "something
outside a narrowly defined normal category," not one specific, known-in-
advance defect class.

❌ "This is a 10-class classification task, just renamed."

✅ **No** -- the model (the autoencoder) **never** sees class labels
during training, and **never** sees **any** anomaly example during
training. We train exclusively on images of the normal class, with no
labels -- the very fact of "detecting anomalies" must come from how well
the model reconstructs what it learned vs. what it never saw.

## 0. Setup (provided)

Library imports, device, random seed, loading Fashion-MNIST
(`torchvision.datasets.FashionMNIST`).

## 1. Data and split -- your task

This is the **most important** methodological step in this lab -- a
mistake here undermines everything that follows.

Build **four** disjoint subsets, with **no** information leakage between
them:

1. **`ae_train`** -- images of **only** the normal class (Sneaker), from
   the original Fashion-MNIST training set. The autoencoder sees **only
   this**.
2. **`val_normal`** -- **other** images of the normal class, from the
   same original training set (disjoint from `ae_train` -- split class
   7's training set into two parts, e.g. `torch.randperm` + slicing).
3. **`val_anomaly`** -- a sample of images from **other** classes, also
   from the original Fashion-MNIST training set (e.g. an equal number
   from each of the 9 classes).
4. **`test_normal`**/**`test_anomaly`** -- from the **original
   Fashion-MNIST test set** (separate, never touched until this point),
   respectively: all class-7 images, and a sample of images from other
   classes.

**A rule that must not be broken at any later step:** `test_*` **must
not** influence any decision (architecture, hyperparameters, detection
threshold) -- we use it **only** for the final evaluation, once.

❌ "I can later 'fine-tune' or adjust the threshold based on the results
on the test set, since I already have it anyway."

✅ This is **exactly** the methodological bug we investigate in the
diagnostic task (section 7). If the test set influences any decision, the
measured result stops being a trustworthy estimate of performance on new
data.

**Task (code):** print the sizes of all four/five subsets and **verify
programmatically** (not "by eye") that no two of them overlap (e.g. by
comparing indices, not just sizes).

## 2. Autoencoder -- architecture and training -- your task

Build a small AE (MLP, following the lecture's pattern): an `encoder` to
a latent dimension $d_e$, a `decoder` back to $28\times 28$, MSE loss.

**Task:** implement the `AE(nn.Module)` class and a training function.
Train it **only** on `ae_train` (never on `val_*` or `test_*`). Start
with $d_e=16$ -- this is your **main** model for sections 3-7.

**A note on cost**: this training is **very** fast on CPU (a fraction of
a second to a few seconds, depending on the number of epochs/dataset
size) -- this lab's time budget should go into **analysis**, not into
waiting for training.

## 3. Reconstruction error and the normal-vs-anomaly distributions -- your task

Implement `per_image_error(model, X)` -- the reconstruction error **per
image** (not aggregated over the whole set):
`((x_hat - x)**2).flatten(1).mean(dim=1)`.

Compute this error for `val_normal` and `val_anomaly`. **Visualize** both
distributions (a histogram or overlapping density plots) on one plot.
Compute the mean/standard deviation/median for each.

❌ "If the two distributions overlap at all, the method is useless."

✅ Partial overlap is **normal** and expected -- the question is not "do
the distributions overlap," but "how well do they **separate** at a
**specific** threshold choice" (section 4) and "what happens in the
overlap region" (section 6, error analysis).

## 4. Choosing a threshold -- your task

Set the detection threshold **only** based on the `val_normal` error
distribution (e.g. mean + $k\times$ standard deviation, for a $k$ of your
choice -- justify the choice). **Do not** use `val_anomaly` to choose the
threshold -- this is explicitly a **validation-supervised** procedure
(you have labels in validation and decide the architecture/
hyperparameters in general based on them), but choosing the **specific
threshold value** based only on the normal class's distribution is closer
to the practical scenario where you have no anomalies at all while
configuring the system.

❌ "This is fully unsupervised anomaly detection."

✅ **Not true, name it precisely.** You have access to labels throughout
the data-preparation process (you know which images are "normal") -- this
is a **validation-supervised** procedure, not fully unsupervised. Truly
unsupervised anomaly detection (with no labels at all, even for "normal"
data) is a different, harder task, out of scope for this lab.

## 5. Metrics on the test set -- your task

**First, on a small example with a known result** (before using the real
data): for `scores = [0.1,0.2,0.3,0.4,0.9, 0.5,0.6,0.7,0.8,1.0]`,
`labels = [0,0,0,0,0, 1,1,1,1,1]` (the first five normal, the second five
anomalies), threshold $=0.45$ -- **compute by hand** (on paper or in a
comment) TP/FP/TN/FN, precision/recall/F1, **then** verify with your code
that the result matches (expected: TP=5, FP=1, TN=4, FN=0,
precision$\approx 0.833$, recall$=1.0$, F1$\approx 0.909$).

**Note**: `scikit-learn` is **not** installed in this repository --
implement ROC-AUC and PR-AUC (*average precision*) by hand:

- **ROC-AUC** via pairwise comparison (ranking): $P(\text{score}_{\text{anomaly}}
  > \text{score}_{\text{normal}})$ over all pairs (you may use ranking,
  it need not be $O(n^2)$ for large sets -- but for **verification** on
  the small example, $O(n^2)$ via two loops is fine and helps check the
  correctness of the faster version).
- **PR-AUC** (*average precision*): sort examples by score in decreasing
  order, for each **true positive** (anomaly) compute precision at that
  point in the ranking, average those precisions.

Compute, **on the real test data** (section 1): precision, recall, F1, a
2x2 confusion matrix, ROC-AUC, PR-AUC.

❌ "ROC-AUC/PR-AUC is the ultimate quality measure for an anomaly-
detection model -- I don't need anything else."

✅ ROC-AUC/PR-AUC evaluate the **ranking** (do anomalies generally have a
higher error than normal examples, regardless of a specific threshold) --
but the **decision** "is THIS specific image an anomaly" requires a
**specific threshold** (section 4). A high AUC does not tell you how well
you will do at the **chosen** threshold -- hence you need **both** kinds
of metrics, not one.

## 6. Error analysis and a simple baseline -- your task

**A baseline with no learning**: compute the error as the distance to the
**mean image of the normal class** (`ae_train.mean(dim=0)`), with no
training at all -- `((x - mean_normal_image)**2).mean()`. Compute the
same metrics (section 5) for this baseline on the test set.

**Compare** AE against this trivial baseline.

❌ "A trained autoencoder must radically beat a trivial baseline, because
it is 'smarter.'"

✅ **Not necessarily** -- check this **empirically on your data**, rather
than assuming it. If the autoencoder does not clearly beat the trivial
baseline, that is an **important, valuable conclusion**, not a failed
experiment: it means the normal/anomaly separation in this specific task
largely comes from simple pixel differences between classes (a sneaker's
shape vs. other clothing), not from something subtle the autoencoder
alone learned. A critical evaluation of the method **always** requires
this comparison, not only when we expect it to "work out."

**Error analysis (required):** find **at least one specific** case of:
1. An anomaly (an image outside the normal class) with a **low**
   reconstruction error (a false negative) -- check which class it comes
   from, and whether it makes visual sense (e.g. does it resemble a
   sneaker).
2. A normal example (a sneaker) with a **high** reconstruction error (a
   false positive) -- check visually whether it is unusual within its own
   class.

❌ "An autoencoder that generally separates normal/anomaly well must
reconstruct **every** anomaly example poorly."

✅ **Not true in general** -- an anomaly visually/structurally similar to
the normal class (e.g. a different type of footwear, if the normal class
is a specific type of footwear) can be reconstructed **quite well**,
despite formally being an anomaly. This is **not** an implementation bug
-- it is a real limitation of the method, which you must **demonstrate
with a specific example**, not just cite from the lecture.

## 7. Diagnostic task: data leakage in threshold selection

The following code (in the notebook) **looks** reasonable -- it searches
many threshold values and picks the one giving the **best F1** -- but it
has a bug: the search and selection happen **using the test set's
labels**.

**Task:**
1. Run the provided code: search a grid of thresholds, for each compute F1
   **on `test_normal`/`test_anomaly`** (with their true labels), choose
   the threshold with the highest F1.
2. Compare the threshold chosen this way with **your** threshold from
   section 4 (chosen only from `val_normal`). Are they identical?
3. Compare the reported F1: the "leaky" one (chosen and evaluated on the
   same test set) vs. your honest one (the threshold from section 4, F1
   measured on the test set). **Even if the difference is small** --
   explain **why the size of the difference alone is not what makes this
   procedure wrong**.
4. Write down why this procedure is methodologically wrong,
   **regardless** of whether the difference turned out large or small in
   your specific run.
5. Propose a fix (exactly what you did in section 4) and verify it is
   consistent with the rule from section 1 (the threshold never sees test
   labels).

❌ "If the data leakage did not change the result significantly, it was
not a real bug."

✅ **This is not true.** A methodological bug (using test labels to make a
modeling decision) is a bug **regardless** of how large an effect it had
in a specific run -- with a different dataset, a different model, or a
smaller/less separated test set, the same bug could give a dramatically
inflated result. You cannot know this in advance by looking at just one
run -- hence the rule "never touch the test set when choosing a
threshold" must be **absolute**, not dependent on whether "this time it
happened not to hurt."

## 8. Conclusions -- your task

Answer, justified with actual numbers/examples from sections 3-7:

1. How well (specific numbers: precision/recall/F1/ROC-AUC) does your
   autoencoder separate the normal class from anomalies on the test set?
2. How does it compare to the trivial baseline (section 6)? What does
   that say about **where** the discriminative ability in this task comes
   from?
3. What is your most convincing **specific example** of the method's
   limitation (a false negative or false positive from section 6)? Does
   it make visual sense?
4. What bug did the data leakage in section 7 introduce, and what is the
   **general** rule (not just for this dataset) you draw from it?
5. Would you consider this system ready for practical use? What would it
   still need (e.g. what FP/FN error cost is acceptable for what
   application)?

## Extension (optional, if time remains)

Choose **one** of the following:

- **Effect of latent dimension**: repeat sections 2-5 for $d_e \in
  \{4, 32\}$ (besides the main $d_e=16$) and compare the metrics. Does a
  larger/smaller bottleneck change the normal/anomaly separation -- and is
  this effect what you intuitively expect?
- **VAE as an alternative**: implement an analogous VAE (following the
  lecture's pattern), use the reconstruction error (or $-$ELBO) as the
  anomaly signal. Compare with AE **without** assuming in advance which
  one "must" do better (see the lecture: VAE is not automatically a
  better anomaly detector).

## Time budget (realistic, verified against actual training cost)

| Block | Time |
|---|---|
| Section 0 (setup) | 5 min |
| Section 1 (data and split, no leakage) | 15 min |
| Section 2 (AE: architecture and training) | 15 min |
| Section 3 (reconstruction error, distributions) | 10 min |
| Section 4 (choosing a validation threshold) | 10 min |
| Section 5 (metrics: known example + test) | 15 min |
| Section 6 (error analysis + baseline) | 15 min |
| Section 7 (diagnostic task: leakage) | 20 min |
| Section 8 (conclusions) | 10 min |

Total: **115 min**, with a 5-minute buffer. Training the AE on this
dataset takes a fraction of a second to a few seconds on CPU -- the time
budget should go into designing the data split, error analysis, and the
diagnostic task, not into waiting for the model. The extension (above) is
**optional** and is **not** counted in these 115 minutes.

## Grading criteria

- The data split (section 1) is verified programmatically to be
  disjoint; the autoencoder is trained **only** on normal data.
- The detection threshold (section 4) is chosen **only** based on
  `val_normal`, never based on `val_anomaly` or any `test_*`.
- The metrics (section 5) are verified on the given known-result
  example **before** being used on the real data.
- The comparison with the trivial baseline (section 6) is **actually
  carried out and described** -- regardless of whether AE did better,
  similarly, or worse.
- The error analysis (section 6) points to a **specific** case (not a
  generality) of a false negative or false positive, with a sensible
  visual comment.
- The diagnostic task (section 7) correctly identifies the data leakage
  and explains why the bug is real regardless of the size of the
  observed difference.
- The conclusions refer to specific numbers/examples from your own run,
  not to generalities from the lecture.

A full reference solution exists locally with the instructor
(`decisions/solutions/12_autoencoders/`) and is not part of this
repository.
