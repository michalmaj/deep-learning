Language: **English (default)** | [Polski](lecture.md)

# Lecture 14 — How to evaluate and interpret models we want to trust

## Lecture goals

1. Move from a single number (accuracy) to a systematic audit of a
   classification model's errors.
2. Understand precision, recall, F1, macro/micro-averaging, and the role
   of class imbalance — and be able to compute them by hand and verify
   them programmatically.
3. Understand the difference between a model's confidence and the actual
   reliability of a prediction — and what calibration measures (and does
   not measure).
4. Be able to derive and independently implement Grad-CAM, with full
   control over dimensions and gradient correctness.
5. Understand that Grad-CAM (and attribution in general) shows
   sensitivity, not causal proof — and know how to check this (occlusion,
   sanity checks).
6. Prepare a systematic error-analysis report that leads to a concrete
   engineering recommendation.

## Why this matters

Since module 6 we have been training models and measuring accuracy. But
accuracy is **one** number, aggregated over the whole set — it hides
**which** classes/examples the model fails on, **how** confident it is in
its mistakes, and **whether** its "view" of the data (attribution) makes
sense. This lecture does not introduce a new architecture — it introduces
a **discipline**: how to ask the model questions that accuracy doesn't
answer, and how not to fool ourselves with our own interpretability tools.

Three lessons from earlier modules are directly useful here: a model with
more parameters need not generalize better (module 5); a generative model
can have a good metric even though a simple baseline is just as effective
(module 12 — AE vs. a trivial baseline on anomaly detection); a single
result does not prove a general claim (module 13 — one MMD run doesn't
decide, one heatmap doesn't decide).

## Evaluation as part of the engineering process

**Train / validation / test — the role of each split:**

- **Train**: learning the parameters $\theta$.
- **Validation**: choosing a configuration (hyperparameters, decision
  threshold, early stopping) — **never** the final evaluation.
- **Test**: **one**, final evaluation of the already-chosen model — never
  used for any configuration decision.

❌ "I can choose the decision threshold (or a hyperparameter, or which
examples to analyze) on the test set, since I want to see how well it
performs anyway."

✅ This is **data leakage of the evaluation protocol** — using the test
set for **any** decision before the final report invalidates its role as
an independent estimate. We will see this literally in the lab (module
12's diagnostic task taught this for the anomaly threshold — here the same
principle applies to the decision threshold, the choice of examples for
error analysis, and the choice of interpretation method).

## Accuracy, precision, recall, F1 — the mathematics

Notation: $C$ = number of classes (reused from module 07 — "the number of
classes in your task"; a **different** role than $C$ = number of image
channels $(C,H,W)$, module 06 — never in the same context, but an
explicit distinction: here $C$ is the size of the confusion matrix, not a
tensor dimension).

**Worked example** (3 classes, $N=20$ examples, deliberately unbalanced:
class 0 — 10, class 1 — 8, class 2 — 2):

Confusion matrix (rows = truth, columns = prediction):

| | pred 0 | pred 1 | pred 2 |
|---|---|---|---|
| **true 0** | 9 | 1 | 0 |
| **true 1** | 0 | 8 | 0 |
| **true 2** | 2 | 0 | 0 |

$$\text{accuracy} = \frac{\sum_c \text{TP}_c}{N} = \frac{9+8+0}{20} = 0.85$$

For class $c$: $\text{TP}_c$ = the diagonal; $\text{FP}_c$ = the sum of
column $c$ minus $\text{TP}_c$; $\text{FN}_c$ = the sum of row $c$ minus
$\text{TP}_c$.

$$\text{precision}_c = \frac{\text{TP}_c}{\text{TP}_c+\text{FP}_c}, \qquad \text{recall}_c = \frac{\text{TP}_c}{\text{TP}_c+\text{FN}_c}, \qquad F1_c = \frac{2\cdot\text{precision}_c\cdot\text{recall}_c}{\text{precision}_c+\text{recall}_c}$$

| class | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0 | 9 | 2 | 1 | 0.818 | 0.900 | 0.857 |
| 1 | 8 | 1 | 0 | 0.889 | 1.000 | 0.941 |
| 2 | 0 | 0 | 2 | **NaN** (0/0) | **0.000** | 0.000 |

Verified programmatically in the notebook — exactly these numbers.

❌ "Very high accuracy means the model works well on every class."

✅ **Not true — this example shows it directly.** Accuracy $=0.85$ (high),
but class 2 (the minority class) has recall $=0$ — the model **never**
correctly recognizes this class, and high accuracy hides this because
class 2 is only $2/20=10\%$ of the examples.

**Macro vs. micro averaging:**

$$\text{macro-F1} = \frac{1}{C}\sum_{c=1}^C F1_c = \frac{0.857+0.941+0.000}{3} = 0.599$$

Micro-F1 (for single-label multi-class classification) is computed from
the **summed** TP/FP/FN over all classes — which for this case (every
example has exactly one prediction and one true label) **mathematically
reduces to accuracy**:

$$\text{micro-F1} = \frac{\sum_c \text{TP}_c}{\sum_c \text{TP}_c + \sum_c \text{FP}_c} = \frac{\sum_c \text{TP}_c}{N} = \text{accuracy} = 0.85$$

(because $\sum_c \text{FP}_c = \sum_c \text{FN}_c$ for single-label
multi-class — every error is an FP for one class and an FN for another,
so the sums cancel out).

❌ "Macro-F1 and micro-F1 are two equivalent ways of writing the same
number."

✅ **Not true — this example shows a $0.60$ vs. $0.85$ difference.**
Macro-F1 **does not weight** classes by size — every class contributes
equally, regardless of whether it has 2 or 1000 examples. Micro-F1 (=
accuracy here) **implicitly weights** by size — large classes dominate the
result. For class imbalance, macro-F1 **reveals** the minority class's
problem, which accuracy/micro-F1 **hides**. Weighted-F1 (weights each
class's F1 by its support) is a third, intermediate option — $0.805$ for
this example.

## Logits, softmax, argmax, ranking, threshold — a distinction

The chain (extending module 02/08/11): $z = f_\theta(x) \in \mathbb R^C$
(logits) $\to \hat y = \text{softmax}(z) \in [0,1]^C$ (probabilities,
summing to 1) $\to \hat c = \arg\max_c \hat y_c$ (the predicted label).
These are **three different objects**: logits are not probabilities (they
can be negative, they don't sum to 1); argmax is a **decision**, not a
**measure of confidence**; ranking examples (e.g. by $\max_c \hat y_c$)
is yet another, separate operation, used for ROC-AUC/PR-AUC.

❌ "ROC-AUC and PR-AUC apply to a multi-class classifier directly, the
same way as to a binary one, with no modification."

✅ **Not true.** ROC-AUC/PR-AUC are defined for a **binary** decision
(positive/negative) as a function of a threshold. For $C>2$ classes,
they are applied **per class**, within a *one-vs-rest* scheme (class $c$
vs. "all others"), not as a single, natively multi-class curve. Module
12's lab already built the full ROC-AUC/PR-AUC machinery (by hand, without
`scikit-learn`) for a **binary** anomaly-detection task — the same code
would work here, applied $C$ times (once per class); we do not develop
this full discussion here (outside this lecture's core).

**Decision threshold and the cost of errors.** For a binary decision (e.g.
"is this class X"), choosing a threshold changes the **balance** between
false positives and false negatives — as in module 12's lab (the
threshold chosen **only** from validation data, never test data). The
cost of an FP and an FN **need not be symmetric** — it depends on the
application (an engineering decision, not a statistical one).

## Calibration

❌ "If the softmax for a prediction is $0.95$, then this SPECIFIC
prediction is correct with 95% probability."

✅ **Not true — this is a category error.** Softmax $0.95$ is the output
of **one** forward pass for **one** example — it is not an empirical
frequency or a guarantee. **Calibration** is a property of a
**population** of predictions, not of one prediction: a model is
calibrated if, **among all** predictions with confidence $\approx p$,
**indeed** about $p$ fraction are correct.

**Confidence** = $\max_c \hat y_c$ (reusing the already-established
$\hat y= \text{softmax}(z)$, module 08 — here: the maximum entry of the
vector). **Reliability diagram**: we group predictions into confidence
bins $B_m$ (**notation note**: this collides with $B_t$ = mini-batch,
module 04 — a different index (m=confidence bin vs. t=training step), a
different concept, too standard in the calibration literature to
change), for each bin we compute the **mean accuracy** and **mean
confidence**; a calibrated model has points close to the diagonal
(acc≈conf).

**Expected Calibration Error:**

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \big|\text{acc}(B_m) - \text{conf}(B_m)\big|$$

**Worked example** (10 predictions, 2 bins): 4 predictions with
confidence$=0.95$ (3 correct, 1 wrong → acc$=0.75$); 6 predictions with
confidence$=0.55$ (2 correct, 4 wrong → acc$=0.3333$).

$$\text{ECE} = \frac{4}{10}|0.75-0.95| + \frac{6}{10}|0.3333-0.55| = 0.08+0.13 = 0.21$$

Verified programmatically — exactly $0.21$.

❌ "ECE is an objective, unambiguous measure — it doesn't depend on how
you compute it."

✅ **Not true — measured, not just claimed.** On a synthetic example with
a continuous confidence distribution: ECE is stable ($\approx 0.157$) for
bin counts $5,10,15,20$, but **jumps** to $0.175$ for $50$ bins (sparse
bins become noisy — few examples per bin, unstable mean). The same
distribution, **different sample size**: ECE $0.193\to0.164\to0.118\to0.157$
for $n=20,50,100,200$ — a clear dependence on sample size, not one stable
number.

❌ "A model can have high accuracy and automatically be well calibrated
(or: it must be poorly calibrated, since neural networks are known for
overconfidence)."

✅ **Neither generalization is safe — you have to measure.** The model
from this lecture (Experiment B) is **well** calibrated (ECE$\approx 0.01$)
— we do not force a narrative of overconfidence here just because that's
the more common result in the literature. **But** even good
**population-level** calibration does not guarantee that **every**
individual prediction is reliable — the same model had wrong predictions
with confidence up to $0.9996$ (see "What we observe"). These are **two
different levels** of evaluation (population vs. example), not one
conclusion.

(Temperature scaling — a technique for calibrating logits **after**
training, dividing $z$ by a scalar $T_{\text{cal}}$ before softmax, chosen
on validation data — **notation note**: this is a **different**
temperature than $\tau$ from module 11 (the **sampling** temperature,
which changes the randomness of text generation) — here, the **calibration**
temperature, which changes the sharpness of softmax for **assessing
reliability**, not for generation. We do not develop a full implementation
in this lecture's core — an extension, if time allows.)

## Grad-CAM: the mathematics

**The most important interpretability part of this lecture.** Context: a
CNN with a chosen convolutional layer, for which we want to understand
which regions of the input the prediction for a specific class "reacts"
to.

**Notation.** $A^k \in \mathbb R^{H\times W}$ — the activation map of
**channel $k$** of the chosen layer (**notation note**: written
**without** parentheses, unlike $A^{(l)}$, module 03 — there, parentheses
denote a **generic layer index** for a batched, averaged representation;
here $A^k$ is a **specific** map of **one, chosen** layer, indexed by
**channel**, not by layer — a different mathematical object). $z_c$ — the
logit of class $c$ (reusing $z$=logits, module 02+, with a class index —
**not** $y^c$, a new symbol would be unnecessary here, and **not** the
softmax $\hat y_c$ — this is an important distinction, not a stylistic
one (explanation below).

**Why the logit, not the softmax.** Softmax $\hat y_c = \exp(z_c) /
\sum_{c'} \exp(z_{c'})$ normalizes by the sum over **all** classes — the
gradient $\partial \hat y_c/\partial A^k$ therefore carries the influence
of **all** other logits (through the denominator), not just class $c$.
The gradient of the logit $\partial z_c/\partial A^k$ "sees" only class
$c$, without this contamination. Additionally, when softmax is close to
$0$ or $1$ (the model is very confident), its gradient **saturates** close
to zero — Grad-CAM computed on the softmax would lose signal precisely
where the prediction is most unambiguous. For these two reasons
(contamination by other classes, saturation), the original Grad-CAM paper
(Selvaraju et al., 2017) targets the logit $z_c$, not $\hat y_c$.

**Step 1 — gradient of the output with respect to the activation maps:**

$$\frac{\partial z_c}{\partial A^k} \in \mathbb R^{H\times W}$$

**Step 2 — channel weights via global average pooling of the gradient
(global average pooling over the spatial dimensions):**

$$\alpha_k^c = \frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W \frac{\partial z_c}{\partial A^k_{ij}}$$

(**notation note**: this is the **fourth** meaning of $\alpha$ in this
course — attention weights $\alpha_{ij}$, module 10; the VAE
interpolation, bare $\alpha$, module 12; the diffusion schedule $\alpha_t$,
module 13 — here $\alpha_k^c$, a two-level sub+superscript index, not
reused before. Too standard in the Grad-CAM paper to rename.)

**Step 3 — weighted sum of the activation maps + ReLU:**

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right) \in \mathbb R^{H\times W}$$

ReLU **here** (on the weighted sum, not on individual $A^k$) reflects that
we are only interested in regions with a **positive** influence on $z_c$
— a negative influence (a region "against" class $c$) is clipped to zero,
not kept as a negative value.

**Step 4 — rescale to image resolution** (interpolation, usually
bilinear, from the layer's $(H,W)$ to the input image's
$(H_{\text{img}}, W_{\text{img}})$) and normalize to $[0,1]$ (handling the
zero-spread case — below).

### A small worked example

Two channels, a $2\times 2$ map each. $A^0 = \begin{pmatrix}1&2\\3&4\end{pmatrix}$,
$A^1=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ — **hypothetical, hand-picked**
values, to illustrate the formula, not the result of any training.
Let's also take a hypothetical, hand-picked gradient
$\partial z_c/\partial A^k$: for channel 0,
$\begin{pmatrix}-0.2&-0.2\\-0.2&0.2\end{pmatrix}$, for channel 1,
$\begin{pmatrix}0.2&0.4\\0.4&0.2\end{pmatrix}$.

Step 2 (global averaging, $1/(HW)=1/4$):
$\alpha_0^c = (-0.2-0.2-0.2+0.2)/4 = -0.1$,
$\alpha_1^c = (0.2+0.4+0.4+0.2)/4 = 0.3$ — note that $\alpha_k^c$ **can be
negative** (a channel "arguing against" class $c$, not only "for" it).

Weighted sum: $\alpha_0^c A^0 + \alpha_1^c A^1 =
-0.1\begin{pmatrix}1&2\\3&4\end{pmatrix} +
0.3\begin{pmatrix}0&1\\1&0\end{pmatrix} =
\begin{pmatrix}-0.1&0.1\\0.0&-0.4\end{pmatrix}$. After ReLU:
$\begin{pmatrix}0&0.1\\0.0&0\end{pmatrix}$ — only one cell (top-right)
has a positive influence on $z_c$ once both channels are accounted for;
ReLU zeroes out the rest.

This whole example is **fully hand-computable** step by step from the
very definition of $\alpha_k^c$ given above — it does not come from any
actual training run or notebook. Independently of it, the agreement of
the practical, hook-based implementation (forward/backward hooks) with an
independent, manual computation via `retain_grad()` is checked in the
test suite (`tests/test_lecture_14_notebook.py`, not in this notebook, on
a separate, small, randomly-initialized network) — **within numerical
tolerance** (`atol=1e-5`), not as bit-exact identity.

### Three different targets — three different questions

❌ "Grad-CAM shows one, universal map of 'what the model is looking at'."

✅ **Not true — the choice of $c$ changes the question.** Grad-CAM for:

1. the **predicted** class $\hat c=\arg\max_c z_c$ — "what supported
   **this** decision?"
2. the **true** class $y$ (if $y\neq\hat c$) — "what **would have**
   supported the correct answer that the model did not choose?"
3. **any other** class chosen by the analyst — "what would support
   **this specific, hypothetical** decision?"

These are **three different maps for the same image** — using one instead
of another without justification confuses the questions.

### Handling a zero-spread map

If the weighted sum $\sum_k \alpha_k^c A^k$ is **non-positive everywhere**,
ReLU zeroes out the **entire** map — the normalization $(x-\min)/(\max-\min)$
would divide by $\approx 0$. The correct handling: check
$(\max-\min)>\epsilon$ before dividing, otherwise return a zero map
(verified programmatically — not a theoretical edge case, see "What we
observe": this happens **often**, not only in theory).

## Limits of interpretation

- Grad-CAM shows a **specific form of sensitivity** of the prediction to
  the activations of a **chosen** layer — not "what the model thinks" in
  general.
- A heatmap **is not** a segmentation mask — the low resolution of a
  convolutional layer (e.g. $7\times7$ before interpolation to
  $28\times28$ or $224\times224$) means that **no** interpolation
  recovers a pixel-level precision that was never there.
- A high attribution value **does not prove** that the model recognizes a
  semantically correct object — only that this region **correlates** with
  a higher $z_c$ under this **specific** gradient parameterization.
- A map's agreement with human intuition **does not guarantee** the
  explanation's correctness — a map that matches intuition may result
  from an accidental correlation, not from the mechanism that actually
  drives the decision.
- The method **depends** on the architecture, the chosen layer, the
  chosen class, and the way gradients are computed — changing any of
  these can give a different map for the **same** image.
- Large/smoothed regions (typical after interpolating from a small map)
  **do not mean** precise feature localization.

## Occlusion sensitivity: an independent verification by perturbation

Instead of a gradient, we **physically** occlude a fragment of the image
(replacing it with a sensible baseline value, consistent with the
preprocessing — e.g. a normalized "gray" pixel for a model with
ImageNet normalization, not raw zero) and measure the **change** in the
logit $z_c$ of the analyzed class. Result: a heatmap of "how much the
logit dropped when we occluded this region" — a conceptually different
signal than the gradient (it asks "what happens if I remove this," not
"how does the result change locally under an infinitesimal change in
activation").

❌ "Occlusion sensitivity is a method without limitations, because it
doesn't use gradients."

✅ **It has its own, different limitations.** Masking **creates** modified
inputs that may **drastically** deviate from the natural distribution of
the training data (a large, uniform gray square on a satellite image is
something the model never saw during training) — a change in the result
may reflect a reaction to an **unnatural** input, not a pure "importance"
of the occluded region.

## Sanity checks for saliency maps

❌ "An attribution method that looks sensible is, by that fact, reliable."

✅ **This has to be verified, not assumed.** One test: compute Grad-CAM
for a trained model, **then** randomly reset selected weights (e.g. the
final classifier) and compute Grad-CAM **again** for the same image and
the same class. If the method is **sensitive to the model** (as it
should be), the map should **change** — if it doesn't change at all, the
attribution may not reflect the actually learned parameters, only
something structural (e.g. the image's own shape). We do not assume in
advance how much change will occur — we **measure** it (e.g. mean
absolute difference between maps, not correlation — correlation is
undefined when one of the maps is constant/zero, which happens, see
"handling a zero-spread map").

## `model.eval()` vs. `torch.no_grad()` vs. `torch.inference_mode()`

❌ "Grad-CAM can be computed in the same mode as regular gradient-free
inference — `model.eval()` is enough."

✅ **Not true, a measured contrast:**

- `model.eval()` **alone** does not affect gradient tracking — it only
  affects the behavior of `BatchNorm`/`Dropout`. `backward()` works
  normally.
- `torch.no_grad()`: `backward()` **raises** a `RuntimeError` ("element 0
  of tensors does not require grad and does not have a grad_fn") —
  Grad-CAM **requires** an active gradient graph.
- `torch.inference_mode()`: the same error **inside** the context, but
  additionally, tensors created there **cannot** even get
  `requires_grad_(True)` **outside** the context ("Setting
  requires_grad=True on inference tensor outside InferenceMode is not
  allowed") — a stronger, more "contagious" restriction than `no_grad()`.

**A subtler, actually-discovered case**: for a model with a **frozen**
backbone (all weights `requires_grad=False`, only the head trained — as
in the lab), Grad-CAM on a backbone layer **fails** (gradient = `None`)
**even without** `no_grad()`, **if** the input tensor $x$ also lacks
`requires_grad=True` — the autograd graph never forms along frozen
weights when nothing "upstream" requires gradients. Fix: `x =
x.requires_grad_(True)` **on the input** (not on the backbone's weights)
— this doesn't change what we are learning, it only ensures that a
gradient graph actually exists down to the hook.

## What we observe

All numbers come from an actual run of `examples.ipynb` (CPU, this
machine) — Fashion-MNIST, a subset of 8000 train / 2000 test, `SmallCNN`
from module 06 (with a fix: separate `relu1`/`relu2` instances, necessary
for hook-based Grad-CAM to work correctly — module 06 never needed hooks,
so the original version with one shared instance never exposed this
problem there).

- **Experiment A.** Training (10 epochs, Adam, CrossEntropyLoss): test
  accuracy $=0.8745$, macro-F1 $=0.8765$ (close — classes in this subset
  are roughly balanced, unlike the hand-built 3-class example above).
  Per-class F1 ranges $0.70$–$0.97$ — real variability, not one number.
  Most confused pair: **Coat→Pullover** (34 cases), second:
  **T-shirt/top→Shirt** (22) — both are known, visually justified
  confusions for this dataset, not a coincidence.
- **Experiment B.** Mean confidence: correct $=0.9082$, incorrect
  $=0.6517$ — different distributions, as expected. ECE $=0.0099$ — **the
  model is well calibrated**, we state this directly, without forcing an
  overconfidence narrative. Even so: **29** wrong predictions had
  confidence $>0.9$, up to $0.9996$ — good **population-level**
  calibration does not eliminate individual, very confident errors.
- **Experiment C.** Zero-map rate (relu2, 32 channels): **about 65%** of
  predictions gave a Grad-CAM with zero spread (the weighted sum negative
  everywhere) — **uncorrelated** with confidence ($0.854$ vs $0.849$
  confidence, degenerate vs. non-degenerate). Selected, non-degenerate
  examples: a correct prediction (idx=2, Trouser→Trouser, conf=1.0000); an
  error (idx=12, Sneaker→Bag, conf=0.4602); a misleading high-confidence
  error (idx=23, Ankle boot→Sandal, conf=0.9996 — **this is** the same
  example as the "max confidence among wrong predictions" from Experiment
  B).
- **Experiment D.** Occlusion sensitivity (4×4 patch, stride 4,
  baseline=0, consistent with the $[0,1]$ range of `ToTensor()`): for a
  correct example (idx=2, Trouser), base_logit$=9.644$, the range of logit
  change after occluding patches: $[-0.975,\ 5.369]$ — regions with a
  large logit drop **partially** overlap with the Grad-CAM regions, not
  identically (different questions, see above). Time: $0.007$s per image
  (49 patches) — negligible.
- **Experiment E.** Sanity check (a random reset of the `fc` weights, the
  same image and class): **mean absolute difference** between the CAM
  before/after $=0.1384$ — a real, measured change, even though the conv
  features are unchanged (only the final classification is randomly
  changed). Correlation as a metric **would be** undefined (`NaN`) if we
  happened to hit one of the degenerate (zero) CAMs from the ~65% of
  cases identified in Experiment C — that's why mean absolute difference
  (robust to constant/zero maps) is the **better** default choice here,
  verified by the actual frequency of degeneracy, not just a theoretical
  possibility.

## Common misconceptions

(A summary — each has a full ❌/✅ box above)

1. High accuracy does not mean good performance on every class.
2. High confidence is not a guarantee of correctness for a single
   prediction.
3. A model can be poorly calibrated despite high accuracy — and
   conversely, a well-calibrated model still has some confident, wrong
   predictions.
4. ECE depends on the number of bins and the sample size — measured, not
   theoretical.
5. A confusion matrix does not by itself explain the causes of errors.
6. Grad-CAM is not a segmentation or ground truth.
7. Grad-CAM is not proof of a causal explanation of a prediction.
8. A nice-looking visualization does not mean an interpretability method
   is reliable — it has to be sanity-checked.
9. Attention weights (module 10) are not a substitute for Grad-CAM and do
   not constitute a universal model explanation — different methods,
   different assumptions.
10. Test results should not be used to tune the model/threshold/method.
11. One example of an error does not prove a systematic model weakness.
12. Robustness to a single transformation does not mean robustness to a
    real distribution shift.
13. A good offline result does not guarantee equally good performance
    after deployment (developed in module 15).
14. Grad-CAM requires an active gradient graph — `model.eval()` alone is
    not enough, and a frozen backbone without `requires_grad` on the
    input fails differently than `torch.no_grad()`, but for the same
    underlying reason.

## Engineering implications

- **Overall quality vs. per-class quality** are two different decisions —
  a model "good overall" may be unacceptable for one, important class.
- **The cost of FP/FN is a business decision**, not a statistical one —
  a threshold chosen "optimally" for one metric may be poorly chosen for
  the real cost of an error.
- **Calibration** has practical importance wherever confidence is used
  for a decision (e.g. "automatically accept only predictions >0.95") — a
  poorly calibrated model makes these decisions on a bad basis.
- **Reviewing difficult examples** (chosen by an explicit, non-cherry-picked
  procedure) is part of the process, not a one-off curiosity.
- **The limitations of interpretability methods must be documented** — "a
  nice heatmap" without a sanity check is insufficient justification for a
  decision.
- **Reproducibility of evaluation reports** — the same data split, the
  same metrics, so comparisons between model versions are fair.
- **The risk of selectively showing good cases** — a report must show a
  representative picture, not just impressive examples.
- **Documenting known model limitations** (model cards, briefly) — part
  of responsible deployment, not an optional extra.
- Evaluation **before** deployment (this module) and monitoring **after**
  deployment (module 15) are related, but **not the same process** — a
  good offline result does not guarantee that monitoring is unnecessary.

## What to remember

- Train/validation/test have strictly different roles — test never
  influences any decision before the final report.
- Accuracy hides per-class weaknesses; macro-F1 reveals them (it doesn't
  weight classes by size), micro-F1 = accuracy for single-label
  multi-class.
- Confidence ≠ the reliability of one prediction; calibration is a
  property of a **population** of predictions (ECE), not a guarantee for
  every example.
- Grad-CAM: $\alpha_k^c$ = GAP of the gradient of $z_c$ w.r.t. $A^k$; the
  map = ReLU(weighted sum); requires an **active gradient graph**, not
  just `eval()`.
- Three different Grad-CAM targets (predicted/true/any class) = three
  different questions.
- Occlusion sensitivity verifies Grad-CAM with a **different** kind of
  signal (perturbation, not gradient) — agreement is not guaranteed.
- Sanity checks (e.g. a random weight reset) check whether the method
  reacts to the model at all, not only to the shape of the input.

## Questions for students

1. You have a 5-class classifier with accuracy $0.92$. What additional
   numbers would you ask to see before calling the model "good"?
2. Why does micro-F1 for single-label multi-class equal accuracy, but not
   for multi-label (more than one true label per example)?
3. You have two models with identical ECE. Does that mean they are "just
   as reliable" for individual predictions? Justify your answer.
4. Explain why Grad-CAM for the predicted class and for the true class
   can point to **different** regions of the image, even for the same,
   misclassified example.
5. You have Grad-CAM and occlusion sensitivity, which **disagree** about
   the most important region of the image. How would you decide what
   this means — and what you **cannot** conclude from it?

## Bibliography

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
  Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*. ICCV. arXiv:1610.02391 (the Grad-CAM
  source paper).
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B.
  (2018). *Sanity Checks for Saliency Maps*. NeurIPS. arXiv:1810.03292
  (weight/label randomization as a sensitivity test for attribution
  methods).
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration
  of Modern Neural Networks*. ICML. arXiv:1706.04599 (ECE, reliability
  diagram, temperature scaling).
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST*.
  arXiv:1708.07747 (see also `data/registry.yml`).
- PyTorch documentation: *Autograd mechanics* (`torch.no_grad`,
  `torch.inference_mode`), `https://pytorch.org/docs/stable/notes/autograd.html`.
