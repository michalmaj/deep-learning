Language: **English (default)** | [Polski](lecture.md)

# Lecture 15 — from a model in a notebook to a system you can observe and maintain

## Lecture goals

1. Be able to save a model as an **artifact** (weights + architecture +
   configuration + metadata) and reproduce it in a separate process with
   **identical** predictions.
2. Understand training-serving skew and know exactly what breaks when
   preprocessing/class order/`eval()` disagree between training and
   inference.
3. Be able to design a minimal API contract (health/ready/predict) and
   correctly validate input.
4. Be able to precisely measure and interpret latency (p50/p95),
   throughput, and memory cost — distinguishing cold from warm, batch
   from a single request.
5. Understand the difference between system monitoring and data/quality
   monitoring — and know what **cannot** be measured without labels.
6. Understand the difference between data drift, label shift, and
   concept drift — and know that detecting drift **is not** proof of a
   quality drop.

## Why this matters

Since module 6 we have been training and evaluating models — but
**offline evaluation** (module 14: accuracy, macro-F1, Grad-CAM) and a
**working system** are two different questions. A model with great
accuracy in a notebook can fail as a system if: another process cannot
reproduce its predictions (training-serving skew); the service is too
slow or unavailable; nobody notices that incoming data has long stopped
looking like the training data. This lecture does **not** introduce a new
network architecture — it introduces the **engineering discipline**
needed for a model to actually work outside a notebook.

**Four different conditions, none guarantees the others:**

1. **The code works** — the service responds, doesn't raise exceptions.
2. **The predictions are correct** — the model actually returns what it
   should.
3. **The service is reliable** — availability, handling bad inputs,
   latency.
4. **Quality is stable over time** — predictions are still accurate as
   the data changes.

A model can work correctly while the API is unavailable. The API can work
correctly while the model is systematically wrong. These are
**independent** axes of evaluation.

### Connections to earlier modules

- **Module 5**: validation and data leakage — here: **temporal** leakage
  (monitoring data "from the future" used for decisions about the past)
  and **configuration** leakage (a drift threshold chosen after seeing
  the result).
- **Module 7**: pretrained weights and their preprocessing — here: the
  **same** preprocessing must work identically in training and in the
  service.
- **Module 11**: inference cost, cold start, latency — here: the same
  measurement discipline, applied to a local HTTP API, not to an LLM.
- **Module 14**: error analysis, confidence, calibration — here: the
  same concepts, applied to **monitoring over time**, not a one-time
  evaluation.

## From a notebook to a model artifact

An **artifact** is more than weights. It must unambiguously specify:

- the architecture (or its explicit configuration, not just the name of a
  class that happened to exist in the process's memory during training);
- the parameters (`state_dict`);
- the **version** of the model;
- the **order and names of the classes** (index $\to$ label — this is
  information separate from the weights themselves);
- the input size and number of channels;
- the preprocessing (the exact steps, not just "normalized");
- the versions of the libraries needed for inference;
- the model's provenance (what data, when, what offline result).

❌ "It's enough to save the entire model object (`torch.save(model)`) to
have a complete artifact."

✅ **We do not prefer this.** Serializing an entire Python object (via
`pickle`) ties the artifact to a **specific** class structure in a
specific environment (module paths, library versions) and is
**dangerous** to load from an untrusted source (reconstructing the object
can execute arbitrary code). We prefer **`state_dict`** (a dictionary of
parameter name → tensor, with no class logic) **plus separate, readable
metadata** (JSON) — and explicitly building the architecture in the
service's code, not from a serialized object.

**Safe loading.** Since PyTorch 2.6, `torch.load(..., weights_only=
True)` is the **default** behavior and restricts the reconstructed types
to a safe list (tensors, basic types) — this prevents arbitrary code
execution when loading an artifact from an untrusted source. Explicitly
passing `weights_only=True` (regardless of version) is good practice, not
just "a default setting that can be ignored."

### Compatibility experiment: a bit-exact artifact

1. We train a small classifier, compute a prediction for a fixed input.
2. We save `state_dict` + metadata (JSON).
3. **A new model instance** (simulating a separate process) — we build
   the architecture **from scratch**, load the weights.
4. We compute a prediction for **the same** input.
5. We compare the logits.

Verified programmatically: `torch.equal(logits_before, logits_after)` —
**exactly** (not `allclose`, not within tolerance — literal bit-for-bit
identity). This is a **deserved** use of the phrase "bit-exact" (unlike
the case found in the modules 08-10 review, where "bit-exact" was
incorrectly used for a result with `atol>0`) — here there is no
nondeterministic operation between saving and reconstructing, so the
identity is literal.

### What can break (brief, measured examples of risk)

❌ "If the weights load without an error, the model has been reproduced
correctly."

✅ **Not true — loading without error is not proof of correctness.**
Measured examples:

- **Forgetting `eval()`.** On our classifier (with a `Dropout` layer in
  the head, as in module 5), `model.train()` vs. `model.eval()`: maximum
  logit difference $=1.81$ — a **large, nontrivial** shift (for this
  specific image, `argmax` **happened** to agree, but the logits are
  clearly different — for a different image/model, the same mechanism
  **could** already change the predicted class, not just its confidence).
  `Dropout` randomly zeroes part of the activations **only** in
  `train()` mode. An **even more dramatic** case: for a frozen ResNet18
  backbone (all weights `requires_grad=False`!, the lab) in `train()`
  mode, `BatchNorm` uses statistics from the **current batch**, not the
  frozen running stats — there, the logit difference
  $=1.04$, `argmax` completely different. In both cases: **frozen
  weights do not mean frozen behavior** for the whole network.
- **Inconsistent class order.** The prediction index (`argmax`) and the
  logit value are **identical** regardless of how the metadata maps the
  index to a class name — if the class list in the metadata has a
  different order than during training, **only the text label** is
  wrong. The model "guessed" correctly (the correct index) — we report
  the wrong name.
- **Inconsistent preprocessing (training-serving skew).** Fully developed
  in the lab — briefly here: a manually reconstructed `resize`/`crop`/
  normalization, inconsistent with `weights.transforms()`, gave a maximum
  logit difference of $3.67$ on a random test input, and changed the
  predicted class for **1 of 20** real test images (the rest: the same
  class, shifted logits) — an **honest** result, not every example is
  close to a decision boundary, but the effect is real and measurable.
- **Only the weights reproduced, not the configuration.** If the
  architecture is built with default/guessed parameters (e.g. a different
  number of classes in the last layer), loading the `state_dict` will
  **either** raise a dimension-mismatch error (the safe case), **or** —
  worse — if the dimensions happen to match, it will load without error
  and give a **silently wrong** result.

## API and the system boundary

**Minimal contract:**

- `GET /health` — is the process alive (says nothing about the model).
- `GET /ready` — is the model loaded and ready for prediction.
- `POST /predict` — a single prediction.
- `GET /model-info` — the model version, the class list.

**Framework choice: FastAPI + Uvicorn** (past the prototype stage — the
brief allows this approach if simplicity/quality justify the dependency).
Verified: it resolves cleanly with the existing `torch`/Python 3.12;
`TestClient` (built on `httpx`) lets you call the API **inside the
notebook's process**, without running a separate server — this eliminates
instability from managing processes inside Jupyter, while still going
through a **real** HTTP/validation layer.

### Input, output, validation

The input contract for an image classifier: file format, maximum size,
number of channels, **the same** preprocessing as during evaluation.
Response: the predicted class (a name, not just an index), confidence,
the model version.

❌ "An invalid input (a corrupted file, wrong dimensions) should return
the raw Python exception message, so the client knows what happened."

✅ **Not true — that is a leak of internal information.** A raw traceback
can reveal local file paths, code structure, library versions — an API
client should get a **clear, but limited** error message (e.g. HTTP code
400 "invalid image format"), not the full call stack.

### The model's lifecycle in the process

The model is loaded **once**, at process startup — never retrained or
reloaded for every request. The separation: preparing the artifact
(offline, once) → starting the service (loading the model once) →
handling a request (only a forward pass) → monitoring (a separate,
parallel observation process).

❌ "An HTTP service that correctly responds to requests is
'production-ready'."

✅ **Not true — this is a local prototype.** Handling HTTP is **one**
condition out of four (above) — it says nothing about reliability under
load, security, monitoring over time, or the update procedure.

## Mathematics and measuring inference cost

**Latency** of a single request = the time from sending the request to
receiving the response. For $n$ measurements $\ell_1,\ldots,\ell_n$: the
mean $\bar\ell=\frac1n\sum_i \ell_i$; **p50** (median) and **p95** — the
values below which 50%/95% of the measurements lie, respectively.

**A verified note.** `torch.median()` and `torch.quantile(x, 0.5)` **need
not agree** for an even number of measurements — `median()` returns the
**lower** of the two middle elements (the "lower" convention), `quantile`
by default **interpolates** linearly between them. For an example of 20
measurements ($1,\ldots,20$ ms): `median()`$=10.0$, `quantile(0.5)`$=10.5$
— different numbers, different conventions, not a bug.

**Worked example** (the same set, p95): rank $q=0.95$ among $n=20$
measurements corresponds to position (0-indexed, linear interpolation)
$r=q\cdot(n-1)=18.05$ — between the 19th and 20th sorted value:
$19+0.05\cdot(20-19)=19.05$ms. Verified programmatically —
`torch.quantile` agrees exactly with this manual formula.

**Throughput** = the number of requests/images per second. For single
requests: $1000/\bar\ell$ (if $\bar\ell$ is in ms). **For a batch**:
throughput is computed from the batch's **total** time divided by the
**number of elements in the batch** — this is a **different** number
than "the time of a single request," and the two must not be confused.

❌ "A larger batch always gives better throughput (more elements per
second) — this is a general truth about inference on CPU."

✅ **Not true — measured, and the result depends on the architecture, it
is not one universal number.** On our `SmallCNN` (this lecture's notebook,
Fashion-MNIST): time per element **drops** with a larger batch
($batch{=}1\to0.107$ms; $batch{=}8\to0.033$ms; $batch{=}32\to
0.026$ms) — here batching **helps**, as intuition would suggest. **But**
on ResNet18 (the lab, the same CPU, `torch.set_num_threads(4)`, the
result repeated 3× for confirmation): time per element **grows** with a
larger batch ($batch{=}1\to5.86$ms; $batch{=}8\to5.62$ms slightly better;
$batch{=}32\to12.6$–$13.5$ms **clearly worse**) — there batching **hurts**.
The number of parameters/FLOPs is **not** a sufficient predictor of
inference time (see also module 7: MobileNetV3-Small had fewer FLOPs, but
was **slower** on CPU than ResNet18) — the effect of batching **depends
on the specific architecture and hardware**, you have to **measure** it,
not assume it in either direction.

### Experimental measurements (measured, methodology stated explicitly)

- **Cold vs. warm (our `SmallCNN`)**: the 1st call after loading the
  model ($0.162$ms) vs. the next 30 calls (mean$=0.103$ms, p50$=0.102$ms,
  p95$=0.114$ms) — **the cold-start effect here is modest**, not dramatic
  (unlike, say, a CUDA context on GPU). On ResNet18 (the lab, a much
  bigger model): cold $=6.34$ms, warm mean$=5.91$ms — also a modest
  relative effect, but in absolute numbers the model is $\sim$60× slower
  than `SmallCNN` — model size affects the **scale** of the time, not
  just the cold/warm relationship.
- **Anti-example — reloading the model on every request**: for
  `SmallCNN`, $0.57$ms/request (load + inference) vs. $0.10$ms/request
  (the model loaded once) — $5.5\times$ slower. For ResNet18 (the lab,
  more parameters to load): $103.82$ms vs. $5.91$ms — $17.6\times$
  slower. **The same mechanism, a different scale of penalty** — always
  harmful, but more so when the model is larger.

**Methodology** (explicitly reported, not assumed by default): number of
trials (10-30 repetitions per configuration), hardware (this machine,
CPU), number of threads (`torch.set_num_threads(4)`), warm-up (the first
call excluded from the percentiles, counted separately as "cold").

❌ "P95 computed from 3-5 measurements is a reliable estimate of the tail
of the latency distribution."

✅ **Not true.** A high-order percentile (p95, p99) needs **many**
observations to be stable — with 3-5 measurements, "p95" in practice is
simply the maximum or the second-largest, not a true percentile.

## Monitoring: two independent levels

### A. System monitoring

Number of requests, response time, HTTP errors, invalid inputs,
availability, the model version serving the request, input size.

❌ "A service outage and a wrong model prediction are the same type of
problem."

✅ **Not true — these are different levels.** A service can be
**working** (responding, code 200) and returning **systematically wrong**
predictions — system monitoring (availability, response time) **will
not detect** this on its own.

### B. Data and quality monitoring

The distribution of input features, the distribution of predictions, the
distribution of confidence, basic image statistics (e.g. brightness) —
**without** needing labels. Accuracy/F1 — **only** when true labels are
available (usually with a delay).

❌ "We can compute the model's current production accuracy by looking at
the distribution of prediction confidence."

✅ **Not true — this is a category error.** Confidence is the **model's**
output, not a comparison with the truth. **Without labels, the current
accuracy cannot be reliably computed** — we can only monitor **proxy**
signals (the distribution of inputs, predictions, confidence), which
**may** (not must) correlate with a quality change.

❌ "An increase in mean confidence means the model has gotten better."

✅ **Not true — this is a different signal.** Confidence can rise for
many reasons unrelated to actual correctness (e.g. incoming data is
"easier," or the model is poorly calibrated in a new regime — module 14).

**A privacy principle**: monitoring logs **should not** by default store
raw images/text/potentially sensitive data — we store **statistics**
(brightness, size, prediction distribution), not the input data itself,
unless explicitly needed and justified.

## The mathematics of data drift

**Three different phenomena, not synonyms — they can co-occur:**

- **Covariate shift / data drift**: the distribution $P(X)$ of inputs
  changes.
- **Label shift**: the distribution $P(Y)$ of classes changes.
- **Concept drift**: $P(Y\mid X)$ changes — the same input feature now
  corresponds to a **different** true label.

**Population Stability Index (PSI)** — the chosen method (simple, widely
used in production monitoring; KS/JS divergence — mentioned as
alternatives, not implemented in addition, to avoid building a catalog of
detectors). For a feature split into bins $i=1,\ldots,k$, with reference
proportions $p_i$ and current proportions $q_i$:

$$\text{PSI} = \sum_{i=1}^k (q_i - p_i) \ln\frac{q_i}{p_i}$$

**Worked example** (2 bins, ref$=[0.5,0.5]$, cur$=[0.7,0.3]$):

$$\text{PSI} = 0.2\ln(1.4) + ({-}0.2)\ln(0.6) = 0.0673+0.1022 = 0.1695$$

Verified programmatically — exactly this number. Identical distributions
give PSI$=0$ (exactly).

**Zero frequencies.** A bin with zero samples in one of the distributions
gives $\ln(0)=-\infty$ — verified programmatically: **without**
smoothing, PSI is literally `inf`. Solution: smoothing (e.g.
$\max(q_i,\varepsilon)$ for a small $\varepsilon$) — **not arbitrary**,
just necessary for the number to be computable at all; measured: with
$\varepsilon=10^{-4}$, the same case gives a large, but **finite** PSI
($4.60$), not a "fixed" result that pretends there was no problem.

**Conventional thresholds** (an industry heuristic, **not** a statistical
guarantee): PSI$<0.1$ no significant change; $0.1$–$0.25$ moderate;
$>0.25$ significant.

### Limitations of PSI (and measures of this type in general)

- **Dependence on sample size** — few samples give unstable bin
  proportions, susceptible to random fluctuation.
- **Sensitivity to histogram binning** — a different number/boundaries of
  bins can give a different result for **the same** data.
- **The multi-dimensionality problem** — per-feature PSI does not detect
  a change in **correlation** between features (one feature may look
  stable, even though the joint distribution has changed).
- **The possibility of false alarms** — a small, harmless fluctuation can
  exceed the threshold, especially with a small sample.
- **No guarantee of quality degradation** — PSI measures the **input
  distribution**, not $P(Y\mid X)$ — detecting input drift does not say
  whether the model still handles **new** inputs well.

❌ "Detecting data drift is proof that the model's accuracy has dropped —
or: the strength of the PSI signal is proportional to the accuracy drop."

✅ **Not true — measured directly in the notebook (Experiment D), on two
different brightness-shift magnitudes, not one chosen to fit a thesis.**
**Moderate** shift ($+0.03$): PSI$=0.1317$ (drift detected, "moderate"
level), accuracy **exactly unchanged** ($0.8700\to0.8700$). **Stronger**
shift ($+0.15$): PSI$=2.1316$ (drift strongly detected), accuracy
**noticeably dropped** ($0.8700\to0.7800$). Both results from the **same**
experiment, **not** selectively chosen — they show that the relationship
between drift strength and quality drop **is not fixed in advance**:
sometimes drift doesn't hurt, sometimes it clearly does — both quantities
have to be **measured**, not inferred from each other.

❌ "No drift alarm for a chosen feature means the model is operating
stably."

✅ **Not true.** No change in **one, monitored** feature (e.g. brightness)
does not rule out a change in other data dimensions, nor concept drift
(a change in $P(Y\mid X)$ without a change in $P(X)$) — monitoring one
feature is **one** signal, not a complete guarantee.

## What we observe

All numbers come from an actual run of `examples.ipynb` (CPU, this
machine) — Fashion-MNIST, `SmallCNN`.

- **Experiment A.** The artifact saved and reconstructed in a new model
  instance: the logits are **exactly** identical (`torch.equal`).
  Forgetting `eval()` (on a model with `Dropout`) gives a measured,
  nontrivial logit difference ($1.81$; for this image `argmax` happened
  to agree, but the logit shift is real) — described above, with an
  additional, even more dramatic `BatchNorm` case from the lab
  (difference $1.04$, `argmax` completely different).
- **Experiment B.** Cold $0.162$ms vs. warm mean $0.103$ms
  (p50$=0.102$, p95$=0.114$); batch=1→$0.107$ms/elem,
  batch=8→$0.033$ms/elem, batch=32→$0.026$ms/elem (**better** — a
  contrast with ResNet18 in the lab, where a larger batch **hurts**);
  reloading the model per-request: $5.5\times$ slower than a model loaded
  once ($17.6\times$ for the larger ResNet18 in the lab).
- **Experiment C.** A prediction through `TestClient` (a real HTTP/
  validation layer, full preprocessing with a PNG round-trip) agrees
  **exactly** with a direct model call for the same image — the same
  predicted class and confidence. A corrupted input file: HTTP code
  $400$, a clear message, with no traceback revealed.
- **Experiment D.** PSI(ref, control) $=0.0225$ (correctly low, no false
  alarm); PSI(ref, a moderate brightness shift $+0.03$) $=0.1317$ (drift
  detected, "moderate" level), accuracy **exactly unchanged**
  ($0.8700\to0.8700$); PSI(ref, a stronger shift $+0.15$) $=2.1316$
  (drift strongly detected), accuracy **noticeably dropped**
  ($0.8700\to0.7800$) — **both** results from the same experiment, not
  selectively chosen to fit a thesis.
- **Experiment E.** An operational decision based on D: the strength of
  drift and its effect on quality **are not** in a simple relationship —
  the recommendation differs for a moderate shift ("continue monitoring,
  no alarm") and a strong one ("additional analysis / collecting
  labels"), in **neither** case automatic retraining without independent
  verification.

## Versioning, updates, rollback

A simple process: detecting a problem → collecting data/labels → a new
candidate → offline evaluation → comparison with the current model → a
**new artifact version** (not overwriting the old one) → a
reconstruction and API-contract test → deployment → monitoring →
**rollback**, if there is a regression.

❌ "Rolling back a model is just restoring the old weights file."

✅ **Not true, if preprocessing/class mapping has changed in the
meantime.** Rolling back the weights **without** a consistent rollback of
the metadata (class order, preprocessing) can create a **different**, new
form of training-serving skew — a rollback must restore the **entire**
artifact (weights + metadata), not just the weights.

## Security, reliability, privacy (basics)

Limiting input size/format; payload validation; safe artifact loading
(`weights_only=True`); not exposing internal exceptions/paths; avoiding
logging raw user data and secrets; the model loaded once, not downloaded
on every request; data/weight license compliance with the use case.

**This local demonstrator has no authentication or rate limiting** — we
run it **locally**, we do not expose it to the public internet. Full
authorization/rate-limiting are real production concerns, outside the
scope of this module.

## Common misconceptions

(A summary — each has a full ❌/✅ box above)

1. The code working, correct predictions, service reliability, and
   quality stability over time are **four independent** conditions.
2. Loading the weights without error is not proof of correctly
   reconstructing the model.
3. Frozen weights do not mean frozen behavior (`BatchNorm` in `train()`
   mode).
4. A larger batch does not guarantee better throughput — you have to
   measure.
5. P95 from a few measurements is not a reliable estimate of the tail of
   the distribution.
6. Without labels, current accuracy cannot be reliably computed.
7. An increase in confidence does not automatically mean better quality.
8. Data drift, label shift, and concept drift are different phenomena,
   they can co-occur, none automatically implies the others.
9. Detecting drift is not proof of an accuracy drop; no alarm is not
   proof of stable quality.
10. One monitoring alarm should not automatically trigger
    retraining/deployment — it needs a diagnosis.
11. A rollback without consistent preprocessing/class mapping can also be
    wrong.
12. A local HTTP prototype is not automatically "production-ready."

## Engineering implications

- **The artifact is a contract**, not just a weights file — a change in
  one place (e.g. class order) without propagating it to all consumers
  of the artifact is a source of silent bugs.
- **Inference cost depends on hardware/thread configuration/batch size**
  in a way that the number of parameters won't predict — measure, don't
  assume.
- **System monitoring and quality monitoring answer different
  questions** — both are needed, neither replaces the other.
- **Drift is a signal for diagnosis, not a verdict** — it requires
  further analysis (labels, comparison with a baseline), not an automatic
  reaction.
- **Versioning the artifact (not just the weights) enables a safe
  rollback.**
- **Privacy and security start with what we do NOT log** — not only with
  what we secure.

## What to remember

- Artifact = weights (`state_dict`) + architecture + metadata (classes,
  preprocessing, version) — saved and reconstructed, they give
  **bit-exact** identical predictions, verified with `torch.equal`.
- `model.eval()` is not a formality — for a network with
  `BatchNorm`/`Dropout`, the difference from `train()` can be dramatic,
  even with frozen weights.
- Latency (p50/p95), throughput, and memory require an explicit
  methodology (number of trials, hardware, threads, warm-up) — without
  it, the numbers are worthless.
- Monitoring without labels: the distribution of inputs/predictions/
  confidence. Monitoring with (delayed) labels: actual accuracy/F1 —
  these are **different** levels of certainty.
- PSI (or KS/JS) detects a change in the input distribution — it does
  **not** prove a change in model quality; both must be measured
  **separately**.
- An operational decision (no action / further analysis / retraining /
  rollback) requires a **diagnosis**, not one alarm.

## Questions for students

1. You have two artifacts of the same model, differing only in the class
   order in the metadata. How would you detect this, having access only
   to the API (not the service's code)?
2. Why can batch size=32 give **worse** per-element throughput than batch
   size=1 on some CPU configurations — and how would you verify this
   before deploying a specific batch size in production?
3. You have a PSI alert above the threshold for one monitored feature.
   What **three** steps would you take before deciding on retraining?
4. Explain how covariate shift differs from concept drift — and whether
   monitoring the input distribution (without labels) can detect the
   latter.
5. You have a model that, after rolling back to the previous weights
   version, still gives wrong results. What two causes (besides the
   weights themselves) would you check first?

## Bibliography

- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning
  Systems*. NeurIPS (training-serving skew, technical debt in ML).
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance
  Deep Learning Library*. NeurIPS (context for `state_dict`/serialization).
- PyTorch documentation: *Saving and Loading Models*,
  `https://pytorch.org/tutorials/beginner/saving_loading_models.html`.
- PyTorch documentation: `torch.load` (`weights_only`),
  `https://pytorch.org/docs/stable/generated/torch.load.html`.
- FastAPI documentation: `https://fastapi.tiangolo.com/`.
- Moreno-Torres, J. G., et al. (2012). *A unifying view on dataset shift in
  classification*. Pattern Recognition (the covariate/label/concept shift
  distinction).
- PSI: a widely cited heuristic in credit/risk model monitoring (e.g.
  Siddiqi, N. (2012). *Credit Risk Scorecards*. Wiley) — the $0.1$/$0.25$
  thresholds are an industry convention, not a result from a source paper
  with statistical proof.
