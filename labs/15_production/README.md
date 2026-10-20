Language: **English (default)** | [Polski](README.pl.md)

# Lab 15: a local inference service with monitoring (EuroSAT + ResNet18)

Time: ca. 120 minutes. **The last lab of the course** -- it combines data
preparation, training, evaluation, cost, and interpretability from
earlier modules into one, practical engineering experiment. Builds on
[`lectures/15_production/lecture.en.md`](../../lectures/15_production/lecture.en.md),
but on a **different problem** than the lecture (there: Fashion-MNIST
from scratch; here: EuroSAT + ResNet18 frozen backbone, following
modules 07/14).

**Goal**: finish the lab with a **working local API**, the ability to
reproduce the artifact, benchmark results, and a short report on the
decision to deploy/maintain the model.

## This module's directory structure

```
labs/15_production/
├── README.md                (this file)
├── lab.en.ipynb               (the notebook tying everything together, with TODOs)
├── prepare_artifact.py       (provided script -- training + saving the artifact)
├── app/
│   ├── model.py               (load_artifact PROVIDED, preprocessing+predict YOURS)
│   ├── schemas.py              (response models -- provided)
│   └── server.py               (health/ready/model-info endpoints provided, /predict YOURS)
├── benchmark.py               (sending requests provided, computing stats YOURS)
└── monitoring.py               (streams/histogram provided, PSI YOURS)
```

The `app/`, `benchmark.py`, `monitoring.py`, `model_colleague.py`, and
`prepare_artifact.py` files are **shared code**, not duplicated per
language -- they are already in English (identifiers, docstrings,
comments), since they are part of the service's actual, working contract,
not instructional prose. `lab.en.ipynb` imports from these same files,
identically to `lab.ipynb`.

The artifact (`artifacts/weights.pt` + `artifacts/metadata.json`) is
generated **locally** by `prepare_artifact.py` and **ignored by Git**
(`.gitignore`) -- you never commit the weights or the dataset.

## Exact commands

1. **Environment** (if not already done): `uv sync`.
2. **Preparing the artifact** (downloads EuroSAT + ResNet18 weights on
   first run, ~35s): `uv run python labs/15_production/prepare_artifact.py`.
3. **Tests** (after filling in the TODOs): `uv run pytest tests/test_lab_15_scaffold.py -v`.
4. **Notebook**: `uv run jupyter lab labs/15_production/lab.en.ipynb` --
   there you build a `TestClient`, send requests, run the
   benchmark/monitoring.
5. The lab does **not** require running a separate server process
   (`uvicorn ...`) -- `fastapi.testclient.TestClient` calls the same
   application inside the notebook's process, through a real HTTP/
   validation layer.

## 1. The model artifact (provided + verification)

`prepare_artifact.py` trains the head (`nn.Linear`) on frozen ResNet18
(`IMAGENET1K_V1`) features and saves `state_dict` + `metadata.json`
(version, class list, preprocessing, input size). This is **scaffold**
code -- not your task (training the head is a review of modules 07/14,
not a new topic).

Image samples are streamed through a `DataLoader` in small batches during
feature extraction. The script retains the extracted feature tensors, not
all decoded 224x224 images, which bounds the image-memory portion by the
batch size.

The generator and loader share one artifact contract. Before loading the
weights, `load_artifact()` rejects metadata with an incompatible
architecture, class mapping, input shape, or preprocessing declaration.
This prevents a valid state dict from being served with stale or misleading
metadata.

**Your task:** run the script, **verify programmatically** in the
notebook that the model loaded from the artifact gives **identical**
(`torch.equal`) logits to the model right after training (the same test
image).

## 2. `app/model.py` -- preprocessing and inference -- your task

`load_artifact()` is provided (it builds the architecture, loads the
weights + metadata). Your task:

1. `preprocess_image_bytes(data)` -- decode the image (`PIL.Image`),
   apply **exactly** the preprocessing used during training
   (`ResNet18_Weights.IMAGENET1K_V1.transforms()` -- **do not**
   reconstruct it by hand via `resize`/`crop`/normalization, see the
   diagnostic task below for exactly what breaks if you do).
2. `predict(loaded, image_bytes)` -- a forward pass, softmax, build the
   response dict (`predicted_class`, `confidence`, `model_version`).

## 3. `app/server.py` -- the API contract -- your task

`/health`, `/ready`, `/model-info` are provided. Your task: `/predict`:

The provided application uses FastAPI's lifespan context to load the model
once at startup and release the in-process reference during shutdown.
`/model-info` returns HTTP 503 whenever the model is not ready.

Upload safety has three layers:

- `RequestBodyLimitMiddleware` caps the complete `/predict` request while
  ASGI chunks arrive, before Starlette can spool a multipart upload to disk.
  The request allowance includes 64 KiB of multipart overhead around the
  5 MB file limit.
- `read_upload_limited()` enforces the exact 5 MB file limit and rejects an
  empty upload.
- `model.decode_image_bytes()` rejects images that expand beyond 25 million
  decoded pixels, even when the compressed file is small.

Your endpoint should:

1. Return HTTP 503 when the model is not ready.
2. Read the file with `read_upload_limited()`.
3. Return `await run_prediction(...)`. The helper moves synchronous,
   CPU-bound inference off the event-loop thread and maps only known input
   errors to safe HTTP 400/413 responses. Unexpected model or infrastructure
   failures remain HTTP 500 errors.

## 4. API consistency with a direct call -- your task

Build `TestClient(app)`, send a real EuroSAT image, compare the API
response with a direct call to `model.predict()` on the same image -- it
**must match exactly** (the same class and confidence).

## 5. `benchmark.py` -- measuring cost -- your task

`send_requests(client, image_bytes, n)` is provided. It measures elapsed
time with a monotonic clock and rejects unsuccessful HTTP responses, so
only valid predictions become benchmark samples. Your task:
`compute_stats(times)` -- mean, p50, p95 (ms), throughput (req/s).
**Measure**: cold (the 1st request) vs. warm (the next 20+), batch size=1
for single requests (the API accepts one image per request -- throughput
is computed from **many sequential** requests, not batching within one
request).

## 6. `monitoring.py` -- PSI and controlled streams -- your task

`histogram_counts()` and `brightness_feature()` are provided. Your task:
`psi(ref_counts, cur_counts, eps)` -- the PSI formula from the lecture,
**with smoothing** of zero frequencies (check in the notebook what
happens without `eps`).

## 7. Monitoring scenarios -- your task

**Scenario A (control)**: an independent sample from the **same**
distribution as the reference one. Check: PSI **should not** give a
false alarm.

**Scenario B (a controlled change)**: choose **one**, measurable
transformation (e.g. brightness +X, contrast ×Y) and **one** magnitude
for it. Check: does PSI detect the change?

**Scenario C (delayed labels)**: for the data from scenario B, compute
accuracy (you have EuroSAT labels -- you are simulating that they become
available "with a delay," after the fact). **Compare** with the PSI
result from scenario B -- **do drift and the accuracy change go in the
same direction, or not?**

❌ "If PSI detected drift in scenario B, accuracy must have dropped in
scenario C."

✅ **Check this, don't assume it.** See the lecture: on a different
dataset (Fashion-MNIST), a strong drift **lowered** accuracy; a moderate
one did not. On **your** data (EuroSAT), the result may come out
**differently** -- this is something you must **measure** in this
specific experiment, not predict in advance.

**Important**: don't pick the PSI threshold or the transformation
strength **after** seeing the accuracy result, just to "show" an alarm --
choose them **before** checking accuracy, following the evaluation
methodology from module 5/10.

## 8. Diagnostic task: training-serving skew

The following, **separately provided** `model_colleague.py` implements
`preprocess_image_bytes` **by hand** (resizing to 224×224 with no
intermediate 256→crop step, without using `weights.transforms()`),
instead of using the library transform.

**Task:**
1. Run a prediction on the same image through **your** `model.py` and
   through `model_colleague.py`. Compare the logits.
2. Explain **exactly** which preprocessing step differs and why it
   matters (refer to `ResNet18_Weights.IMAGENET1K_V1.
   transforms()` -- what exactly it does: `resize(256)` →
   `center_crop(224)` → normalization -- and what the hand-rolled version
   does not do).
3. Check on **several** (e.g. 20) real EuroSAT images: how many
   predictions **change** (not just the logits)?
4. Confirm that **your** implementation (section 2) agrees with the
   direct, verified call (section 4) -- while the colleague's version
   does not.

❌ "This is a bug that requires guessing how the library's hooks/internal
mechanisms work."

✅ **No -- this is a difference in VISIBLE preprocessing steps**, directly
comparable (image size after each step, pixel values before/after
normalization) -- it does not require guessing.

## 9. Report: a decision on deploying and maintaining the model -- your task

A short report (markdown in the notebook), answering:

1. Model version, data source, the offline evaluation protocol.
2. Accuracy and the most important limitations (e.g. from a module-14-style
   analysis, if you have time -- optional, not required again here).
3. The API's input/output format.
4. Latency p50/p95, throughput (your measured numbers).
5. The results of scenarios A/B/C.
6. The level of confidence with which you interpret the drift result
   (one feature, a limited number of samples -- name this explicitly).
7. Known risks, recommended next steps.

**Do not force the conclusion "the model is ready for production."** An
acceptable, and sometimes the best, conclusion: *"the system works
technically, but needs additional data, tests, and safeguards before
real deployment."*

## Time budget (realistic, verified against actual CPU cost)

| Block | Time |
|---|---|
| Section 1 (artifact, verification) | 8 min |
| Section 2 (`model.py`: preprocessing + predict) | 15 min |
| Section 3 (`server.py`: `/predict`) | 12 min |
| Section 4 (API vs. direct call consistency) | 8 min |
| Section 5 (`benchmark.py`: statistics) | 10 min |
| Section 6 (`monitoring.py`: PSI) | 10 min |
| Section 7 (scenarios A/B/C) | 15 min |
| Section 8 (diagnostic task) | 15 min |
| Section 9 (report) | 12 min |

Total: **105 min** core + a **15-minute** buffer (120 min total). CPU
cost: `prepare_artifact.py` ~35s (one-time, mostly feature extraction); a
single request through the full API stack ~7-8ms (measured); monitoring
scenarios -- seconds. The budget goes into implementation and
interpretation, not into waiting for the model.

## Grading criteria

- The artifact is verified to give **identical** predictions after
  reconstruction (`torch.equal`).
- `/predict` correctly calls the **real** preprocessing (library-based,
  not hand-reconstructed) and returns a correct response.
- The API response agrees **exactly** with a direct model call for the
  same image.
- Invalid inputs (a corrupted file, an oversized request/file, or an image
  with too many decoded pixels) are handled without leaking internal
  information, while server failures are not mislabeled as HTTP 400.
- The benchmark distinguishes cold/warm, reports p50/p95 with an explicit
  methodology (number of trials).
- PSI correctly handles zero frequencies (smoothing, not `inf`/`nan`).
- Scenario A does not generate a false alarm; scenario B/C are
  **actually measured**, not assumed -- the conclusion about the drift/
  accuracy relationship is based on **this** experiment, not a general
  rule.
- The diagnostic task correctly identifies the preprocessing difference
  and its measured impact on predictions (not just the logits).
- The report answers all the points, without forcing a "ready for
  production" conclusion.

A full reference solution exists locally with the instructor
(`decisions/solutions/15_production/`) and is not part of this
repository.
