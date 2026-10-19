Language: **English (default)** | [Polski](README.pl.md)

# Deep Learning

Course repository for **Deep Learning**, part of the *Artificial
Intelligence Engineering and Management* program.

The course covers 30 hours of lectures and 30 hours of labs (15 × 2h
lecture + 15 × 2h lab). This repository is meant to be a **versioned
textbook and an executable course repository**, not just a slide deck.

The teaching sequence used throughout the materials:

> intuition → theory → mathematics → executable experiment → independent
> student work → engineering implications

The course assumes basic Python, NumPy, pandas, elementary machine
learning, linear algebra, and differential calculus. It builds up from a
single neuron to MLPs, backpropagation, optimization, CNNs, transfer
learning, embeddings, sequence models, attention and Transformers, a
small LLM pipeline, autoencoders/VAEs, diffusion models, evaluation and
interpretability, and ends with production engineering (serving,
monitoring, drift). All core experiments run on **CPU**; GPU is an
optional speed-up, never a requirement.

## Status

All 15 modules are complete in Polish, the course's original language.
**English localization is in progress.** Modules 1-13 are now fully
available in both languages (lecture, lecture notebook, lab instructions,
and lab notebook) -- the first thirteen modules, from PyTorch basics
through LLM engineering, autoencoders/VAEs, and diffusion models, can be
followed start to finish in English. Modules 14-15 are not yet
localized -- their lecture and lab links below point to the existing,
Polish-only materials, clearly marked as such. See
[the Polish version of this page](README.pl.md) for the course described
in its original language.

## Course structure

| # | Lecture | Lab direction |
|---|---|---|
| 1 | [Deep Learning, ML vs DL, tensors, training pipeline](lectures/01_intro/lecture.en.md) | [PyTorch, tensors, device handling, first model](labs/01_intro/README.md) |
| 2 | [Perceptron, MLP, activation functions](lectures/02_mlp/lecture.en.md) | [Tabular classification with MLP](labs/02_mlp/README.md) |
| 3 | [Backpropagation and computational graphs](lectures/03_backprop/lecture.en.md) | [Backprop from scratch + autograd](labs/03_backprop/README.md) |
| 4 | [Optimization: SGD, momentum, Adam](lectures/04_optimization/lecture.en.md) | [Optimizer comparison](labs/04_optimization/README.md) |
| 5 | [Generalization, overfitting, regularization](lectures/05_generalization/lecture.en.md) | [Dropout, weight decay, early stopping](labs/05_generalization/README.md) |
| 6 | [CNNs and image representation](lectures/06_cnn/lecture.en.md) | [Image classification with a CNN](labs/06_cnn/README.md) |
| 7 | [Transfer learning and fine-tuning](lectures/07_transfer_learning/lecture.en.md) | [A pretrained vision model](labs/07_transfer_learning/README.md) |
| 8 | [Embeddings and learned representations](lectures/08_embeddings/lecture.en.md) | [Embedding and similarity experiments](labs/08_embeddings/README.md) |
| 9 | [RNNs, LSTM, GRU, and sequential data](lectures/09_rnn/lecture.en.md) | [Sequence/text classification](labs/09_rnn/README.md) |
| 10 | [Attention and the Transformer](lectures/10_attention/lecture.en.md) | [Attention from scratch + a small Transformer](labs/10_attention/README.md) |
| 11 | [LLM engineering fundamentals](lectures/11_llm_engineering/lecture.en.md) | [Tokenization, inference, Hugging Face](labs/11_llm_engineering/README.md) |
| 12 | [Autoencoders and VAE](lectures/12_autoencoders/lecture.en.md) | [Latent space / anomaly detection](labs/12_autoencoders/README.md) |
| 13 | [Generative AI and diffusion](lectures/13_diffusion/lecture.en.md) | [A small generative experiment / diffusion analysis](labs/13_diffusion/README.md) |
| 14 | [Evaluation, interpretability, error analysis](lectures/14_evaluation_interpretability/lecture.md) (Polish only) | [Error analysis, Grad-CAM/interpretability](labs/14_evaluation_interpretability/README.md) (Polish only) |
| 15 | [Production DL: inference, cost, monitoring, drift](lectures/15_production/lecture.md) (Polish only) | [Deployment/API, final engineering experiment](labs/15_production/README.md) (Polish only) |

## Environment

The project uses [uv](https://docs.astral.sh/uv/) to manage dependencies
and the Python environment.

```bash
uv sync                 # installs dependencies (uv.lock)
uv run jupyter lab      # starts Jupyter Lab inside the project's environment
```

Available `make` targets:

```bash
make setup         # uv sync
make test           # tests (pytest)
make lint           # ruff check
make verify-data    # checksum validation against data/registry.yml
```

The code runs on CPU as the baseline platform; GPU (CUDA) is an optional
speed-up, never a requirement.

## Data, weights, and required downloads

- Datasets are documented in [`data/registry.yml`](data/registry.yml)
  (source, license, redistribution status, checksums) -- see
  [`data/README.md`](data/README.md).
- Pretrained weights used by later modules (e.g. ResNet18, a small LLM)
  follow the same provenance/licensing discipline and are listed in the
  same registry -- never committed to the repository, always downloaded
  on demand by the notebooks that need them.
- External figures are documented in
  [`third_party/SOURCES.yml`](third_party/SOURCES.yml) -- see
  [`third_party/README.md`](third_party/README.md).
- Central bibliography: [`references/references.bib`](references/references.bib).

No external material is added to the repository without first checking
its license and redistribution status.

## Tests and reproducibility

Every lecture/lab notebook is expected to run from a clean kernel
("Restart Kernel -> Run All"). `tests/` pairs a full-notebook-execution
test for each module with always-on, network-free synthetic tests that
check the underlying mechanics by hand-computed example -- the synthetic
tests run in CI without internet access or large downloads; the full
notebook-execution tests are skipped in CI (`CI=true`) when they need a
real dataset/model download, and are expected to be run locally before
opening a PR. Run `make test` (or `uv run pytest`) for the full local
suite, or `CI=true uv run pytest` for the fast, network-free subset.

## License

- Original source code: **MIT** -- see [`LICENSE`](LICENSE).
- Original teaching content (lecture text, lab materials, original
  diagrams): **CC BY-SA 4.0** -- see
  [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
- Third-party materials keep their own, original licenses -- see
  [`third_party/SOURCES.yml`](third_party/SOURCES.yml).

**Full lab solutions are never published in this repository**, in either
language -- only task descriptions, starter code, scaffolding, and tests
are public.
