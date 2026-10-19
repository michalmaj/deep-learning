**English** | [Polski](README.pl.md)

# Deep Learning

*A practical, code-first guide to understanding deep learning with PyTorch.*

Learn how neural networks actually work by exploring the math behind
them, running the experiments that demonstrate it, and then building the
models yourself. Every lecture moves from intuition to formal mathematics
to runnable code, and every lab hands you a real, partially-built
notebook to finish on your own.

[![CI](https://github.com/michalmaj/deep-learning/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/michalmaj/deep-learning/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/michalmaj/deep-learning)](https://github.com/michalmaj/deep-learning/releases/latest)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![Languages: EN | PL](https://img.shields.io/badge/languages-EN%20%7C%20PL-555555)](README.pl.md)

## Start here

1. **Read the introduction** — [`lectures/01_intro/lecture.en.md`](lectures/01_intro/lecture.en.md). Fifteen minutes gets you from "what is deep learning, really" to deriving a gradient by hand.
2. **Run your first experiment** — [`lectures/01_intro/examples.en.ipynb`](lectures/01_intro/examples.en.ipynb). The same ideas from the lecture, but executed, plotted, and yours to tweak.
3. **Try the first lab** — [`labs/01_intro/README.md`](labs/01_intro/README.md) sets up the task, [`labs/01_intro/lab.en.ipynb`](labs/01_intro/lab.en.ipynb) is where you do it.

Lab notebooks ship with the scaffolding done and the actual exercises
left as `TODO`s for you to fill in — that's the format, not something
broken. A private reference solution exists for every lab, just not in
this repository (more on that below).

## Setup

```bash
git clone https://github.com/michalmaj/deep-learning.git
cd deep-learning
uv sync
uv run jupyter lab
```

You'll need [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Python 3.12 (uv will fetch the right interpreter for you).

A CPU is enough for everything here — GPU support is automatic if you
have one, but never required. A handful of lectures and labs (mostly the
later ones, e.g. transfer learning, embeddings, LLM inference) download a
dataset or a pretrained model the first time you run them; after that
they're cached locally. Training times vary by machine — none of the
material depends on a specific speed, but a few experiments are
noticeably slower on an older CPU.

## Explore the material

- **Fundamentals** — tensors, MLPs, backpropagation, optimization, generalization
  [`01`](lectures/01_intro/lecture.en.md) ·
  [`02`](lectures/02_mlp/lecture.en.md) ·
  [`03`](lectures/03_backprop/lecture.en.md) ·
  [`04`](lectures/04_optimization/lecture.en.md) ·
  [`05`](lectures/05_generalization/lecture.en.md)
- **Computer vision and representations** — CNNs, transfer learning, embeddings
  [`06`](lectures/06_cnn/lecture.en.md) ·
  [`07`](lectures/07_transfer_learning/lecture.en.md) ·
  [`08`](lectures/08_embeddings/lecture.en.md)
- **Sequences and language** — RNNs, attention, Transformers, LLM engineering
  [`09`](lectures/09_rnn/lecture.en.md) ·
  [`10`](lectures/10_attention/lecture.en.md) ·
  [`11`](lectures/11_llm_engineering/lecture.en.md)
- **Generative modeling** — autoencoders, VAEs, diffusion
  [`12`](lectures/12_autoencoders/lecture.en.md) ·
  [`13`](lectures/13_diffusion/lecture.en.md)
- **Evaluation and deployment** — interpretability, calibration, serving, monitoring
  [`14`](lectures/14_evaluation_interpretability/lecture.en.md) ·
  [`15`](lectures/15_production/lecture.en.md)

Each number links to that module's lecture; every module also has a
matching lecture notebook and lab under `lectures/` and `labs/`.

<details>
<summary>Browse all modules (lecture + lab links)</summary>

| # | Lecture | Lab |
|---|---|---|
| 01 | [Deep Learning, ML vs DL, tensors, training loop](lectures/01_intro/lecture.en.md) | [PyTorch basics](labs/01_intro/README.md) |
| 02 | [Perceptron, MLP, activation functions](lectures/02_mlp/lecture.en.md) | [Tabular classification with an MLP](labs/02_mlp/README.md) |
| 03 | [Backpropagation and computational graphs](lectures/03_backprop/lecture.en.md) | [Backprop from scratch + autograd](labs/03_backprop/README.md) |
| 04 | [Optimization: SGD, momentum, Adam](lectures/04_optimization/lecture.en.md) | [Optimizer comparison](labs/04_optimization/README.md) |
| 05 | [Generalization, overfitting, regularization](lectures/05_generalization/lecture.en.md) | [Dropout, weight decay, early stopping](labs/05_generalization/README.md) |
| 06 | [CNNs and image representation](lectures/06_cnn/lecture.en.md) | [Image classification with a CNN](labs/06_cnn/README.md) |
| 07 | [Transfer learning and fine-tuning](lectures/07_transfer_learning/lecture.en.md) | [A pretrained vision model](labs/07_transfer_learning/README.md) |
| 08 | [Embeddings and learned representations](lectures/08_embeddings/lecture.en.md) | [Embedding and similarity experiments](labs/08_embeddings/README.md) |
| 09 | [RNNs, LSTM, GRU, and sequential data](lectures/09_rnn/lecture.en.md) | [Sequence/text classification](labs/09_rnn/README.md) |
| 10 | [Attention and the Transformer](lectures/10_attention/lecture.en.md) | [Attention from scratch + a small Transformer](labs/10_attention/README.md) |
| 11 | [LLM engineering fundamentals](lectures/11_llm_engineering/lecture.en.md) | [Tokenization, inference, Hugging Face](labs/11_llm_engineering/README.md) |
| 12 | [Autoencoders and VAEs](lectures/12_autoencoders/lecture.en.md) | [Latent space / anomaly detection](labs/12_autoencoders/README.md) |
| 13 | [Generative modeling with diffusion](lectures/13_diffusion/lecture.en.md) | [A small generative experiment / diffusion analysis](labs/13_diffusion/README.md) |
| 14 | [Evaluation, interpretability, error analysis](lectures/14_evaluation_interpretability/lecture.en.md) | [Error analysis, Grad-CAM/interpretability](labs/14_evaluation_interpretability/README.md) |
| 15 | [Production DL: inference, cost, monitoring, drift](lectures/15_production/lecture.en.md) | [Deployment/API, a final engineering experiment](labs/15_production/README.md) |

</details>

## How to use this repository

These materials grew out of a university course (**Deep Learning**, for
the *Artificial Intelligence Engineering and Management* programme) —
but you don't need to be enrolled anywhere to use them. The suggested
rhythm, per topic:

1. Read the lecture notes to get the concept and the math.
2. Run the lecture notebook and see the experiment that backs it up.
3. Change a parameter, break something on purpose, see what happens.
4. Do the lab's exercises yourself — that's where the material actually sinks in.
5. Read your own results critically: what do they show, and what are their limits?

## For contributors

If you want to fix something, add a module, or just understand how the
English/Polish structure is kept in sync, see
[`CONTRIBUTING.md`](CONTRIBUTING.md) — it covers the bilingual file
convention and how notebooks are validated.

## Data and licensing

- Datasets: documented with source, license, and redistribution status in [`data/registry.yml`](data/registry.yml).
- External figures and assets: [`third_party/SOURCES.yml`](third_party/SOURCES.yml).
- Bibliography: [`references/references.bib`](references/references.bib).
- Code: **MIT** — see [`LICENSE`](LICENSE).
- Original lecture/lab content: **CC BY-SA 4.0** — see [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
- Third-party material keeps its own original license.

**Reference solutions to the labs aren't published here** — only the
task descriptions, starter code, and tests are public, so the exercises
stay exercises.

---

Polish version: [`README.pl.md`](README.pl.md). Latest release notes:
[GitHub Releases](https://github.com/michalmaj/deep-learning/releases/latest).
