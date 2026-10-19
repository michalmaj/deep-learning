[English](README.md) | **Polski**

# Deep Learning

*Praktyczny, oparty na kodzie przewodnik po deep learningu z użyciem PyTorcha.*

Zrozum, jak naprawdę działają sieci neuronowe — poznając matematykę, która
za nimi stoi, uruchamiając eksperymenty, które to pokazują, a na końcu
budując modele samodzielnie. Każdy wykład prowadzi od intuicji przez
formalną matematykę do działającego kodu, a każde laboratorium daje Ci
realny, częściowo przygotowany notebook do samodzielnego dokończenia.

[![CI](https://github.com/michalmaj/deep-learning/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/michalmaj/deep-learning/actions/workflows/ci.yml)
[![Najnowsze wydanie](https://img.shields.io/github/v/release/michalmaj/deep-learning)](https://github.com/michalmaj/deep-learning/releases/latest)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![Języki: EN | PL](https://img.shields.io/badge/languages-EN%20%7C%20PL-555555)](README.md)

## Zacznij tutaj

1. **Przeczytaj wprowadzenie** — [`lectures/01_intro/lecture.md`](lectures/01_intro/lecture.md). Piętnaście minut wystarczy, żeby przejść od "co to właściwie jest deep learning" do ręcznego wyprowadzenia gradientu.
2. **Uruchom pierwszy eksperyment** — [`lectures/01_intro/examples.ipynb`](lectures/01_intro/examples.ipynb). Te same pomysły z wykładu, ale wykonane, narysowane i gotowe do Twoich modyfikacji.
3. **Zrób pierwsze laboratorium** — [`labs/01_intro/README.pl.md`](labs/01_intro/README.pl.md) opisuje zadanie, a [`labs/01_intro/lab.ipynb`](labs/01_intro/lab.ipynb) to miejsce, gdzie je wykonujesz.

Notebooki laboratoryjne mają gotowy szkielet, a same zadania są
zaznaczone jako `TODO` do samodzielnego uzupełnienia — to jest format,
nie błąd. Do każdego laboratorium istnieje prywatne rozwiązanie wzorcowe,
po prostu nie w tym repozytorium (więcej o tym niżej).

## Konfiguracja

```bash
git clone https://github.com/michalmaj/deep-learning.git
cd deep-learning
uv sync
uv run jupyter lab
```

Potrzebujesz [uv](https://docs.astral.sh/uv/getting-started/installation/)
i Pythona 3.12 (uv sam pobierze właściwy interpreter).

CPU wystarcza do wszystkiego, co jest tutaj — wsparcie GPU włącza się
automatycznie, jeśli je masz, ale nigdy nie jest wymagane. Część
wykładów i laboratoriów (głównie późniejszych, np. transfer learning,
embeddingi, inferencja LLM) pobiera zbiór danych albo wytrenowany model
przy pierwszym uruchomieniu; potem jest to zapisane lokalnie w cache.
Czas treningu różni się w zależności od komputera — żaden materiał nie
zależy od konkretnej szybkości, ale kilka eksperymentów jest zauważalnie
wolniejszych na starszym CPU.

## Poznaj materiały

- **Podstawy** — tensory, MLP, backpropagation, optymalizacja, generalizacja
  [`01`](lectures/01_intro/lecture.md) ·
  [`02`](lectures/02_mlp/lecture.md) ·
  [`03`](lectures/03_backprop/lecture.md) ·
  [`04`](lectures/04_optimization/lecture.md) ·
  [`05`](lectures/05_generalization/lecture.md)
- **Widzenie komputerowe i reprezentacje** — CNN, transfer learning, embeddingi
  [`06`](lectures/06_cnn/lecture.md) ·
  [`07`](lectures/07_transfer_learning/lecture.md) ·
  [`08`](lectures/08_embeddings/lecture.md)
- **Sekwencje i język** — RNN, attention, Transformery, inżynieria LLM
  [`09`](lectures/09_rnn/lecture.md) ·
  [`10`](lectures/10_attention/lecture.md) ·
  [`11`](lectures/11_llm_engineering/lecture.md)
- **Modelowanie generatywne** — autoenkodery, VAE, dyfuzja
  [`12`](lectures/12_autoencoders/lecture.md) ·
  [`13`](lectures/13_diffusion/lecture.md)
- **Ewaluacja i wdrażanie** — interpretowalność, kalibracja, serwowanie, monitoring
  [`14`](lectures/14_evaluation_interpretability/lecture.md) ·
  [`15`](lectures/15_production/lecture.md)

Każdy numer prowadzi do wykładu danego modułu; każdy moduł ma też
odpowiadający notebook wykładowy i laboratorium w `lectures/` i `labs/`.

<details>
<summary>Przeglądaj wszystkie moduły (linki do wykładu i laboratorium)</summary>

| # | Wykład | Laboratorium |
|---|---|---|
| 01 | [Deep Learning, ML vs DL, tensory, pipeline treningowy](lectures/01_intro/lecture.md) | [Podstawy PyTorcha](labs/01_intro/README.pl.md) |
| 02 | [Perceptron, MLP, funkcje aktywacji](lectures/02_mlp/lecture.md) | [Klasyfikacja tabelaryczna z MLP](labs/02_mlp/README.pl.md) |
| 03 | [Backpropagation i grafy obliczeniowe](lectures/03_backprop/lecture.md) | [Backprop od zera + autograd](labs/03_backprop/README.pl.md) |
| 04 | [Optymalizacja: SGD, momentum, Adam](lectures/04_optimization/lecture.md) | [Porównanie optymalizatorów](labs/04_optimization/README.pl.md) |
| 05 | [Generalizacja, overfitting, regularyzacja](lectures/05_generalization/lecture.md) | [Dropout, weight decay, early stopping](labs/05_generalization/README.pl.md) |
| 06 | [CNN i reprezentacja obrazu](lectures/06_cnn/lecture.md) | [Klasyfikacja obrazów z CNN](labs/06_cnn/README.pl.md) |
| 07 | [Transfer learning i fine-tuning](lectures/07_transfer_learning/lecture.md) | [Pretrenowany model wizyjny](labs/07_transfer_learning/README.pl.md) |
| 08 | [Embeddingi i reprezentacje uczone](lectures/08_embeddings/lecture.md) | [Eksperymenty z embeddingami i podobieństwem](labs/08_embeddings/README.pl.md) |
| 09 | [RNN, LSTM, GRU i dane sekwencyjne](lectures/09_rnn/lecture.md) | [Klasyfikacja sekwencji/tekstu](labs/09_rnn/README.pl.md) |
| 10 | [Attention i Transformer](lectures/10_attention/lecture.md) | [Attention od podstaw + mały Transformer](labs/10_attention/README.pl.md) |
| 11 | [Podstawy inżynierii LLM](lectures/11_llm_engineering/lecture.md) | [Tokenizacja, inferencja, Hugging Face](labs/11_llm_engineering/README.pl.md) |
| 12 | [Autoenkodery i VAE](lectures/12_autoencoders/lecture.md) | [Przestrzeń latentna / wykrywanie anomalii](labs/12_autoencoders/README.pl.md) |
| 13 | [Modelowanie generatywne z dyfuzją](lectures/13_diffusion/lecture.md) | [Mały eksperyment generatywny / analiza dyfuzji](labs/13_diffusion/README.pl.md) |
| 14 | [Ewaluacja, interpretowalność, analiza błędów](lectures/14_evaluation_interpretability/lecture.md) | [Analiza błędów, Grad-CAM/interpretowalność](labs/14_evaluation_interpretability/README.pl.md) |
| 15 | [Produkcyjny DL: inferencja, koszt, monitoring, drift](lectures/15_production/lecture.md) | [Deployment/API, finalny eksperyment inżynierski](labs/15_production/README.pl.md) |

</details>

## Jak korzystać z tego repozytorium

Te materiały powstały jako kurs uniwersytecki (**Deep Learning**, program
*Inżynieria i zarządzanie sztuczną inteligencją*) — ale żeby z nich
korzystać, nie musisz być na nim zapisany. Proponowany rytm pracy, dla
każdego tematu:

1. Przeczytaj notatki wykładowe, żeby zrozumieć koncepcję i matematykę.
2. Uruchom notebook wykładowy i zobacz eksperyment, który to potwierdza.
3. Zmień jakiś parametr, celowo coś zepsuj, zobacz co się stanie.
4. Zrób samodzielnie zadania z laboratorium — to tam materiał faktycznie
   "siada".
5. Krytycznie przeczytaj własne wyniki: co pokazują i jakie są ich
   ograniczenia?

## Dla kontrybutorów

Jeśli chcesz coś poprawić, dodać moduł, albo po prostu zrozumieć, jak
struktura polsko-angielska jest utrzymywana w zgodności, zobacz
[`CONTRIBUTING.md`](CONTRIBUTING.md) (po angielsku) — opisuje konwencję
plików dwujęzycznych i sposób walidacji notebooków.

## Dane i licencje

- Zbiory danych: dokumentowane ze źródłem, licencją i statusem redystrybucji w [`data/registry.yml`](data/registry.yml).
- Zewnętrzne obrazy i materiały: [`third_party/SOURCES.yml`](third_party/SOURCES.yml).
- Bibliografia: [`references/references.bib`](references/references.bib).
- Kod: **MIT** — zobacz [`LICENSE`](LICENSE).
- Oryginalne treści wykładowe/laboratoryjne: **CC BY-SA 4.0** — zobacz [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
- Materiały zewnętrzne zachowują swoją oryginalną licencję.

**Rozwiązania wzorcowe do laboratoriów nie są tutaj publikowane** —
publiczne są tylko opisy zadań, kod startowy i testy, dzięki czemu
zadania pozostają zadaniami.

---

Wersja angielska: [`README.md`](README.md). Najnowsze informacje o
wydaniu: [GitHub Releases](https://github.com/michalmaj/deep-learning/releases/latest).
