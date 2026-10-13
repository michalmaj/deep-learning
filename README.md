# Deep Learning

Repozytorium kursu **Deep Learning** (program: *Inżynieria i zarządzanie
sztuczną inteligencją*).

Kurs obejmuje 30h wykładów i 30h laboratoriów (15 × 2h wykład + 15 × 2h
laboratorium). To repozytorium ma być **wersjonowanym podręcznikiem +
wykonywalnym repozytorium kursu**, nie tylko zestawem slajdów.

Sekwencja dydaktyczna przyjęta w materiałach:

> intuicja → teoria → matematyka → eksperyment wykonywalny → samodzielna
> praca studenta → implikacje inżynierskie

## Status

Repozytorium jest obecnie na etapie **fundamentu** (środowisko, narzędzia,
rejestry źródeł/danych, licencje). Moduły wykładowe i laboratoryjne będą
dodawane przyrostowo, w kolejnych, samodzielnych pull requestach.

## Planowana struktura kursu

| # | Wykład | Kierunek laboratorium |
|---|---|---|
| 1 | Deep Learning, ML vs DL, tensory, pipeline treningowy | PyTorch, tensory, device handling, pierwszy model |
| 2 | Perceptron, MLP, funkcje aktywacji | Klasyfikacja tabelaryczna z MLP |
| 3 | Backpropagation i grafy obliczeniowe | Backprop od zera + autograd |
| 4 | Optymalizacja: SGD, momentum, Adam | Porównanie optymalizatorów |
| 5 | Generalizacja, overfitting, regularyzacja | Dropout, weight decay, early stopping |
| 6 | CNN i reprezentacja obrazu | Klasyfikacja obrazów z CNN |
| 7 | Transfer learning i fine-tuning | Pretrenowany model wizyjny |
| 8 | Embeddingi i reprezentacje uczone | Eksperymenty z embeddingami i podobieństwem |
| 9 | RNN, LSTM, GRU i dane sekwencyjne | Klasyfikacja sekwencji/tekstu |
| 10 | Attention i Transformer | Attention od podstaw + mały Transformer |
| 11 | Podstawy inżynierii LLM | Tokenizacja, inferencja, Hugging Face |
| 12 | Autoencodery i VAE | Przestrzeń latentna / wykrywanie anomalii |
| 13 | Generative AI i dyfuzja | Mały eksperyment generatywny / analiza dyfuzji |
| 14 | Ewaluacja, interpretowalność, analiza błędów | Analiza błędów, Grad-CAM/interpretowalność |
| 15 | Produkcyjny DL: inferencja, koszt, monitoring, drift | Deployment/API/finalny eksperyment inżynierski |

## Środowisko

Projekt używa [uv](https://docs.astral.sh/uv/) do zarządzania zależnościami
i środowiskiem Python.

```bash
uv sync                 # instaluje zależności (uv.lock)
uv run jupyter lab      # uruchamia Jupyter Lab w środowisku projektu
```

Dostępne cele `make`:

```bash
make setup         # uv sync
make test           # testy (pytest)
make lint            # ruff check
make verify-data     # walidacja checksumów danych wg data/registry.yml
```

Kod działa na CPU jako bazowej platformie; GPU (CUDA) jest opcjonalnym
przyspieszeniem, nigdy wymaganiem.

## Dane i materiały zewnętrzne

- Zbiory danych są dokumentowane w [`data/registry.yml`](data/registry.yml)
  (źródło, licencja, status redystrybucji, checksumy) — zobacz
  [`data/README.md`](data/README.md).
- Zewnętrzne obrazy/figury są dokumentowane w
  [`third_party/SOURCES.yml`](third_party/SOURCES.yml) — zobacz
  [`third_party/README.md`](third_party/README.md).
- Centralna bibliografia: [`references/references.bib`](references/references.bib).

Żaden zewnętrzny materiał nie jest dodawany do repozytorium bez sprawdzenia
jego licencji i statusu redystrybucji.

## Licencja

- Oryginalny kod źródłowy: **MIT** — zobacz [`LICENSE`](LICENSE).
- Oryginalne treści dydaktyczne (tekst wykładów, materiały laboratoryjne,
  oryginalne diagramy): **CC BY-SA 4.0** — zobacz
  [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
- Materiały zewnętrzne (third-party) pozostają na swoich oryginalnych
  licencjach — zobacz [`third_party/SOURCES.yml`](third_party/SOURCES.yml).

Pełne rozwiązania laboratoryjne nie są publikowane w tym repozytorium.
