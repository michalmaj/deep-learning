Język: [English](README.md) | **Polski**

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

Wszystkie 15 modułów (wykłady + laboratoria) są kompletne po polsku, a
**wszystkie 15 modułów mają już też kompletne tłumaczenie na angielski**
(wykład, notebook wykładowy, instrukcja laboratorium, notebook
studencki) — cały kurs, od podstaw PyTorcha po ewaluację/
interpretowalność i inżynierię produkcyjną, można przejść od początku do
końca po angielsku. Ta partia lokalizacji przeszła przez końcowy audyt
dwujęzyczny: wszystkie 30 par notebooków PL/EN przechodzą automatyczną
kontrolę równoważności strukturalnej/liczbowej, nawigacja między
językami została zweryfikowana dla każdego modułu, a znany backlog
tłumaczeniowy został rozstrzygnięty — zobacz `CONTRIBUTING.md` (po
angielsku) po dokładny zakres tego, co ten audyt zweryfikował, a czego
nie (sprawdzał parytet strukturalny/liczbowy, nie jakość prozy przez
native speakera, i nie był testowany na żywych zajęciach) — zobacz
[angielską wersję tej strony](README.md).

## Struktura kursu

Wszystkie 15 modułów są kompletne. Link w kolumnie "Wykład" prowadzi do
`lecture.md`, link w kolumnie "Laboratorium" do `README.pl.md`, polskiej
wersji instrukcji laboratorium.

| # | Wykład | Laboratorium |
|---|---|---|
| 1 | [Deep Learning, ML vs DL, tensory, pipeline treningowy](lectures/01_intro/lecture.md) | [PyTorch, tensory, device handling, pierwszy model](labs/01_intro/README.pl.md) |
| 2 | [Perceptron, MLP, funkcje aktywacji](lectures/02_mlp/lecture.md) | [Klasyfikacja tabelaryczna z MLP](labs/02_mlp/README.pl.md) |
| 3 | [Backpropagation i grafy obliczeniowe](lectures/03_backprop/lecture.md) | [Backprop od zera + autograd](labs/03_backprop/README.pl.md) |
| 4 | [Optymalizacja: SGD, momentum, Adam](lectures/04_optimization/lecture.md) | [Porównanie optymalizatorów](labs/04_optimization/README.pl.md) |
| 5 | [Generalizacja, overfitting, regularyzacja](lectures/05_generalization/lecture.md) | [Dropout, weight decay, early stopping](labs/05_generalization/README.pl.md) |
| 6 | [CNN i reprezentacja obrazu](lectures/06_cnn/lecture.md) | [Klasyfikacja obrazów z CNN](labs/06_cnn/README.pl.md) |
| 7 | [Transfer learning i fine-tuning](lectures/07_transfer_learning/lecture.md) | [Pretrenowany model wizyjny](labs/07_transfer_learning/README.pl.md) |
| 8 | [Embeddingi i reprezentacje uczone](lectures/08_embeddings/lecture.md) | [Eksperymenty z embeddingami i podobieństwem](labs/08_embeddings/README.pl.md) |
| 9 | [RNN, LSTM, GRU i dane sekwencyjne](lectures/09_rnn/lecture.md) | [Klasyfikacja sekwencji/tekstu](labs/09_rnn/README.pl.md) |
| 10 | [Attention i Transformer](lectures/10_attention/lecture.md) | [Attention od podstaw + mały Transformer](labs/10_attention/README.pl.md) |
| 11 | [Podstawy inżynierii LLM](lectures/11_llm_engineering/lecture.md) | [Tokenizacja, inferencja, Hugging Face](labs/11_llm_engineering/README.pl.md) |
| 12 | [Autoencodery i VAE](lectures/12_autoencoders/lecture.md) | [Przestrzeń latentna / wykrywanie anomalii](labs/12_autoencoders/README.pl.md) |
| 13 | [Generative AI i dyfuzja](lectures/13_diffusion/lecture.md) | [Mały eksperyment generatywny / analiza dyfuzji](labs/13_diffusion/README.pl.md) |
| 14 | [Ewaluacja, interpretowalność, analiza błędów](lectures/14_evaluation_interpretability/lecture.md) | [Analiza błędów, Grad-CAM/interpretowalność](labs/14_evaluation_interpretability/README.pl.md) |
| 15 | [Produkcyjny DL: inferencja, koszt, monitoring, drift](lectures/15_production/lecture.md) | [Deployment/API/finalny eksperyment inżynierski](labs/15_production/README.pl.md) |

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

## Testy i reprodukowalność

Każdy notebook wykładowy/laboratoryjny powinien wykonywać się od czystego
kernela ("Restart Kernel -> Run All"). `tests/` łączy dla każdego modułu
test wykonujący cały notebook z zawsze-włączonymi, niewymagającymi sieci
testami syntetycznymi, które sprawdzają mechanikę na przykładzie liczonym
ręcznie — testy syntetyczne działają w CI bez dostępu do internetu i bez
dużych pobrań; testy wykonujące cały notebook są pomijane w CI
(`CI=true`), gdy wymagają pobrania prawdziwego zbioru danych/modelu, i
powinny być uruchamiane lokalnie przed otwarciem PR-a. Użyj `make test`
(albo `uv run pytest`) dla pełnego lokalnego zestawu testów, albo
`CI=true uv run pytest` dla szybkiego podzbioru bez sieci.

Utrzymanie samej struktury dwujęzycznej (konwencja plików, mechanizm
kontroli równoważności PL/EN, lista walidacji przed PR-em) jest
opisane w [`CONTRIBUTING.md`](CONTRIBUTING.md) (dokument tylko po
angielsku, analogicznie do `data/README.md`/`third_party/README.md`).

## Licencja

- Oryginalny kod źródłowy: **MIT** — zobacz [`LICENSE`](LICENSE).
- Oryginalne treści dydaktyczne (tekst wykładów, materiały laboratoryjne,
  oryginalne diagramy): **CC BY-SA 4.0** — zobacz
  [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md).
- Materiały zewnętrzne (third-party) pozostają na swoich oryginalnych
  licencjach — zobacz [`third_party/SOURCES.yml`](third_party/SOURCES.yml).

Pełne rozwiązania laboratoryjne nie są publikowane w tym repozytorium.
