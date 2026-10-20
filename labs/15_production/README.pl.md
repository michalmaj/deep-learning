Język: [English](README.md) | **Polski**

# Laboratorium 15: lokalna usługa inferencyjna z monitoringiem (EuroSAT + ResNet18)

Czas: ok. 120 minut. **Ostatnie laboratorium kursu** — łączy przygotowanie
danych, trening, ewaluację, koszt i interpretowalność z poprzednich modułów
w jednym, praktycznym eksperymencie inżynierskim. Bazuje na
[`lectures/15_production/lecture.md`](../../lectures/15_production/lecture.md),
ale na **innym problemie** niż wykład (tam: Fashion-MNIST od zera; tu:
EuroSAT + ResNet18 zamrożony backbone, wzorem modułów 07/14).

**Cel**: zakończyć laboratorium z **działającym lokalnym API**, możliwością
odtworzenia artefaktu, wynikami benchmarku i krótkim raportem decyzji o
wdrożeniu/utrzymaniu modelu.

## Struktura repozytorium tego modułu

```
labs/15_production/
├── README.md                (ten plik)
├── lab.ipynb                 (notebook spajający wszystko, z TODO)
├── prepare_artifact.py       (gotowy skrypt — trening + zapis artefaktu)
├── app/
│   ├── model.py               (load_artifact GOTOWE, preprocessing+predict TWOJE)
│   ├── schemas.py              (modele odpowiedzi — gotowe)
│   └── server.py               (endpoints health/ready/model-info gotowe, /predict TWOJE)
├── benchmark.py               (wysyłanie żądań gotowe, liczenie statystyk TWOJE)
└── monitoring.py              (strumienie/histogram gotowe, PSI TWOJE)
```

Artefakt (`artifacts/weights.pt` + `artifacts/metadata.json`) jest
generowany **lokalnie** przez `prepare_artifact.py` i **ignorowany przez
Git** (`.gitignore`) — nigdy nie commitujesz wag ani datasetu.

## Dokładne komendy

1. **Środowisko** (jeśli jeszcze nie zrobione): `uv sync`.
2. **Przygotowanie artefaktu** (pobiera EuroSAT + wagi ResNet18 przy
   pierwszym uruchomieniu, ~35s): `uv run python labs/15_production/prepare_artifact.py`.
3. **Testy** (po uzupełnieniu TODO): `uv run pytest tests/test_lab_15_scaffold.py -v`.
4. **Notebook**: `uv run jupyter lab labs/15_production/lab.ipynb` — tam
   budujesz `TestClient`, wysyłasz żądania, uruchamiasz benchmark/monitoring.
5. Lab **nie** wymaga uruchamiania osobnego procesu serwera (`uvicorn ...`)
   — `fastapi.testclient.TestClient` wywołuje tę samą aplikację w procesie
   notebooka, przez prawdziwą warstwę HTTP/walidacji.

## 1. Artefakt modelu (gotowe + weryfikacja)

`prepare_artifact.py` trenuje głowicę (`nn.Linear`) na zamrożonych cechach
ResNet18 (`IMAGENET1K_V1`) i zapisuje `state_dict` + `metadata.json`
(wersja, lista klas, preprocessing, rozmiar wejścia). To jest **scaffold**
— nie Twoje zadanie (trening głowicy to przegląd modułów 07/14, nie nowy
temat).

Podczas ekstrakcji cech obrazy są odczytywane strumieniowo przez
`DataLoader` w małych partiach. Skrypt przechowuje wyekstrahowane tensory
cech, a nie wszystkie zdekodowane obrazy 224x224, dzięki czemu szczytowe
zużycie pamięci przez obrazy jest ograniczone rozmiarem partii.

Generator i loader współdzielą jeden kontrakt artefaktu. Przed wczytaniem
wag `load_artifact()` odrzuca metadane z niezgodną architekturą, mapowaniem
klas, kształtem wejścia lub deklaracją preprocessingu. Zapobiega to
serwowaniu poprawnego `state_dict` z nieaktualnymi albo mylącymi metadanymi.

**Twoje zadanie:** uruchom skrypt, **zweryfikuj programowo** w notebooku,
że model wczytany z artefaktu daje **identyczne** (`torch.equal`) logity co
model bezpośrednio po treningu (ten sam obraz testowy).

## 2. `app/model.py` — preprocessing i inferencja — Twoje zadanie

`load_artifact()` jest gotowe (buduje architekturę, wczytuje wagi +
metadane). Twoje zadanie:

1. `preprocess_image_bytes(data)` — zdekoduj obraz (`PIL.Image`), zastosuj
   **dokładnie** ten preprocessing, który był użyty podczas treningu
   (`ResNet18_Weights.IMAGENET1K_V1.transforms()` — **nie** odtwarzaj go
   ręcznie przez `resize`/`crop`/normalizację, patrz zadanie diagnostyczne
   niżej za to, co się psuje, gdy to zrobisz).
2. `predict(loaded, image_bytes)` — forward pass, softmax, zbuduj słownik
   odpowiedzi (`predicted_class`, `confidence`, `model_version`).

## 3. `app/server.py` — kontrakt API — Twoje zadanie

`/health`, `/ready`, `/model-info` są gotowe. Twoje zadanie: `/predict`:

Gotowa aplikacja używa kontekstu lifespan FastAPI, aby wczytać model raz
przy starcie i zwolnić referencję w procesie podczas zamykania.
`/model-info` zwraca HTTP 503, gdy model nie jest gotowy.

Bezpieczeństwo uploadu ma trzy warstwy:

- `RequestBodyLimitMiddleware` ogranicza całe żądanie `/predict` podczas
  odbierania fragmentów ASGI, zanim Starlette zapisze upload multipart na
  dysku. Limit żądania uwzględnia 64 KiB narzutu multipart ponad limit pliku
  wynoszący 5 MB.
- `read_upload_limited()` egzekwuje dokładny limit pliku 5 MB i odrzuca
  pusty upload.
- `model.decode_image_bytes()` odrzuca obrazy rozwijające się do ponad
  25 milionów zdekodowanych pikseli, nawet gdy skompresowany plik jest mały.

Twój endpoint powinien:

1. Zwrócić HTTP 503, gdy model nie jest gotowy.
2. Odczytać plik przez `read_upload_limited()`.
3. Zwrócić `await run_prediction(...)`. Helper przenosi synchroniczną,
   obciążającą CPU inferencję poza wątek pętli zdarzeń i mapuje wyłącznie
   znane błędy wejścia na bezpieczne odpowiedzi HTTP 400/413. Nieoczekiwane
   awarie modelu lub infrastruktury pozostają błędami HTTP 500.

## 4. Zgodność API z bezpośrednim wywołaniem — Twoje zadanie

Zbuduj `TestClient(app)`, wyślij prawdziwy obraz EuroSAT, porównaj
odpowiedź API z bezpośrednim wywołaniem `model.predict()` na tym samym
obrazie — **musi się zgadzać dokładnie** (ta sama klasa i confidence).

## 5. `benchmark.py` — pomiar kosztu — Twoje zadanie

`send_requests(client, image_bytes, n)` jest gotowe. Mierzy czas zegarem
monotonicznym i odrzuca nieudane odpowiedzi HTTP, więc próbkami benchmarku
są tylko prawidłowe predykcje. Twoje zadanie: `compute_stats(times)` — mean,
p50, p95 (ms), throughput (req/s). **Zmierz**: cold (1. żądanie) vs warm
(kolejne 20+), batch size=1 dla pojedynczych żądań (API przyjmuje jeden
obraz na żądanie — throughput liczony z **wielu sekwencyjnych** żądań, nie
batchingu w jednym żądaniu).

## 6. `monitoring.py` — PSI i kontrolowane strumienie — Twoje zadanie

`histogram_counts()` i `brightness_feature()` są gotowe. Twoje zadanie:
`psi(ref_counts, cur_counts, eps)` — wzór PSI z wykładu, **z wygładzaniem**
zerowych częstości (sprawdź w notebooku, co się stanie bez `eps`). Gotowy
histogram zachowuje wartości poniżej/powyżej stałego zakresu referencyjnego
w pierwszym/ostatnim binie, zamiast odrzucać obserwacje z driftem. `psi()`
waliduje kształty, skończone nieujemne liczniki, dodatnie sumy i `eps`,
zanim zostanie wykonany Twój wzór.

## 7. Scenariusze monitoringu — Twoje zadanie

**Scenariusz A (kontrola)**: niezależna próba z **tej samej** dystrybucji
co referencyjna. Sprawdź: PSI **nie** powinno dać fałszywego alarmu.

**Scenariusz B (kontrolowana zmiana)**: wybierz **jedną**, mierzalną
transformację (np. jasność +X, kontrast ×Y) i **jedną** jej siłę. Sprawdź:
czy PSI wykrywa zmianę?

**Scenariusz C (etykiety z opóźnieniem)**: dla danych ze scenariusza B,
policz accuracy (masz etykiety EuroSAT — symulujesz, że są dostępne "z
opóźnieniem", po fakcie). **Porównaj** z wynikiem PSI ze scenariusza B —
**czy drift i zmiana accuracy idą w tę samą stronę, czy nie?**

❌ "Jeśli PSI wykryło drift w scenariuszu B, accuracy musiała spaść w
scenariuszu C."

✅ **Sprawdź to, nie zakładaj.** Zobacz wykład: na innym zbiorze (Fashion-
MNIST) silny drift **obniżył** accuracy; umiarkowany — nie. Na **Twoich**
danych (EuroSAT) wynik może wyjść **inaczej** — to jest coś, co musisz
**zmierzyć** w tym konkretnym eksperymencie, nie przewidzieć z góry.

**Ważne**: nie dobieraj progu PSI ani siły transformacji **po** zobaczeniu
wyniku accuracy, wyłącznie żeby "pokazać" alarm — wybierz je **przed**
sprawdzeniem accuracy, zgodnie z metodologią ewaluacji z modułu 5/10.

## 8. Zadanie diagnostyczne: training-serving skew

Poniższy, **oddzielnie dostarczony** `model_colleague.py` implementuje
`preprocess_image_bytes` **ręcznie** (resize do 224×224 bez pośredniego
kroku 256→crop, bez użycia `weights.transforms()`), zamiast używać
bibliotecznej transformacji.

**Zadanie:**
1. Uruchom predykcję tym samym obrazem przez **Twój** `model.py` i przez
   `model_colleague.py`. Porównaj logity.
2. Wyjaśnij **dokładnie**, który krok preprocessingu się różni i czemu to
   ma znaczenie (odwołaj się do `ResNet18_Weights.IMAGENET1K_V1.
   transforms()` — co dokładnie robi: `resize(256)` → `center_crop(224)` →
   normalizacja — a czego nie robi ręczna wersja).
3. Sprawdź na **kilku** (np. 20) prawdziwych obrazach EuroSAT: ile
   predykcji **się zmienia** (nie tylko logity)?
4. Potwierdź, że **Twoja** implementacja (sekcja 2) zgadza się z
   bezpośrednim, zweryfikowanym wywołaniem (sekcja 4) — a wersja kolegi nie.

❌ "To jest błąd, który wymaga zgadywania, jak działają hooki/wewnętrzne
mechanizmy biblioteki."

✅ **Nie — to różnica w WIDOCZNYCH krokach preprocessingu**, porównywalna
bezpośrednio (rozmiar obrazu po każdym kroku, wartości pikseli przed/po
normalizacji) — nie wymaga zgadywania.

## 9. Raport: decyzja o wdrożeniu i utrzymaniu modelu — Twoje zadanie

Krótki raport (markdown w notebooku), odpowiadający na:

1. Wersja modelu, źródło danych, protokół ewaluacji offline.
2. Accuracy i najważniejsze ograniczenia (np. z modułu 14-stylu analizy,
   jeśli masz czas — opcjonalnie, nie wymagane ponownie tutaj).
3. Format wejścia/wyjścia API.
4. Latency p50/p95, throughput (Twoje zmierzone liczby).
5. Wyniki scenariuszy A/B/C.
6. Poziom pewności, z jakim interpretujesz wynik driftu (jedna cecha,
   ograniczona liczba próbek — nazwij to explicit).
7. Znane ryzyka, rekomendowane następne działania.

**Nie wymuszaj konkluzji "model nadaje się do produkcji."** Dopuszczalna, a
czasem najlepsza konkluzja: *"system działa technicznie, ale przed
rzeczywistym wdrożeniem potrzeba dodatkowych danych, testów i
zabezpieczeń."*

## Podział czasu (realistyczny, zweryfikowany względem rzeczywistego kosztu CPU)

| Blok | Czas |
|---|---|
| Sekcja 1 (artefakt, weryfikacja) | 8 min |
| Sekcja 2 (`model.py`: preprocessing + predict) | 15 min |
| Sekcja 3 (`server.py`: `/predict`) | 12 min |
| Sekcja 4 (zgodność API vs bezpośrednie wywołanie) | 8 min |
| Sekcja 5 (`benchmark.py`: statystyki) | 10 min |
| Sekcja 6 (`monitoring.py`: PSI) | 10 min |
| Sekcja 7 (scenariusze A/B/C) | 15 min |
| Sekcja 8 (zadanie diagnostyczne) | 15 min |
| Sekcja 9 (raport) | 12 min |

Suma: **105 min** rdzenia + **15-minutowy** bufor (razem 120 min). Koszt
CPU: `prepare_artifact.py` ~35s (jednorazowo, głównie ekstrakcja cech);
pojedyncze żądanie przez pełny stos API ~7-8ms (zmierzone); scenariusze
monitoringu — sekundy. Budżet idzie w implementację i interpretację, nie w
czekanie na model.

## Kryteria oceny

- Artefakt zweryfikowany jako dający **identyczne** predykcje po
  odtworzeniu (`torch.equal`).
- `/predict` poprawnie wywołuje **prawdziwy** preprocessing (biblioteczny,
  nie ręcznie odtworzony) i zwraca poprawną odpowiedź.
- Odpowiedź API zgadza się **dokładnie** z bezpośrednim wywołaniem modelu
  dla tego samego obrazu.
- Błędne wejścia (uszkodzony plik, zbyt duże żądanie/plik lub obraz ze zbyt
  dużą liczbą zdekodowanych pikseli) obsłużone bez wycieku informacji
  wewnętrznych, a awarie serwera nie są błędnie oznaczane jako HTTP 400.
- Benchmark rozróżnia cold/warm, raportuje p50/p95 z jawną metodologią
  (liczba prób).
- PSI poprawnie obsługuje zerowe częstości (wygładzanie, nie `inf`/`nan`).
- Scenariusz A nie generuje fałszywego alarmu; scenariusz B/C **rzeczywiście
  zmierzone**, nie zakładane — wniosek o relacji drift/accuracy oparty na
  **tym** eksperymencie, nie na ogólnej zasadzie.
- Zadanie diagnostyczne poprawnie identyfikuje różnicę preprocessingu i
  zmierzony wpływ na predykcje (nie tylko logity).
- Raport odpowiada na wszystkie punkty, nie wymusza konkluzji "gotowe do
  produkcji".

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/15_production/`) i nie jest częścią tego
repozytorium.
