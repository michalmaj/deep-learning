Język: [English](lecture.en.md) | **Polski**

# Wykład 15 — od modelu w notebooku do systemu, który można obserwować i utrzymywać

## Cele wykładu

1. Umieć zapisać model jako **artefakt** (wagi + architektura + konfiguracja
   + metadane) i odtworzyć go w osobnym procesie z **identycznymi**
   predykcjami.
2. Rozumieć training-serving skew i wiedzieć, co konkretnie się psuje, gdy
   preprocessing/kolejność klas/`eval()` nie zgadzają się między treningiem
   i inferencją.
3. Umieć zaprojektować minimalny kontrakt API (health/ready/predict) i
   poprawnie zwalidować wejście.
4. Umieć precyzyjnie zmierzyć i zinterpretować latency (p50/p95),
   throughput i koszt pamięci — odróżniając cold od warm, batch od
   pojedynczego żądania.
5. Rozumieć różnicę między monitoringiem systemowym a monitoringiem danych/
   jakości — i wiedzieć, czego **nie można** zmierzyć bez etykiet.
6. Rozumieć różnicę między data drift, label shift i concept drift — i
   wiedzieć, że wykrycie driftu **nie jest** dowodem spadku jakości.

## Dlaczego to jest ważne

Od modułu 6 trenujemy i oceniamy modele — ale **ocena offline** (moduł 14:
accuracy, macro-F1, Grad-CAM) i **działający system** to dwa różne pytania.
Model, który ma świetną accuracy w notebooku, może zawieść jako system, jeśli:
inny proces nie może odtworzyć jego predykcji (training-serving skew);
usługa jest zbyt wolna albo niedostępna; nikt nie zauważa, że napływające
dane już dawno przestały wyglądać jak dane treningowe. Ten wykład **nie**
wprowadza nowej architektury sieci — wprowadza **dyscyplinę inżynierską**
potrzebną, żeby model faktycznie działał poza notebookiem.

**Cztery różne warunki, żadny nie gwarantuje pozostałych:**

1. **Działanie kodu** — usługa odpowiada, nie rzuca wyjątków.
2. **Poprawność predykcji** — model faktycznie zwraca to, co powinien.
3. **Niezawodność usługi** — dostępność, obsługa błędnych wejść, latency.
4. **Stabilność jakości w czasie** — predykcje są nadal trafne, gdy dane się
   zmieniają.

Model może działać poprawnie, podczas gdy API jest niedostępne. API może
działać poprawnie, podczas gdy model systematycznie się myli. To są
**niezależne** osie oceny.

### Powiązania z wcześniejszymi modułami

- **Moduł 5**: walidacja i data leakage — tu: leakage **czasowy**
  (dane monitoringu "z przyszłości" użyte do decyzji o przeszłości) i
  leakage **konfiguracyjny** (próg driftu dobrany po zobaczeniu wyniku).
- **Moduł 7**: pretrained weights i ich preprocessing — tu: ten **sam**
  preprocessing musi działać identycznie w treningu i w serwisie.
- **Moduł 11**: koszt inferencji, cold start, latency — tu: ta sama
  dyscyplina pomiaru, zastosowana do lokalnego HTTP API, nie do LLM.
- **Moduł 14**: analiza błędów, confidence, kalibracja — tu: te same
  pojęcia, zastosowane do **monitorowania w czasie**, nie jednorazowej
  oceny.

## Od notebooka do artefaktu modelu

**Artefakt** to coś więcej niż wagi. Musi jednoznacznie określać:

- architekturę (albo jej jawną konfigurację, nie tylko nazwę klasy w
  pamięci procesu, która akurat istniała podczas treningu);
- parametry (`state_dict`);
- **wersję** modelu;
- **kolejność i nazwy klas** (indeks $\to$ etykieta — to osobna informacja
  od samych wag);
- rozmiar i liczbę kanałów wejścia;
- preprocessing (dokładne kroki, nie tylko "znormalizowane");
- wersję bibliotek potrzebnych do inferencji;
- pochodzenie modelu (jakie dane, kiedy, jaki wynik offline).

❌ "Wystarczy zapisać cały obiekt modelu (`torch.save(model)`), żeby mieć
kompletny artefakt."

✅ **Nie preferujemy tego.** Serializacja całego obiektu Pythona (przez
`pickle`) wiąże artefakt z **konkretną** strukturą klasy w konkretnym
środowisku (ścieżki modułów, wersje bibliotek) i jest **niebezpieczna** do
wczytania z niezaufanego źródła (odtworzenie obiektu może wykonać dowolny
kod). Preferujemy **`state_dict`** (słownik nazwa parametru → tensor, bez
logiki klasy) **plus osobne, czytelne metadane** (JSON) — i jawne
budowanie architektury w kodzie serwisu, nie z zserializowanego obiektu.

**Bezpieczne wczytywanie.** Od PyTorch 2.6, `torch.load(..., weights_only=
True)` jest **domyślnym** zachowaniem i ogranicza odtwarzane typy do
bezpiecznej listy (tensory, podstawowe typy) — zapobiega to wykonaniu
dowolnego kodu przy wczytywaniu artefaktu z niezaufanego źródła. Jawne
podanie `weights_only=True` (niezależnie od wersji) jest dobrą praktyką,
nie tylko "domyślnym ustawieniem, które można zignorować".

### Eksperyment zgodności: artefakt bit-exact

1. Trenujemy mały klasyfikator, liczymy predykcję dla ustalonego wejścia.
2. Zapisujemy `state_dict` + metadane (JSON).
3. **Nowa instancja** modelu (symulacja osobnego procesu) — budujemy
   architekturę **od zera**, wczytujemy wagi.
4. Liczymy predykcję dla **tego samego** wejścia.
5. Porównujemy logity.

Zweryfikowane programowo: `torch.equal(logits_before, logits_after)` —
**dokładnie** (nie `allclose`, nie w tolerancji — literalna identyczność
bitowa). To jest **zasłużone** użycie słowa "bit-exact" (w odróżnieniu od
przypadku znalezionego w przeglądzie modułów 08-10, gdzie "bit-exact" było
błędnie użyte dla wyniku z `atol>0`) — tu nie ma żadnej operacji
niedeterministycznej między zapisem i odtworzeniem, więc identyczność jest
literalna.

### Co może się zepsuć (krótkie, zmierzone przykłady zagrożeń)

❌ "Jeśli wagi się wczytały bez błędu, model jest odtworzony poprawnie."

✅ **Nieprawda — wczytanie bez błędu nie jest dowodem poprawności.**
Zmierzone przykłady:

- **Zapominanie `eval()`.** Na naszym klasyfikatorze (z warstwą `Dropout`
  w głowicy, tak jak w module 5), `model.train()` vs `model.eval()`:
  maksymalna różnica logitów $=1{,}81$ — **spory, nietrywialny** przesunięcie
  (dla tego konkretnego obrazu `argmax` **akurat** się zgodził, ale logity
  są wyraźnie inne — dla innego obrazu/modelu ten sam mechanizm może
  **już** zmienić predykowaną klasę, nie tylko jej pewność). `Dropout`
  losowo zeruje część aktywacji **tylko** w trybie `train()`. **Jeszcze
  bardziej dramatyczny** przypadek: dla zamrożonego backbone'u ResNet18
  (wszystkie wagi `requires_grad=False`!, laboratorium) w trybie `train()`,
  `BatchNorm` używa statystyk **aktualnego batcha**, nie zamrożonych
  running stats — tam różnica logitów
  $=1{,}04$, `argmax` kompletnie inny. W obu przypadkach: **zamrożone wagi
  nie oznaczają zamrożonego zachowania** całej sieci.
- **Niezgodna kolejność klas.** Indeks predykcji (`argmax`) i wartość
  logitu są **identyczne** niezależnie od tego, jak metadane mapują indeks
  na nazwę klasy — jeśli lista klas w metadanych ma inny porządek niż
  podczas treningu, **tylko etykieta tekstowa** jest błędna. Model
  "zgadł" poprawnie (poprawny indeks) — raportujemy złą nazwę.
- **Niezgodny preprocessing (training-serving skew).** Rozwinięte w pełni
  w laboratorium — tu krótko: ręcznie odtworzony `resize`/`crop`/
  normalizacja, niezgodny z `weights.transforms()`, dał maksymalną różnicę
  logitów $3{,}67$ na losowym wejściu testowym, i zmienił predykowaną
  klasę dla **1 z 20** prawdziwych obrazów testowych (reszta: ta sama
  klasa, przesunięte logity) — **uczciwy** wynik, nie każdy przykład jest
  blisko granicy decyzyjnej, ale efekt jest realny i mierzalny.
- **Odtworzone tylko wagi, nie konfiguracja.** Jeśli architektura jest
  zbudowana z domyślnymi/zgadywanymi parametrami (np. inna liczba klas w
  ostatniej warstwie), wczytanie `state_dict` **albo** rzuci błąd
  niezgodności wymiarów (bezpieczny przypadek), **albo** — gorzej — jeśli
  przez przypadek wymiary się zgadzają, wczyta się bez błędu i da
  **cichy, błędny** wynik.

## API i granica systemu

**Minimalny kontrakt:**

- `GET /health` — czy proces żyje (nie mówi nic o modelu).
- `GET /ready` — czy model jest wczytany i gotowy do predykcji.
- `POST /predict` — pojedyncza predykcja.
- `GET /model-info` — wersja modelu, lista klas.

**Wybór frameworka: FastAPI + Uvicorn** (po prototypie — brief dopuszcza
to podejście, jeśli prostota/jakość uzasadniają zależność). Zweryfikowane:
rozwiązuje się czysto z istniejącym `torch`/Python 3.12; `TestClient`
(oparty na `httpx`) pozwala wywołać API **w procesie notebooka**, bez
uruchamiania osobnego serwera — eliminuje niestabilność związaną z
zarządzaniem procesami w Jupyterze, przy zachowaniu **rzeczywistego**
przejścia przez warstwę HTTP/walidacji.

### Wejście, wyjście, walidacja

Kontrakt wejścia dla klasyfikatora obrazów: format pliku, maksymalny
rozmiar, liczba kanałów, **ten sam** preprocessing co podczas ewaluacji.
Odpowiedź: predykowana klasa (nazwa, nie tylko indeks), confidence, wersja
modelu.

„Maksymalny rozmiar” wymaga kilku zabezpieczeń. Trzeba ograniczyć żądanie
HTTP przed parsowaniem multipart, wyegzekwować dokładny limit
skompresowanego pliku i ograniczyć liczbę pikseli po dekodowaniu. Mały
skompresowany obraz może w przeciwnym razie rozwinąć się do dużej alokacji.
Demo w notebooku ogranicza odczyt pliku i liczbę zdekodowanych pikseli,
a laboratorium dodaje middleware ASGI odrzucający zbyt duże żądanie przed
zapisaniem multipart na dysku.

❌ "Błędne wejście (uszkodzony plik, złe wymiary) powinno zwrócić surowy
komunikat wyjątku Pythona, żeby klient wiedział, co się stało."

✅ **Nieprawda — to wyciek informacji wewnętrznych.** Surowy traceback może
ujawnić lokalne ścieżki plików, strukturę kodu, wersje bibliotek — klient
API powinien dostać **jasny, ale ograniczony** komunikat błędu (np. kod
HTTP 400 "nieprawidłowy format obrazu"), nie pełny stos wywołań.

Tylko oczekiwane błędy wejścia i dekodowania są błędami klienta. Ogólny
handler zwracający HTTP 400 ukrywa awarie modelu i infrastruktury, które
powinny pozostać błędami HTTP 500. Synchroniczną inferencję obciążającą CPU
należy też uruchomić w wątku roboczym, aby endpoint `async` nie blokował
pętli zdarzeń.

### Cykl życia modelu w procesie

Model jest wczytywany **raz**, przy starcie procesu — nie trenowany ani
wczytywany od nowa dla każdego żądania. Rozdzielenie: przygotowanie
artefaktu (offline, raz) → start usługi (wczytanie modelu raz) → obsługa
żądania (tylko forward pass) → monitoring (osobny, równoległy proces
obserwacji).

❌ "Usługa HTTP, która poprawnie odpowiada na żądania, jest 'production-
ready'."

✅ **Nieprawda — to jest lokalny prototyp.** Obsługa HTTP to **jeden**
warunek z czterech (wyżej) — nie mówi nic o niezawodności pod obciążeniem,
bezpieczeństwie, monitorowaniu w czasie, czy procedurze aktualizacji.

## Matematyka i pomiar kosztu inferencji

**Latency** pojedynczego żądania = czas od wysłania żądania do otrzymania
odpowiedzi. Dla $n$ pomiarów $\ell_1,\ldots,\ell_n$: średnia
$\bar\ell=\frac1n\sum_i \ell_i$; **p50** (mediana) i **p95** — wartości, pod
którymi leży odpowiednio 50%/95% pomiarów.

**Uwaga, zweryfikowana.** `torch.median()` i `torch.quantile(x, 0.5)` **nie
muszą się zgadzać** dla parzystej liczby pomiarów — `median()` zwraca
**dolny** z dwóch środkowych elementów (konwencja "lower"), `quantile`
domyślnie **interpoluje** liniowo między nimi. Na przykładzie 20 pomiarów
($1,\ldots,20$ ms): `median()`$=10{,}0$, `quantile(0.5)`$=10{,}5$ — różne
liczby, różne konwencje, nie błąd.

**Przykład ręczny** (ten sam zestaw, p95): rangę $q=0{,}95$ w $n=20$
pomiarach odpowiada pozycji (0-indeksowanej, interpolacja liniowa)
$r=q\cdot(n-1)=18{,}05$ — między 19. i 20. wartością sortowaną:
$19+0{,}05\cdot(20-19)=19{,}05$ms. Zweryfikowane programowo —
`torch.quantile` zgadza się dokładnie z tym ręcznym wzorem.

**Throughput** = liczba żądań/obrazów na sekundę. Dla pojedynczych żądań:
$1000/\bar\ell$ (jeśli $\bar\ell$ w ms). **Dla batcha**: throughput liczymy
z **całkowitego** czasu batcha podzielonego przez **liczbę elementów w
batchu** — to jest **inna** liczba niż "czas pojedynczego żądania", i nie
wolno ich mylić.

❌ "Większy batch zawsze daje lepszy throughput (więcej elementów na
sekundę) — to jest ogólna prawda o inferencji na CPU."

✅ **Nieprawda — zmierzone, a wynik zależy od architektury, nie jest
jedną uniwersalną liczbą.** Na naszym `SmallCNN` (notebook tego wykładu,
Fashion-MNIST): czas per-element **spada** z większym batchem
($batch{=}1\to0{,}107$ms; $batch{=}8\to0{,}033$ms; $batch{=}32\to
0{,}026$ms) — tu batching **pomaga**, jak intuicja by podsuwała. **Ale**
na ResNet18 (laboratorium, ten sam CPU, `torch.set_num_threads(4)`,
wynik powtórzony 3× dla potwierdzenia): czas per-element **rośnie** z
większym batchem ($batch{=}1\to5{,}86$ms; $batch{=}8\to5{,}62$ms lekko
lepiej; $batch{=}32\to12{,}6$–$13{,}5$ms **wyraźnie gorzej**) — tam
batching **szkodzi**. Liczba parametrów/FLOP-ów **nie jest** wystarczającym
predyktorem czasu inferencji (zobacz też moduł 7: MobileNetV3-Small miało
mniej FLOP-ów, ale było **wolniejsze** na CPU niż ResNet18) — efekt
batchingu **zależy od konkretnej architektury i sprzętu**, trzeba
**zmierzyć**, nie zakładać w żadną stronę.

### Pomiary eksperymentalne (zmierzone, metodologia jawna)

- **Cold vs warm (nasz `SmallCNN`)**: 1. wywołanie po wczytaniu modelu
  ($0{,}162$ms) vs kolejne 30 wywołań (mean$=0{,}103$ms, p50$=0{,}102$ms,
  p95$=0{,}114$ms) — **efekt cold-start jest tu skromny**, nie dramatyczny
  (inaczej niż np. przy CUDA context na GPU). Na ResNet18 (laboratorium,
  model dużo większy): cold $=6{,}34$ms, warm mean$=5{,}91$ms — też skromny
  efekt względnie, ale w absolutnych liczbach model jest **$\sim$60×
  wolniejszy** niż `SmallCNN` — rozmiar modelu wpływa na **skalę** czasu,
  nie tylko na relację cold/warm.
- **Antyprzykład — ponowne wczytywanie modelu przy każdym żądaniu**: dla
  `SmallCNN`, $0{,}57$ms/żądanie (wczytanie + inferencja) vs $0{,}10$ms/
  żądanie (model wczytany raz) — $5{,}5\times$ wolniej. Dla ResNet18
  (laboratorium, więcej parametrów do wczytania): $103{,}82$ms vs
  $5{,}91$ms — $17{,}6\times$ wolniej. **Ten sam mechanizm, inna skala
  kary** — zawsze szkodliwy, ale bardziej, gdy model jest większy.

**Metodologia** (jawnie raportowana, nie domyślna): liczba prób (10-30
powtórzeń per konfiguracja), sprzęt (ten komputer, CPU), liczba wątków
(`torch.set_num_threads(4)`), warm-up (pierwsze wywołanie odrzucone z
percentyli, liczone osobno jako "cold").

❌ "P95 policzony na 3-5 pomiarach jest wiarygodnym oszacowaniem ogona
rozkładu latencji."

✅ **Nieprawda.** Percentyl wysokiego rzędu (p95, p99) wymaga **wielu**
obserwacji, by być stabilny — z 3-5 pomiarów, "p95" to w praktyce po prostu
maksimum albo drugie-największe, nie prawdziwy percentyl rozkładu.

## Monitoring: dwa niezależne poziomy

### A. Monitoring systemowy

Liczba żądań, czas odpowiedzi, błędy HTTP, nieprawidłowe wejścia,
dostępność, wersja modelu obsługująca żądanie, rozmiar wejścia.

❌ "Awaria usługi i błędna predykcja modelu to ten sam typ problemu."

✅ **Nieprawda — to różne poziomy.** Usługa może **działać** (odpowiada,
kod 200) i zwracać **systematycznie błędne** predykcje — monitoring
systemowy (dostępność, czas odpowiedzi) **nie wykryje** tego samodzielnie.

### B. Monitoring danych i jakości

Rozkład cech wejścia, rozkład predykcji, rozkład confidence, podstawowe
statystyki obrazu (np. jasność) — **bez** potrzeby etykiet. Accuracy/F1 —
**tylko** gdy dostępne są prawdziwe etykiety (zwykle z opóźnieniem).

❌ "Możemy obliczyć bieżącą accuracy modelu na produkcji, patrząc na
rozkład confidence predykcji."

✅ **Nieprawda — to błąd kategorii.** Confidence jest wyjściem **modelu**,
nie porównaniem z prawdą. **Bez etykiet, bieżącej accuracy nie da się
wiarygodnie obliczyć** — można jedynie monitorować **proxy** sygnały
(rozkład wejść, rozkład predykcji, rozkład confidence), które **mogą**
(nie muszą) korelować ze zmianą jakości.

❌ "Wzrost średniego confidence oznacza, że model stał się lepszy."

✅ **Nieprawda — to inny sygnał.** Confidence może wzrosnąć z wielu
przyczyn niezwiązanych z rzeczywistą poprawnością (np. dane napływające są
"łatwiejsze", albo model jest źle skalibrowany w nowym reżimie — moduł 14).

**Zasada prywatności**: logi monitoringu **nie** powinny domyślnie zapisywać
surowych obrazów/tekstów/danych potencjalnie wrażliwych — zapisujemy
**statystyki** (jasność, rozmiar, rozkład predykcji), nie same dane
wejściowe, jeśli nie jest to jawnie potrzebne i uzasadnione.

## Matematyka data drift

**Trzy różne zjawiska, nie synonimy — mogą współwystępować:**

- **Covariate shift / data drift**: zmienia się rozkład $P(X)$ wejść.
- **Label shift**: zmienia się rozkład $P(Y)$ klas.
- **Concept drift**: zmienia się $P(Y\mid X)$ — ta sama cecha wejścia
  odpowiada teraz **innej** prawdziwej etykiecie.

**Population Stability Index (PSI)** — wybrana metoda (prosta, szeroko
używana w monitoringu produkcyjnym; KS/JS divergence — wspomniane jako
alternatywy, nie implementowane dodatkowo, żeby nie budować katalogu
detektorów). Dla cechy podzielonej na biny $i=1,\ldots,k$, z proporcjami
referencyjnymi $p_i$ i aktualnymi $q_i$:

$$\text{PSI} = \sum_{i=1}^k (q_i - p_i) \ln\frac{q_i}{p_i}$$

**Przykład ręczny** (2 biny, ref$=[0{,}5,0{,}5]$, cur$=[0{,}7,0{,}3]$):

$$\text{PSI} = 0{,}2\ln(1{,}4) + ({-}0{,}2)\ln(0{,}6) = 0{,}0673+0{,}1022 = 0{,}1695$$

Zweryfikowane programowo — dokładnie ta liczba. Identyczne rozkłady dają
PSI$=0$ (dokładnie).

**Zerowe częstości.** Bin z zerową liczbą próbek w jednym z rozkładów daje
$\ln(0)=-\infty$ — zweryfikowane programowo: **bez** wygładzania, PSI
dosłownie `inf`. Rozwiązanie: wygładzanie (np. $\max(q_i,\varepsilon)$ dla
małego $\varepsilon$) — **nie arbitralne**, tylko konieczne, żeby liczba
była w ogóle obliczalna; zmierzone: z $\varepsilon=10^{-4}$, ten sam
przypadek daje duże, ale **skończone** PSI ($4{,}60$), nie "naprawiony"
wynik udający, że problemu nie było.

**Konwencjonalne progi** (heurystyka branżowa, **nie** gwarancja
statystyczna): PSI$<0{,}1$ brak istotnej zmiany; $0{,}1$–$0{,}25$
umiarkowana; $>0{,}25$ istotna.

### Ograniczenia PSI (i miar tego typu w ogólności)

- **Zależność od liczności prób** — mało próbek daje niestabilne proporcje
  binów, podatne na przypadkowe wahania.
- **Wrażliwość na podział histogramu** — inna liczba/granice binów mogą
  dać inny wynik dla **tych samych** danych.
- **Problem wielowymiarowości** — PSI per-cecha nie wykrywa zmiany
  **współzależności** między cechami (jedna cecha może wyglądać stabilnie,
  mimo że łączny rozkład się zmienił).
- **Możliwość false alarms** — niewielka, nieszkodliwa fluktuacja może
  przekroczyć próg, zwłaszcza przy małej próbie.
- **Brak gwarancji degradacji jakości** — PSI mierzy **rozkład wejść**, nie
  $P(Y\mid X)$ — wykrycie driftu wejść nie mówi, czy model wciąż dobrze
  radzi sobie z **nowymi** wejściami.

❌ "Wykrycie data drift jest dowodem, że accuracy modelu spadła — albo:
siła sygnału PSI jest proporcjonalna do spadku accuracy."

✅ **Nieprawda — zmierzone bezpośrednio w notebooku (Eksperyment D), na
dwóch różnych siłach przesunięcia jasności, nie jednej wybranej pod
tezę.** **Umiarkowane** przesunięcie ($+0{,}03$): PSI$=0{,}1317$ (drift
wykryty, poziom "umiarkowany"), accuracy **dokładnie niezmieniona**
($0{,}8700\to0{,}8700$). **Silniejsze** przesunięcie ($+0{,}15$):
PSI$=2{,}1316$ (drift silnie wykryty), accuracy **zauważalnie spadła**
($0{,}8700\to0{,}7800$). Oba wyniki z **tego samego** eksperymentu, **nie**
wybrane selektywnie — pokazują, że zależność między siłą driftu i spadkiem
jakości **nie jest ustalona z góry**: czasem drift nie szkodzi, czasem
szkodzi wyraźnie — trzeba **zmierzyć obie wielkości**, nie wnioskować
jednej z drugiej.

❌ "Brak alarmu driftowego dla wybranej cechy oznacza, że model działa
stabilnie."

✅ **Nieprawda.** Brak zmiany w **jednej, monitorowanej** cechy (np.
jasności) nie wyklucza zmiany w innych wymiarach danych, ani concept
driftu (zmiany $P(Y\mid X)$ bez zmiany $P(X)$) — monitoring jednej cechy
jest **jednym** sygnałem, nie pełną gwarancją.

## Co obserwujemy

Wszystkie liczby pochodzą z rzeczywistego wykonania `examples.ipynb` (CPU,
ten komputer) — Fashion-MNIST, `SmallCNN`.

- **Eksperyment A.** Artefakt zapisany i odtworzony w nowej instancji
  modelu: logity **dokładnie** identyczne (`torch.equal`). Zapominanie
  `eval()` (na modelu z `Dropout`) daje zmierzoną, nietrywialną różnicę
  logitów ($1{,}81$; dla tego obrazu `argmax` akurat się zgodził, ale
  przesunięcie logitów jest realne) — opisane wyżej, z dodatkowym, jeszcze
  bardziej dramatycznym przypadkiem `BatchNorm` z laboratorium (różnica
  $1{,}04$, `argmax` kompletnie inny).
- **Eksperyment B.** Cold $0{,}162$ms vs warm mean $0{,}103$ms
  (p50$=0{,}102$, p95$=0{,}114$); batch=1→$0{,}107$ms/elem,
  batch=8→$0{,}033$ms/elem, batch=32→$0{,}026$ms/elem (**lepiej** —
  kontrast z ResNet18 w laboratorium, gdzie większy batch **szkodzi**);
  ponowne wczytywanie modelu per-request: $5{,}5\times$ wolniej niż model
  wczytany raz ($17{,}6\times$ dla większego ResNet18 w laboratorium).
- **Eksperyment C.** Predykcja przez `TestClient` (prawdziwa warstwa HTTP/
  walidacji, pełny preprocessing z PNG round-trip) zgadza się **dokładnie**
  z bezpośrednim wywołaniem modelu dla tego samego obrazu — ta sama
  przewidziana klasa i confidence. Uszkodzony plik wejściowy: kod
  HTTP $400$, komunikat jasny, bez ujawnienia tracebacka.
- **Eksperyment D.** PSI(ref, kontrola) $=0{,}0225$ (poprawnie niski, brak
  false alarm); PSI(ref, umiarkowane przesunięcie jasności $+0{,}03$)
  $=0{,}1317$ (drift wykryty, poziom "umiarkowany"), accuracy **dokładnie
  niezmieniona** ($0{,}8700\to0{,}8700$); PSI(ref, silniejsze przesunięcie
  $+0{,}15$) $=2{,}1316$ (drift silnie wykryty), accuracy **zauważalnie
  spadła** ($0{,}8700\to0{,}7800$) — **oba** wyniki z tego samego
  eksperymentu, nie wybrane selektywnie pod tezę.
- **Eksperyment E.** Decyzja operacyjna na podstawie D: siła driftu i
  wpływ na jakość **nie są** w prostej relacji — rekomendacja różna dla
  umiarkowanego przesunięcia ("dalszy monitoring, bez alarmu") i silnego
  ("dodatkowa analiza / zbieranie etykiet"), w **żadnym** przypadku
  automatyczny retraining bez niezależnej weryfikacji.

## Wersjonowanie, aktualizacja, rollback

Prosty proces: wykrycie problemu → zebranie danych/etykiet → nowy
kandydat → ewaluacja offline → porównanie z obecnym modelem → **nowa
wersja artefaktu** (nie nadpisanie starej) → test odtworzenia i kontraktu
API → wdrożenie → monitoring → **rollback**, jeśli regresja.

❌ "Rollback modelu to po prostu przywrócenie starego pliku wag."

✅ **Nieprawda, jeśli preprocessing/mapowanie klas zmieniło się w
międzyczasie.** Rollback wag **bez** zgodnego rollbacku metadanych
(kolejność klas, preprocessing) może dać **inną**, nową formę training-
serving skew — rollback musi przywracać **cały** artefakt (wagi +
metadane), nie tylko wagi.

## Bezpieczeństwo, niezawodność, prywatność (podstawy)

Ograniczenie rozmiaru/formatu wejścia; walidacja payloadu; bezpieczne
wczytywanie artefaktów (`weights_only=True`); brak ujawniania wewnętrznych
wyjątków/ścieżek; unikanie logowania surowych danych użytkownika i
sekretów; model wczytywany raz, nie pobierany przy każdym żądaniu;
zgodność licencji danych/wag z zastosowaniem.

**Ten lokalny demonstrator nie ma uwierzytelniania ani rate limiting** —
uruchamiamy go **lokalnie**, nie wystawiamy do publicznego internetu. Pełna
autoryzacja/ograniczanie liczby żądań to realne zagadnienia produkcyjne,
poza zakresem tego modułu.

## Częste błędne przekonania

(Zbiorcze podsumowanie — każde ma pełny ❌/✅ blok wyżej)

1. Działanie kodu, poprawność predykcji, niezawodność usługi i stabilność
   jakości w czasie to **cztery niezależne** warunki.
2. Wczytanie wag bez błędu nie jest dowodem poprawnego odtworzenia modelu.
3. Zamrożone wagi nie oznaczają zamrożonego zachowania (`BatchNorm` w
   trybie `train()`).
4. Większy batch nie gwarantuje lepszego throughput — trzeba zmierzyć.
5. P95 z kilku pomiarów nie jest wiarygodnym oszacowaniem ogona rozkładu.
6. Bez etykiet nie da się wiarygodnie policzyć bieżącej accuracy.
7. Wzrost confidence nie oznacza automatycznie lepszej jakości.
8. Data drift, label shift i concept drift to różne zjawiska, mogą
   współwystępować, żadne nie implikuje automatycznie pozostałych.
9. Wykrycie driftu nie jest dowodem spadku accuracy; brak alarmu nie jest
   dowodem stabilnej jakości.
10. Jeden alarm monitoringowy nie powinien automatycznie uruchamiać
    retrainingu/wdrożenia — wymaga diagnozy.
11. Rollback bez zgodnego preprocessingu/mapowania klas też może być błędny.
12. Lokalny prototyp HTTP nie jest automatycznie "production-ready".

## Implikacje inżynierskie

- **Artefakt to kontrakt**, nie tylko plik wag — zmiana w jednym miejscu
  (np. kolejność klas) bez propagacji do wszystkich konsumentów artefaktu
  jest źródłem cichych błędów.
- **Koszt inferencji zależy od sprzętu/konfiguracji wątków/batch size** w
  sposób, którego nie przewidzi liczba parametrów — mierzyć, nie zakładać.
- **Monitoring systemowy i monitoring jakości odpowiadają na różne
  pytania** — oba potrzebne, żaden nie zastępuje drugiego.
- **Drift jest sygnałem do diagnozy, nie wyrokiem** — wymaga dalszej
  analizy (etykiety, porównanie z baseline), nie automatycznej reakcji.
- **Wersjonowanie artefaktu (nie tylko wag) umożliwia bezpieczny rollback.**
- **Prywatność i bezpieczeństwo zaczynają się od tego, czego NIE logujemy**
  — nie tylko od tego, co zabezpieczamy.

## Co warto zapamiętać

- Artefakt = wagi (`state_dict`) + architektura + metadane (klasy,
  preprocessing, wersja) — zapisane i odtworzone, dają **bit-exact** te
  same predykcje, zweryfikowane `torch.equal`.
- `model.eval()` nie jest formalnością — dla sieci z `BatchNorm`/`Dropout`,
  różnica względem `train()` może być dramatyczna, nawet z zamrożonymi
  wagami.
- Latency (p50/p95), throughput i pamięć wymagają jawnej metodologii
  (liczba prób, sprzęt, wątki, warm-up) — bez tego liczby są bezwartościowe.
- Monitoring bez etykiet: rozkład wejść/predykcji/confidence. Monitoring z
  etykietami (opóźnionymi): rzeczywista accuracy/F1 — to są **różne**
  poziomy pewności.
- PSI (albo KS/JS) wykrywa zmianę rozkładu wejść — **nie** dowodzi zmiany
  jakości modelu; oba muszą być zmierzone **osobno**.
- Decyzja operacyjna (brak działania / dalsza analiza / retraining /
  rollback) wymaga **diagnozy**, nie jednego alarmu.

## Pytania dla studentów

1. Masz dwa artefakty tego samego modelu, różniące się tylko kolejnością
   klas w metadanych. Jak to wykryjesz, mając dostęp tylko do API (nie do
   kodu serwisu)?
2. Dlaczego batch size=32 może dać **gorszy** throughput per-element niż
   batch size=1 na niektórych konfiguracjach CPU — i jak byś to
   zweryfikował/a przed wdrożeniem konkretnego batch size w produkcji?
3. Masz alert PSI powyżej progu dla jednej monitorowanej cechy. Jakie
   **trzy** kroki wykonałbyś/abyś, zanim zdecydujesz o retrainingu?
4. Wyjaśnij, czym różni się covariate shift od concept driftu — i czy
   monitoring rozkładu wejść (bez etykiet) może wykryć drugi z nich.
5. Masz model, który po rollbacku do poprzedniej wersji wag nadal daje
   błędne wyniki. Jakie dwie przyczyny (poza samymi wagami) sprawdziłbyś/
   abyś najpierw?

## Bibliografia

- Sculley, D., i in. (2015). *Hidden Technical Debt in Machine Learning
  Systems*. NeurIPS (training-serving skew, dług technologiczny w ML).
- Paszke, A., i in. (2019). *PyTorch: An Imperative Style, High-Performance
  Deep Learning Library*. NeurIPS (kontekst `state_dict`/serializacji).
- PyTorch documentation: *Saving and Loading Models*,
  `https://pytorch.org/tutorials/beginner/saving_loading_models.html`.
- PyTorch documentation: `torch.load` (`weights_only`),
  `https://pytorch.org/docs/stable/generated/torch.load.html`.
- FastAPI documentation: `https://fastapi.tiangolo.com/`.
- Moreno-Torres, J. G., i in. (2012). *A unifying view on dataset shift in
  classification*. Pattern Recognition (rozróżnienie covariate/label/
  concept shift).
- PSI: powszechnie cytowana heurystyka w monitoringu modeli kredytowych/
  ryzyka (np. Siddiqi, N. (2012). *Credit Risk Scorecards*. Wiley) — progi
  $0{,}1$/$0{,}25$ to konwencja branżowa, nie wynik z pracy źródłowej z
  dowodem statystycznym.
