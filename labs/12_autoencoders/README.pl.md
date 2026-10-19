Język: [English](README.md) | **Polski**

# Laboratorium 12: wykrywanie anomalii za pomocą autoenkodera i analiza ograniczeń rekonstrukcji

Czas: ok. 120 minut. Bazuje na
[`lectures/12_autoencoders/lecture.md`](../../lectures/12_autoencoders/lecture.md)
(AE, VAE, przestrzeń latentna), ale to **inny problem** niż eksperyment
wykładowy: nie generowanie nowych obrazów, a **detekcja anomalii** przez
błąd rekonstrukcji — z naciskiem na to, **kiedy ta metoda zawodzi**, nie
tylko na to, kiedy działa.

**Dane**: Fashion-MNIST (MIT License — zobacz
[`data/registry.yml`](../../data/registry.yml)). **Klasa normalna**:
*Sneaker* (klasa 7) — wybrana, bo to spójna, jednoznaczna kategoria wizualna
(konkretny typ obuwia). **Anomalie**: pozostałe 9 klas Fashion-MNIST
(T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Bag, Ankle
boot) — traktowane jako "wszystko, co nie jest sneakerem", pedagogicznie
uzasadnione: w praktycznych zastosowaniach (np. kontrola jakości na linii
produkcyjnej) "anomalia" często oznacza "coś spoza wąsko zdefiniowanej
kategorii normalnej", nie jedną konkretną, znaną z góry klasę defektu.

❌ "To zadanie klasyfikacji na 10 klas, tylko inaczej nazwane."

✅ **Nie** — model (autoenkoder) **nigdy** nie widzi etykiet klas podczas
treningu, i **nigdy** nie widzi **żadnego** przykładu anomalii podczas
treningu. Trenujemy wyłącznie na obrazach klasy normalnej, bez etykiet —
sam fakt "wykrywania anomalii" musi wynikać z tego, jak dobrze model
rekonstruuje to, czego się nauczył, vs to, czego nie widział.

## 0. Konfiguracja (gotowe)

Import bibliotek, urządzenie, ziarno losowości, wczytanie Fashion-MNIST
(`torchvision.datasets.FashionMNIST`).

## 1. Dane i podział — Twoje zadanie

To jest **najważniejszy** krok metodologiczny tego laboratorium — błąd tutaj
podważa wszystko, co następuje.

Zbuduj **cztery** rozłączne podzbiory, **bez** przecieku informacji między
nimi:

1. **`ae_train`** — obrazy **wyłącznie** klasy normalnej (Sneaker), z
   oryginalnego zbioru treningowego Fashion-MNIST. Autoenkoder widzi
   **tylko to**.
2. **`val_normal`** — **inne** obrazy klasy normalnej, z tego samego
   oryginalnego zbioru treningowego (rozłączne z `ae_train` — podziel zbiór
   treningowy klasy 7 na dwie części, np. `torch.randperm` + slicing).
3. **`val_anomaly`** — próbka obrazów **innych** klas, też z oryginalnego
   zbioru treningowego Fashion-MNIST (np. po równej liczbie z każdej z 9
   klas).
4. **`test_normal`**/**`test_anomaly`** — z **oryginalnego zbioru
   testowego** Fashion-MNIST (osobny, nigdy nie dotykany do tego momentu),
   odpowiednio: wszystkie obrazy klasy 7, i próbka obrazów innych klas.

**Zasada, której nie można złamać w żadnym późniejszym kroku:** `test_*`
**nie może** wpływać na żadną decyzję (architektura, hiperparametry, próg
detekcji) — używamy go **wyłącznie** do końcowej oceny, raz.

❌ "Mogę potem 'douczyć' albo dostosować próg na podstawie wyników na
zbiorze testowym, skoro i tak już go mam."

✅ To jest **dokładnie** błąd metodologiczny, który badamy w zadaniu
diagnostycznym (sekcja 7). Jeśli test wpłynie na jakąkolwiek decyzję, zmierzony
wynik przestaje być wiarygodnym oszacowaniem działania na nowych danych.

**Zadanie (kod):** wypisz rozmiary wszystkich czterech/pięciu podzbiorów i
**sprawdź programowo** (nie "na oko"), że żadne dwa z nich się nie
nakładają (np. przez porównanie indeksów, nie tylko rozmiarów).

## 2. Autoenkoder — architektura i trening — Twoje zadanie

Zbuduj mały AE (MLP, wzorem wykładu): `encoder` do wymiaru latentnego $d_e$,
`decoder` z powrotem do $28\times 28$, strata MSE.

**Zadanie:** zaimplementuj klasę `AE(nn.Module)` i funkcję treningową.
Wytrenuj **wyłącznie** na `ae_train` (nigdy na `val_*` czy `test_*`).
Zacznij od $d_e=16$ — to jest Twój **główny** model dla sekcji 3-7.

**Uwaga o koszcie**: ten trening jest **bardzo** szybki na CPU (ułamek
sekundy do kilku sekund, zależnie od liczby epok/rozmiaru zbioru) — budżet
czasowy tego laboratorium idzie w **analizę**, nie w czekanie na trening.

## 3. Błąd rekonstrukcji i rozkłady normalne vs anomalie — Twoje zadanie

Zaimplementuj `per_image_error(model, X)` — błąd rekonstrukcji **per obraz**
(nie zagregowany po całym zbiorze): `((x_hat - x)**2).flatten(1).mean(dim=1)`.

Policz ten błąd dla `val_normal` i `val_anomaly`. **Zwizualizuj** oba
rozkłady (histogram albo nakładające się wykresy gęstości) na jednym
wykresie. Policz średnią/odchylenie standardowe/medianę dla każdego z nich.

❌ "Jeśli oba rozkłady się w ogóle nakładają, metoda jest bezużyteczna."

✅ Częściowe nakładanie się jest **normalne** i oczekiwane — pytanie nie
jest "czy rozkłady się nakładają", ale "jak dobrze **rozdzielają** się przy
**konkretnym** wyborze progu" (sekcja 4) i "co się dzieje w regionie
nakładania" (sekcja 6, analiza błędów).

## 4. Wybór progu — Twoje zadanie

Ustaw próg detekcji **wyłącznie** na podstawie rozkładu błędu `val_normal`
(np. średnia + $k\times$ odchylenie standardowe, dla wybranego przez Ciebie
$k$ — uzasadnij wybór). **Nie** używaj `val_anomaly` do wyboru progu — to
jest jawnie **walidacyjnie nadzorowana** procedura (masz etykiety w
walidacji i decydujesz na ich podstawie o architekturze/hiperparametrach
ogólnie), ale wybór **konkretnej wartości progu** oparty tylko na
rozkładzie klasy normalnej jest bliższy praktycznemu scenariuszowi, w
którym anomalii nie masz w ogóle podczas konfiguracji systemu.

❌ "To jest w pełni nienadzorowana (*unsupervised*) detekcja anomalii."

✅ **Nieprawda, nazwij to precyzyjnie.** Masz dostęp do etykiet w całym
procesie przygotowania danych (wiesz, które obrazy są "normalne") — to
**walidacyjnie nadzorowana** procedura, nie w pełni nienadzorowana. Prawdziwie
nienadzorowana detekcja anomalii (bez żadnych etykiet, nawet dla "normalnych"
danych) to inne, trudniejsze zadanie, poza zakresem tego laboratorium.

## 5. Metryki na zbiorze testowym — Twoje zadanie

**Najpierw, na małym przykładzie z znanym wynikiem** (zanim użyjesz
prawdziwych danych): dla `scores = [0.1,0.2,0.3,0.4,0.9, 0.5,0.6,0.7,0.8,1.0]`,
`labels = [0,0,0,0,0, 1,1,1,1,1]` (pierwsza piątka normalna, druga anomalia),
próg $=0{,}45$ — **policz ręcznie** (na papierze albo w komentarzu) TP/FP/
TN/FN, precision/recall/F1, **potem** zweryfikuj swoim kodem, że wynik się
zgadza (oczekiwane: TP=5, FP=1, TN=4, FN=0, precision$\approx 0{,}833$,
recall$=1{,}0$, F1$\approx 0{,}909$).

**Uwaga**: w tym repozytorium `scikit-learn` **nie jest** zainstalowany —
zaimplementuj ROC-AUC i PR-AUC (*average precision*) ręcznie:

- **ROC-AUC** przez porównanie par (ranking): $P(\text{score}_{\text{anomalia}}
  > \text{score}_{\text{normalny}})$ po wszystkich parach (możesz użyć
  rangowania, nie musi być $O(n^2)$ dla dużych zbiorów — ale dla
  **weryfikacji** na małym przykładzie, $O(n^2)$ przez dwie pętle jest ok i
  pomaga sprawdzić poprawność szybszej wersji).
- **PR-AUC** (*average precision*): sortuj przykłady po score malejąco,
  dla każdego **prawdziwie pozytywnego** (anomalia) policz precision w tym
  miejscu rankingu, policz średnią tych precyzji.

Policz **na prawdziwych danych testowych** (sekcja 1): precision, recall,
F1, macierz pomyłek (2×2), ROC-AUC, PR-AUC.

❌ "ROC-AUC/PR-AUC to ostateczna miara jakości modelu detekcji anomalii —
nie potrzebuję nic więcej."

✅ ROC-AUC/PR-AUC oceniają **ranking** (czy anomalie generalnie mają wyższy
błąd niż normalne przykłady, niezależnie od konkretnego progu) — ale
**decyzja** "czy TEN konkretny obraz jest anomalią" wymaga **konkretnego
progu** (sekcja 4). Wysoki AUC nie mówi, jak dobrze wypadniesz przy
**wybranym** progu — stąd potrzebujesz **obu** rodzajów metryk, nie
jednego.

## 6. Analiza błędów i prosty baseline — Twoje zadanie

**Baseline bez uczenia**: policz błąd jako odległość do **średniego obrazu
klasy normalnej** (`ae_train.mean(dim=0)`), bez żadnego treningu —
`((x - mean_normal_image)**2).mean()`. Policz dla tego baseline te same
metryki (sekcja 5) na zbiorze testowym.

**Porównaj** AE vs ten trywialny baseline.

❌ "Wytrenowany autoenkoder musi radykalnie pobić trywialny baseline, bo
jest 'mądrzejszy'."

✅ **Niekoniecznie** — sprawdź to **empirycznie na swoich danych**, zamiast
zakładać. Jeśli autoenkoder nie bije wyraźnie trywialnego baseline'u, to
**ważny, wartościowy wniosek**, nie porażka eksperymentu: oznacza, że
separacja normalne/anomalie w tym konkretnym zadaniu w dużej mierze wynika
z prostych różnic pikselowych między klasami (kształt sneakera vs inne
ubrania), nie z czegoś subtelnego, czego nauczył się dopiero autoenkoder.
Krytyczna ocena metody wymaga takiego porównania **zawsze**, nie tylko gdy
oczekujemy, że "sprawdzi się".

**Analiza błędów (wymagane):** znajdź **przynajmniej jeden konkretny**
przypadek:
1. Anomalia (obraz spoza klasy normalnej) z **niskim** błędem rekonstrukcji
   (false negative) — sprawdź, z jakiej klasy pochodzi, i czy ma sens
   wizualny (np. czy przypomina sneakera).
2. Przykład normalny (sneaker) z **wysokim** błędem rekonstrukcji (false
   positive) — sprawdź wizualnie, czy jest nietypowy w ramach swojej klasy.

❌ "Autoenkoder, który generalnie dobrze rozdziela normalne/anomalie, musi
rekonstruować **każdy** przykład anomalii źle."

✅ **Nie jest to prawda w ogólności** — anomalia wizualnie/strukturalnie
podobna do klasy normalnej (np. inny typ obuwia, jeśli normalna klasa to
konkretny typ obuwia) może zostać zrekonstruowana **całkiem dobrze**, mimo
że formalnie jest anomalią. To **nie** jest błąd implementacji — to
rzeczywiste ograniczenie metody, które musisz **zademonstrować konkretnym
przykładem**, nie tylko zacytować z wykładu.

## 7. Zadanie diagnostyczne: przeciek danych w wyborze progu

Poniższy kod (w notebooku) **wygląda** rozsądnie — przeszukuje wiele
wartości progu i wybiera tę, która daje **najlepsze F1** — ale ma błąd:
przeszukiwanie i wybór odbywają się **na etykietach ze zbioru testowego**.

**Zadanie:**
1. Uruchom dostarczony kod: przeszukaj siatkę progów, dla każdego policz F1
   **na `test_normal`/`test_anomaly`** (z ich prawdziwymi etykietami), wybierz
   próg o najwyższym F1.
2. Porównaj wybrany w ten sposób próg z **Twoim** progiem z sekcji 4
   (wybranym tylko na podstawie `val_normal`). Czy są identyczne?
3. Porównaj zgłoszone F1: "przeciekowe" (wybrane i oceniane na tym samym
   zbiorze testowym) vs Twoje honest (próg z sekcji 4, F1 zmierzone na
   teście). **Nawet jeśli różnica jest niewielka** — wyjaśnij, **dlaczego
   sama wielkość różnicy nie jest tym, co czyni tę procedurę błędną**.
4. Napisz, dlaczego ta procedura jest metodologicznie błędna, **niezależnie**
   od tego, czy w Twoim konkretnym przebiegu różnica wyszła duża czy mała.
5. Zaproponuj poprawkę (dokładnie to, co zrobiłeś/aś w sekcji 4) i zweryfikuj,
   że jest zgodna z zasadą z sekcji 1 (próg nigdy nie widzi etykiet testowych).

❌ "Jeśli przeciek danych nie zmienił wyniku istotnie, to nie był prawdziwym
błędem."

✅ **To nieprawda.** Błąd metodologiczny (użycie etykiet testowych do podjęcia
decyzji modelowania) jest błędem **niezależnie** od tego, jak duży efekt
wywołał w konkretnym przebiegu — przy innym zbiorze danych, innym modelu,
czy mniejszym/mniej rozdzielonym zbiorze testowym, ten sam błąd mógłby dać
dramatycznie zawyżony wynik. Nie możesz tego wiedzieć z góry, patrząc tylko
na jeden przebieg — stąd zasada "nigdy nie dotykaj testu przy wyborze progu"
musi być **bezwzględna**, nie zależna od tego, czy "tym razem akurat nie
zaszkodziło".

## 8. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami/przykładami z sekcji 3-7:

1. Jak dobrze (konkretne liczby: precision/recall/F1/ROC-AUC) Twój
   autoenkoder rozdziela klasę normalną od anomalii na zbiorze testowym?
2. Jak wypada w porównaniu z trywialnym baseline'em (sekcja 6)? Co to mówi o
   tym, **skąd** bierze się zdolność rozróżniania w tym zadaniu?
3. Jaki jest Twój najbardziej przekonujący **konkretny przykład** ograniczenia
   metody (false negative albo false positive z sekcji 6)? Czy ma sens
   wizualny?
4. Jaki błąd wprowadził przeciek danych w sekcji 7, i jaka jest **ogólna**
   (nie tylko dla tego zbioru danych) zasada, którą z tego wyciągasz?
5. Czy uznałbyś/uznałabyś ten system za gotowy do praktycznego zastosowania?
   Czego by jeszcze potrzebował (np. jaki koszt błędu FP/FN jest akceptowalny
   w jakim zastosowaniu)?

## Rozszerzenie (opcjonalne, jeśli zostanie czas)

Wybierz **jedno** z poniższych:

- **Wpływ wymiaru latentnego**: powtórz sekcje 2-5 dla $d_e \in \{4, 32\}$
  (oprócz głównego $d_e=16$) i porównaj metryki. Czy większy/mniejszy
  bottleneck zmienia separację normalne/anomalie — i czy ten efekt jest
  taki, jakiego intuicyjnie oczekujesz?
- **VAE jako alternatywa**: zaimplementuj analogiczny VAE (wzorem wykładu),
  użyj błędu rekonstrukcji (albo $-$ELBO) jako sygnału anomalii. Porównaj z
  AE **bez** zakładania z góry, który "musi" wypaść lepiej (zobacz wykład:
  VAE nie jest automatycznie lepszym detektorem anomalii).

## Podział czasu (realistyczny, zweryfikowany względem rzeczywistego kosztu treningu)

| Blok | Czas |
|---|---|
| Sekcja 0 (konfiguracja) | 5 min |
| Sekcja 1 (dane i podział, bez przecieku) | 15 min |
| Sekcja 2 (AE: architektura i trening) | 15 min |
| Sekcja 3 (błąd rekonstrukcji, rozkłady) | 10 min |
| Sekcja 4 (wybór progu walidacyjnego) | 10 min |
| Sekcja 5 (metryki: znany przykład + test) | 15 min |
| Sekcja 6 (analiza błędów + baseline) | 15 min |
| Sekcja 7 (zadanie diagnostyczne: przeciek) | 20 min |
| Sekcja 8 (wnioski) | 10 min |

Suma: **115 min**, z 5-minutowym buforem. Trening AE na tym zbiorze trwa
ułamek sekundy do kilku sekund na CPU — budżet czasowy idzie w projektowanie
podziału danych, analizę błędów i zadanie diagnostyczne, nie w czekanie na
model. Rozszerzenie (wyżej) jest **opcjonalne** i **nie** wliczone w te
115 minut.

## Kryteria oceny

- Podział danych (sekcja 1) jest programowo zweryfikowany jako rozłączny;
  autoenkoder trenowany **wyłącznie** na danych normalnych.
- Próg detekcji (sekcja 4) wybrany **wyłącznie** na podstawie `val_normal`,
  nigdy na podstawie `val_anomaly` czy jakiegokolwiek `test_*`.
- Metryki (sekcja 5) zweryfikowane na podanym przykładzie z znanym wynikiem,
  **przed** użyciem na prawdziwych danych.
- Porównanie z trywialnym baseline'em (sekcja 6) jest **rzeczywiście
  wykonane i opisane** — niezależnie od tego, czy AE wypadł lepiej, podobnie,
  czy gorzej.
- Analiza błędów (sekcja 6) wskazuje **konkretny** przypadek (nie ogólnik)
  false negative lub false positive, z sensownym wizualnym komentarzem.
- Zadanie diagnostyczne (sekcja 7) prawidłowo identyfikuje przeciek danych i
  wyjaśnia, dlaczego błąd jest realny niezależnie od wielkości zaobserwowanej
  różnicy.
- Wnioski odwołują się do konkretnych liczb/przykładów z Twojego przebiegu,
  nie do ogólników z wykładu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/12_autoencoders/`) i nie jest częścią tego
repozytorium.
