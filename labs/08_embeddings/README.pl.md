# Laboratorium 08: system wyszukiwania podobnych obrazów (content-based image retrieval)

Czas: ok. 2 godziny. Bazuje na [`lectures/08_embeddings/lecture.md`](../../lectures/08_embeddings/lecture.md)
(embeddingi, odległość euklidesowa, cosine similarity, normalizacja L2, k-NN), ale
na **innym zbiorze i innym zadaniu** niż wykład — EuroSAT (zdjęcia satelitarne, nie
Caltech101) i **wyszukiwanie**, nie klasyfikacja ani wizualizacja. Produktem tego
laboratorium jest mały, działający system: zapytanie (obraz) → ranking najbardziej
podobnych obrazów z bazy.

## 0. Konfiguracja (gotowe)

Import bibliotek, urządzenie, ziarno losowości.

## 1. Dane: gallery i queries — Twoje zadanie (częściowo) + gotowe

EuroSAT, 10 klas terenu. Dzielimy dane na dwa **rozłączne** zbiory:

- **gallery** — baza obrazów, którą "przeszukujemy" (40 obrazów/klasę, 400 łącznie),
- **queries** — osobne obrazy, dla których szukamy podobnych w gallery (10
  obrazów/klasę, 100 łącznie, **inne obrazy niż w gallery**).

**Zadanie:** zweryfikuj programowo, że zbiory `gallery_indices` i `query_indices` się
nie nakładają (gotowa asercja czeka na dokończenie). To nie formalność — jeśli obraz
zapytania znajduje się też w gallery, wyszukiwarka "oszukuje": znajdzie samą siebie z
podobieństwem $1{,}0$, co sztucznie zawyża każdą metrykę trafności. To jest **pierwszy
i najważniejszy błąd metodologiczny**, jakiego ma unikać to laboratorium.

## 2. Ekstrakcja embeddingów — Twoje zadanie

Wczytaj `resnet18` z wagami `ResNet18_Weights.IMAGENET1K_V1` i wyekstrahuj embeddingi
(wyjście `avgpool`, **nie** `fc` — zobacz wykład, dlaczego logity to zły wybór do
porównań podobieństwa) dla `gallery_imgs` i `query_imgs`.

**Zadanie:** ustaw model w trybie, który **nie** liczy gradientów i **nie** aktualizuje
statystyk BatchNorm podczas ekstrakcji (dwa osobne mechanizmy — zobacz moduł 7, sekcja
*Błędne przekonania*: `requires_grad` kontroluje co innego niż `model.eval()`, a oba są
inne niż `torch.no_grad()`/`torch.inference_mode()`). Użyj **obu** właściwych
mechanizmów naraz — sama poprawna kombinacja jest częścią zadania, nie tylko wynik.

Zmierz i zanotuj czas ekstrakcji.

### Ważność "cache'u" w pamięci

Ekstrakcja embeddingów całej bazy ($500$ obrazów) trwa tu tylko kilka sekund — liczymy
ją **raz**, w jednej komórce, i używamy wielokrotnie w dalszych sekcjach (nie
przeliczamy cech od nowa przy każdym eksperymencie z metrykami).

**Zadanie (Markdown, 2-3 zdania):** gdyby baza była 1000× większa (a ekstrakcja trwała
godziny, nie sekundy), zapisałbyś/zapisałabyś embeddingi na dysk, żeby nie liczyć ich
ponownie przy każdym uruchomieniu notebooka. Pod jakimi **czterema** warunkami taki
zapisany cache przestałby być poprawny (czyli trzeba by go przeliczyć od zera)? Wskaż
konkretnie — nie "gdy coś się zmieni".

## 3. Normalizacja L2 — Twoje zadanie

Zaimplementuj `l2_normalize(X)`, zwracającą embeddingi podzielone przez swoją normę
(z zabezpieczeniem przed dzieleniem przez zero — zobacz wykład, sekcja o wektorze
zerowym). Zweryfikuj: po normalizacji każdy wektor ma normę bliską $1$ (gotowa
asercja).

## 4. Top-k wyszukiwanie — Twoje zadanie

Zaimplementuj `search(query_embedding, gallery_embeddings, k)`, zwracającą indeksy
$k$ najbardziej podobnych obrazów z gallery wg **cosine similarity**, posortowane od
najlepszego. **Napisz to sam/sama** — nie ma gotowej funkcji bibliotecznej do
wywołania; to jest macierzowe mnożenie i sortowanie, które już znasz z wykładu.

Zweryfikuj swoją implementację na małym, ręcznie skonstruowanym przykładzie z
dokładnie znanym wynikiem (gotowy w notebooku — $6$ wektorów $2$D, $2$ klasy, wynik
rankingu policzony z góry) — Twoja funkcja musi zwrócić dokładnie oczekiwaną kolejność.

## 5. Precision@k — Twoje zadanie

Zdefiniuj trafność: wynik uznajemy za trafny, jeśli jego klasa zgadza się z klasą
zapytania. **To uproszczenie** — zgodność etykiety nie jest pełną miarą podobieństwa
semantycznego (dwa zdjęcia tej samej klasy terenu mogą się różnić bardzo, dwa zdjęcia
różnych klas mogą wyglądać podobnie) — nazwij to wprost w swojej odpowiedzi w sekcji 9.

Zaimplementuj `precision_at_k(query_embeddings, query_labels, gallery_embeddings,
gallery_labels, k)`, zwracającą **średnie** Precision@k po wszystkich zapytaniach:
dla każdego zapytania, jaki ułamek jego $k$ najbliższych sąsiadów ma tę samą etykietę.

Zweryfikuj na tym samym małym przykładzie z sekcji 4 — wynik musi się zgadzać z
ręcznym wyliczeniem.

Policz Precision@1, Precision@5, Precision@10 dla embeddingów ResNet18 (znormalizowane
L2) na prawdziwych danych.

## 6. Baseline: odległość pikseli — Twoje zadanie

Policz te same metryki (Precision@1/5/10) dla **surowych, spłaszczonych pikseli**
zamiast embeddingów ResNet18 (ten sam zbiór gallery/queries, ta sama funkcja `search`
i `precision_at_k` — nie pisz nowego kodu, użyj tego, co już masz). Porównaj wyniki.

## 7. Wpływ normalizacji i metryki — Twoje zadanie

Porównaj Precision@5 dla **trzech** wariantów na embeddingach ResNet18:
1. cosine similarity na znormalizowanych L2 embeddingach (sekcja 5),
2. zwykły iloczyn skalarny (dot product) na **nieznormalizowanych** embeddingach,
3. odległość euklidesowa na nieznormalizowanych embeddingach.

Użyj **tego samego** zestawu gallery i queries dla wszystkich trzech wariantów.

**Zadanie (Markdown, 3-4 zdania):** który wariant wypadł najlepiej? Czy różnica jest
duża czy pomijalna? Jak to się ma do tego, czego nauczyłaś/eś się na wykładzie o
cosine similarity i normie wektora?

## 8. Wizualizacja i analiza błędów — Twoje zadanie

Wybierz $2$-$3$ zapytania: przynajmniej jedno z **wysokim** Precision@5 i przynajmniej
jedno z **niskim**. Narysuj zapytanie razem z jego top-5 wynikami (gotowa funkcja do
wizualizacji czeka na dane — obrazy + ich etykiety).

**Zadanie (Markdown, 3-5 zdań):** dla przykładu z niskim Precision@5 — spójrz na
zwrócone obrazy. Czy błędne wyniki wyglądają "sensownie podobne" (np. podobna tekstura
terenu, mylona klasa) czy wyglądają na przypadkowe trafienia? Które klasy EuroSAT są
najczęściej mylone w wynikach wyszukiwania (możesz policzyć to systematycznie po
wszystkich zapytaniach, nie tylko po wybranych przykładach)?

## 9. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami z sekcji 5-8:

1. O ile embeddingi ResNet18 przebijają baseline surowych pikseli (konkretne liczby
   Precision@k)? Czy ta przewaga jest zaskakująca, biorąc pod uwagę, że EuroSAT to
   domena odległa od ImageNet (moduł 7, lab)?
2. Czy wybór metryki/normalizacji (sekcja 7) miał duży wpływ na wynik? Gdybyś miał/a
   zbudować system retrieval od zera, czy normalizowałbyś/łabyś embeddingi? Dlaczego?
3. Przypomnij sobie zastrzeżenie z sekcji 5: zgodność etykiety to uproszczona miara
   trafności. Czy znalazłaś/eś w sekcji 8 przykład, gdzie wynik miał **inną** etykietę,
   ale wyglądał na sensownie podobny (albo odwrotnie: tę samą etykietę, ale wyglądał
   zupełnie inaczej)? Co to mówi o granicach tej metryki?
4. Jakie jest co najmniej jedno ograniczenie tego eksperymentu, które nie pozwala
   uogólnić wniosków na dowolny system wyszukiwania obrazów?

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, gallery/queries) | 10 min |
| Sekcja 2 (ekstrakcja + pytanie o cache) | 15 min |
| Sekcja 3 (normalizacja L2) | 10 min |
| Sekcja 4 (top-k search + weryfikacja na przykładzie) | 20 min |
| Sekcja 5 (Precision@k + weryfikacja na przykładzie) | 20 min |
| Sekcja 6 (baseline pikseli) | 10 min |
| Sekcja 7 (wpływ metryki/normalizacji) | 15 min |
| Sekcja 8 (wizualizacja + analiza błędów) | 15 min |
| Sekcja 9 (wnioski) | 10 min |

Sekcje 0-8 to rdzeń obowiązkowy. Ekstrakcja całej bazy trwa pojedyncze sekundy na
CPU — budżet czasowy ma iść w implementację rankingu/metryk i analizę wyników, nie w
czekanie na obliczenia.

**Rozszerzenie (opcjonalne, jeśli zostanie czas):** zaimplementuj rzeczywisty cache na
dysku (`torch.save`/`torch.load`) z jawnym tagiem ważności (nazwa wersji wag,
reprezentacja preprocessingu, lista indeksów obrazów, nazwa warstwy) — spójnie z
odpowiedzią z sekcji 2. Albo: sprawdź wpływ rozmiaru gallery (np. połowa obrazów) na
Precision@k.

## Kryteria oceny

- Gallery i queries są rozłączne — asercja w sekcji 1 przechodzi.
- Ekstrakcja embeddingów używa poprawnej kombinacji trybu modelu i kontekstu
  wyłączającego gradienty.
- `l2_normalize`, `search` i `precision_at_k` są zaimplementowane samodzielnie i
  dają poprawny wynik na ręcznie skonstruowanym przykładzie z sekcji 4-5 (nie tylko
  "wyglądają sensownie" na prawdziwych danych).
- Precision@k liczone z jawnym mianownikiem (liczba zapytań) i jawną definicją
  trafności (zgodność klasy, nazwaną wprost jako uproszczenie).
- Porównanie w sekcji 6-7 używa tego samego zestawu gallery/queries dla wszystkich
  wariantów.
- Wnioski w sekcji 9 odwołują się do konkretnych liczb z Twojego przebiegu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/08_embeddings/`) i nie jest częścią tego repozytorium.
