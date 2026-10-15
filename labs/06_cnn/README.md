# Laboratorium 06: projektowanie, trenowanie i analiza CNN do klasyfikacji obrazów

Czas: ok. 2 godziny. Bazuje na [`lectures/06_cnn/lecture.md`](../../lectures/06_cnn/lecture.md)
(konwolucja, `Conv2d`, pooling, liczba parametrów, pole recepcyjne), ale na
**innym zbiorze danych** niż wykład — KMNIST (odręcznie pisane znaki
kuzushiji: `o, ki, su, tsu, na, ha, ma, ya, re, wo`), nie Fashion-MNIST, i
na innym pytaniu: nie "CNN kontra MLP", ale "jak decyzje architektoniczne
wewnątrz samej CNN wpływają na jakość, liczbę parametrów i koszt?".

> **Dlaczego KMNIST, a nie CIFAR-10?** CIFAR-10 był pierwszym kandydatem,
> ale w tym środowisku jego pobranie zmierzono na ok. 90 minut — zbyt
> długo na zajęcia. KMNIST pobiera się w ok. 75 sekund i ma te same
> wymiary co Fashion-MNIST z wykładu ($28\times28$, skala szarości), więc
> umiejętności liczenia kształtów przenoszą się 1:1. Jeśli Twoje
> środowisko ma szybsze łącze, możesz w ramach rozszerzenia spróbować
> tego samego notebooka na CIFAR-10 — wymaga to zmiany kilku stałych
> (liczba kanałów wejściowych z 1 na 3, rozmiar obrazu z 28 na 32).

## 0. Konfiguracja (gotowe)

Import bibliotek, urządzenie (`cpu`/`cuda`), ziarno losowości.

## 1. Dane: struktura, rozkład klas, kształty tensorów (gotowe + krótkie zadanie)

KMNIST pobiera się automatycznie przez `torchvision.datasets.KMNIST` (nigdy
nie trafia do repozytorium — zobacz [`data/registry.yml`](../../data/registry.yml)).
Notebook wypisuje kształt pierwszego obrazu, liczbę przykładów w zbiorze
treningowym/testowym i przykładowe obrazy z podpisami klas.

**Zadanie (kod, 2-3 linie):** policz i wypisz liczbę przykładów każdej z
10 klas w podzbiorze treningowym (`torch.bincount`). Zbiór jest w
przybliżeniu zbalansowany czy nie? To ma znaczenie przy interpretacji
accuracy później.

## 2. Podział na train/val/test i normalizacja bez przecieku — Twoje zadanie

KMNIST ma już wydzielony zbiór testowy (osobny plik, inny niż treningowy).
Brakuje zbioru walidacyjnego — wydziel go **z danych treningowych**
(indeksy bez powtórzeń, gotowa funkcja `make_split` czeka na dokończenie).

Zaimplementuj `normalize_with_train_stats(X_train, X_val, X_test)`:
średnia i odchylenie standardowe pikseli liczone **wyłącznie** na
`X_train`, zastosowane (bez przeliczania) do `X_train`, `X_val` i
`X_test`. To ten sam błąd przecieku z laboratorium 05, teraz na obrazach
zamiast cech tabularnych — liczysz statystyki jednej liczby (średnia
jasność pikseli), nie wektora, ale zasada jest identyczna.

Zweryfikuj: `X_train` po normalizacji ma średnią bliską 0 i odchylenie
bliskie 1 (gotowa asercja to sprawdzi).

## 3. Model bazowy (gotowe)

Prosty liniowy klasyfikator (`Flatten` + jedna warstwa `Linear`) jako
punkt odniesienia — nie jest CNN i nie wykorzystuje struktury 2D obrazu
w żaden sposób. Jego accuracy to dolna granica, z którą porównasz swoje
architektury.

## 4. Projekt Twojej CNN — Wariant 1 — Twoje zadanie

Zaprojektuj i zaimplementuj klasę `CNNVariant1(nn.Module)` używając
`nn.Conv2d`, funkcji aktywacji i (opcjonalnie) `nn.MaxPool2d` — wybierasz
liczbę warstw konwolucyjnych, liczbę filtrów, rozmiar kernela, czy używać
poolingu. Jedyny wymóg: model musi przyjąć tensor `(N, 1, 28, 28)` i
zwrócić `(N, 10)`.

**Wymagany krok — weryfikacja kształtów:** po zdefiniowaniu klasy,
przepuść przez model jeden przykładowy batch i wypisz kształt tensora
**po każdej warstwie** (nie tylko na wyjściu). To jest dokładnie ten krok,
który w praktyce zapobiega najczęstszemu błędowi przy pisaniu CNN —
błędnie policzonemu rozmiarowi wejścia do `nn.Linear` po `flatten()`.
Jeśli się nie zgadza — popraw, zanim przejdziesz dalej.

**Zadanie (Markdown, 2-3 zdania):** jakie wybory architektoniczne
podjęłaś/podjąłeś (liczba warstw, filtry, kernel, pooling) i czym je
uzasadniasz?

## 5. Trening i walidacja Wariantu 1 (pętla gotowa, Twoje wywołanie)

Funkcja `train_eval(model, epochs, lr, seed)` jest gotowa (ten sam wzorzec
co w module 05/wykładzie 06) — wywołaj ją dla swojego modelu i zanotuj
accuracy walidacyjną, liczbę parametrów i czas treningu.

## 6. Projekt Wariantu 2 — Twoje zadanie

Zaprojektuj **drugi** wariant (`CNNVariant2`), różniący się od pierwszego
w **co najmniej jednym konkretnym, uzasadnionym aspekcie** — np. inna
liczba filtrów, inny rozmiar kernela, dodanie/usunięcie poolingu, inna
głębokość. Przed napisaniem kodu zapisz (Markdown, 2-3 zdania) **jaką
konkretną hipotezę sprawdzasz** tą zmianą (np. "więcej filtrów w pierwszej
warstwie powinno poprawić accuracy, ale zwiększy liczbę parametrów i
czas treningu").

Powtórz weryfikację kształtów z sekcji 4 dla tego wariantu.

## 7. Porównanie: baseline, Wariant 1, Wariant 2 — Twoje zadanie

Wytrenuj wszystkie trzy modele na **co najmniej 2 ziarnach losowości**
(np. `seed=0` i `seed=1`) i zestaw wyniki w tabeli/wydruku: accuracy
walidacyjna i testowa, liczba parametrów, zmierzony czas treningu i
inferencji. Zbiór testowy wykorzystujesz tutaj **po raz pierwszy** w tym
laboratorium (dopiero teraz, po wyborze na podstawie walidacji — tak jak
w module 05).

## 8. Ocena jakości: confusion matrix i analiza błędów — Twoje zadanie

Dla swojego najlepszego (według walidacji) wariantu: narysuj confusion
matrix na zbiorze testowym (gotowa funkcja do rysowania czeka na dane) i
wypisz 6-8 błędnie sklasyfikowanych przykładów z prawdziwą i przewidzianą
etykietą.

**Zadanie (Markdown, 3-5 zdań):** które klasy są najczęściej mylone ze
sobą? Spójrz na pomylone przykłady (czy zapisane znaki wizualnie się
przypominają?) — czy błędy modelu wydają się "sensowne" (podobne znaki) czy
przypadkowe? Nie zgaduj z góry — opisz to, co faktycznie widzisz w swojej
confusion matrix.

## 9. Wnioski — Twoje zadanie

Odpowiedz (Markdown, uzasadnione rzeczywistymi liczbami z sekcji 7-8):

1. Który wariant CNN wygrał na walidacji? Czy to ten sam, który wygrał na
   teście? Jeśli nie — co to mówi o wiarygodności pojedynczego porównania?
2. Czy wariant z większą liczbą parametrów był też szybszy, czy wolniej
   się trenował? Czy to, co zaobserwowałaś/eś, jest zgodne z intuicją
   "mniej parametrów = mniej obliczeń", czy ją podważa?
3. Czy wyższa accuracy treningowa Twojego najlepszego modelu oznacza, że
   dobrze generalizuje? Po czym to oceniasz (odwołaj się do konkretnych
   liczb z sekcji 7)?
4. Jakie jest co najmniej jedno ograniczenie tego eksperymentu (rozmiar
   podzbioru danych, liczba epok, konkretne ziarna losowości), które
   nie pozwala Ci uogólnić wniosków na "Wariant X jest zawsze lepszy"?

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, eksploracja danych) | 10 min |
| Sekcja 2 (podział + normalizacja bez przecieku) | 15 min |
| Sekcja 3 (baseline, gotowe, orientacja) | 5 min |
| Sekcja 4 (projekt Wariantu 1 + weryfikacja kształtów) | 30 min |
| Sekcja 5 (trening Wariantu 1) | 5 min |
| Sekcja 6 (projekt Wariantu 2 + hipoteza) | 20 min |
| Sekcja 7 (porównanie, ≥2 ziarna) | 10 min |
| Sekcja 8 (confusion matrix + analiza błędów) | 15 min |
| Sekcja 9 (wnioski) | 10 min |

Sekcje 0-8 to rdzeń obowiązkowy. Trening każdego wariantu na
zredukowanym podzbiorze danych trwa pojedyncze sekundy na CPU — budżet
czasowy ma iść w projektowanie i analizę, nie w czekanie na trening.

**Rozszerzenie (opcjonalne, jeśli zostanie czas):** trzeci wariant
architektury; analiza precision/recall per-klasa (nie tylko ogólna
accuracy); powtórzenie porównania na większym podzbiorze danych lub z
większą liczbą epok, żeby sprawdzić, czy wnioski się utrzymują.

## Kryteria oceny

- Normalizacja w sekcji 2 liczy statystyki tylko z `X_train` i stosuje je
  (bez przeliczania) do `X_val`/`X_test`; asercja przechodzi.
- Oba warianty CNN przyjmują `(N, 1, 28, 28)` i zwracają `(N, 10)`;
  weryfikacja kształtów w sekcjach 4 i 6 jest obecna i poprawna.
- Wariant 2 różni się od Wariantu 1 w konkretny, nazwany sposób — nie jest
  kopią z przypadkową zmianą jednej cyfry bez uzasadnienia.
- Porównanie w sekcji 7 obejmuje ≥2 ziarna losowości i raportuje accuracy
  **razem z** liczbą parametrów i czasem, nie samą accuracy.
- Zbiór testowy pojawia się dopiero w sekcji 7, po wyborze na podstawie
  walidacji.
- Wnioski w sekcji 9 odwołują się do konkretnych liczb z Twojego
  przebiegu, nie do ogólników ani założeń z wykładu bez sprawdzenia.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/06_cnn/`) i nie jest częścią tego repozytorium.
