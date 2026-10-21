Język: [English](README.md) | **Polski**

# Laboratorium 06. Projektowanie, trenowanie i analiza CNN

Laboratorium trwa około 2 godzin i rozwija materiał z
[`lectures/06_cnn/lecture.md`](../../lectures/06_cnn/lecture.md). Obejmuje
konwolucję, `Conv2d`, pooling, liczbę parametrów i pole recepcyjne. Tym razem
używany jest zbiór KMNIST z odręcznie pisanymi znakami kuzushiji: `o, ki, su,
tsu, na, ha, ma, ya, re, wo`. Celem nie jest kolejne porównanie CNN z MLP, lecz
sprawdzenie, jak decyzje dotyczące samej CNN wpływają na jakość, rozmiar i koszt.

> **Dlaczego KMNIST zamiast CIFAR-10?** Pobranie CIFAR-10 trwało w tym
> środowisku około 90 minut, co wyklucza jego użycie podczas zajęć. KMNIST
> pobiera się w około 75 sekund i ma taki sam format jak Fashion-MNIST
> ($28\times28$, skala szarości). Rozszerzenie z CIFAR-10 wymaga zmiany liczby
> kanałów wejściowych z 1 na 3 oraz rozmiaru obrazu z 28 na 32.

## 0. Konfiguracja (gotowe)

Import bibliotek, wybór urządzenia (`cpu` lub `cuda`) i ustawienie ziarna.

## 1. Dane i rozkład klas (gotowe + krótkie zadanie)

KMNIST jest pobierany automatycznie przez `torchvision.datasets.KMNIST` i nie
trafia do repozytorium. Szczegóły znajdują się w
[`data/registry.yml`](../../data/registry.yml). Notebook pokazuje kształt
pierwszego obrazu, liczebność zbiorów oraz przykłady z etykietami klas.

**Zadanie w kodzie, 2-3 linie.** Za pomocą `torch.bincount` należy policzyć i
wypisać liczbę przykładów każdej z 10 klas w podzbiorze treningowym. Na tej
podstawie należy ocenić, czy zbiór jest w przybliżeniu zbalansowany i jak wpływa
to na późniejszą interpretację accuracy.

## 2. Podział train/val/test i normalizacja bez przecieku

KMNIST udostępnia osobny zbiór testowy. Zbiór walidacyjny należy wydzielić z
danych treningowych za pomocą niepowtarzających się indeksów. Do uzupełnienia
pozostaje przygotowana funkcja `make_split`.

Funkcja `normalize_with_train_stats(X_train, X_val, X_test)` powinna obliczać
średnią i odchylenie standardowe pikseli wyłącznie na `X_train`. Te same
statystyki, bez ponownego obliczania, muszą zostać użyte dla `X_train`, `X_val`
i `X_test`. Zasada jest taka sama jak w laboratorium 05, choć tym razem
normalizowana jest jasność pikseli zamiast wektora cech tabularnych.

Gotowa asercja sprawdza, czy po normalizacji `X_train` ma średnią bliską 0 i
odchylenie standardowe bliskie 1.

## 3. Model bazowy (gotowe)

Punktem odniesienia jest klasyfikator złożony z `Flatten` i jednej warstwy
`Linear`. Nie wykorzystuje on dwuwymiarowej struktury obrazu. Jego wynik służy
do porównania z projektowanymi architekturami.

## 4. Projekt CNN, wariant 1

Należy zaimplementować klasę `CNNVariant1(nn.Module)` z użyciem `nn.Conv2d`,
funkcji aktywacji i opcjonalnie `nn.MaxPool2d`. Liczba warstw, filtrów, rozmiar
kernela i użycie poolingu pozostają częścią projektu. Model musi przyjmować
tensor `(N, 1, 28, 28)` i zwracać `(N, 10)`.

**Wymagana weryfikacja kształtów.** Po zdefiniowaniu klasy należy przepuścić
przez model przykładowy batch i wypisać kształt tensora po każdej warstwie.
Pozwala to wykryć błędny rozmiar wejścia do `nn.Linear` po `flatten()` przed
rozpoczęciem treningu.

**Zadanie w Markdown, 2-3 zdania.** Należy opisać wybraną liczbę warstw,
filtrów, rozmiar kernela i sposób użycia poolingu oraz krótko je uzasadnić.

## 5. Trening i walidacja wariantu 1

Gotową funkcję `train_eval(model, epochs, lr, seed)` należy wywołać dla
zaprojektowanego modelu. Do zapisania pozostają accuracy walidacyjna, liczba
parametrów i czas treningu.

## 6. Projekt CNN, wariant 2

Drugi wariant `CNNVariant2` musi różnić się od pierwszego co najmniej jednym
konkretnym i uzasadnionym elementem. Może to być liczba filtrów, rozmiar kernela,
pooling albo głębokość. Przed kodem należy zapisać w 2-3 zdaniach hipotezę
sprawdzaną przez tę zmianę, na przykład oczekiwany wpływ większej liczby filtrów
na accuracy, liczbę parametrów i czas treningu.

Dla drugiego wariantu obowiązuje ta sama weryfikacja kształtów co w sekcji 4.

## 7. Porównanie trzech modeli

Baseline i oba warianty CNN należy wytrenować dla co najmniej dwóch ziaren
losowości, na przykład `seed=0` i `seed=1`. Tabela lub wydruk powinny obejmować
accuracy walidacyjną i testową, liczbę parametrów oraz zmierzony czas treningu i
inferencji. Zbiór testowy może zostać użyty dopiero w tej sekcji, po wyborze
modelu na podstawie walidacji.

## 8. Confusion matrix i analiza błędów

Dla najlepszego według walidacji wariantu należy narysować confusion matrix na
zbiorze testowym. Gotowa funkcja czeka na dane. Należy również pokazać 6-8
błędnie sklasyfikowanych przykładów wraz z prawdziwą i przewidzianą etykietą.

**Zadanie w Markdown, 3-5 zdań.** Analiza powinna wskazać najczęściej mylone
klasy oraz ocenić podobieństwo wizualne błędnie rozpoznanych znaków. Wnioski
muszą wynikać z uzyskanej macierzy i przykładów, a nie z założeń przyjętych
przed eksperymentem.

## 9. Wnioski

Odpowiedzi powinny odwoływać się do rzeczywistych wyników z sekcji 7 i 8.

1. Który wariant wygrał na walidacji i czy ten sam model uzyskał najlepszy wynik
   na teście? Co ewentualna różnica mówi o wiarygodności pojedynczego porównania?
2. Czy wariant z większą liczbą parametrów trenował się szybciej, czy wolniej?
   Czy wynik potwierdza intuicję, że mniej parametrów oznacza mniej obliczeń?
3. Czy wyższa accuracy treningowa najlepszego modelu świadczy o dobrej
   generalizacji? Odpowiedź powinna wskazywać konkretne wyniki z sekcji 7.
4. Jakie ograniczenie eksperymentu, na przykład rozmiar podzbioru, liczba epok
   lub dobór ziaren, utrudnia uznanie jednego wariantu za zawsze lepszy?

## Orientacyjny podział czasu

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, eksploracja danych) | 10 min |
| Sekcja 2 (podział + normalizacja bez przecieku) | 15 min |
| Sekcja 3 (baseline, gotowe, orientacja) | 5 min |
| Sekcja 4 (projekt wariantu 1 + weryfikacja kształtów) | 30 min |
| Sekcja 5 (trening wariantu 1) | 5 min |
| Sekcja 6 (projekt wariantu 2 + hipoteza) | 20 min |
| Sekcja 7 (porównanie, co najmniej 2 ziarna) | 10 min |
| Sekcja 8 (confusion matrix + analiza błędów) | 15 min |
| Sekcja 9 (wnioski) | 10 min |

Sekcje 0-9 tworzą obowiązkowy rdzeń. Trening każdego wariantu na zmniejszonym
podzbiorze trwa na CPU kilka sekund, dlatego większość czasu można przeznaczyć
na projektowanie i analizę.

**Rozszerzenie opcjonalne.** Można przygotować trzeci wariant architektury,
przeanalizować precision i recall dla każdej klasy albo powtórzyć porównanie na
większym podzbiorze lub z większą liczbą epok.

## Kryteria oceny

- Normalizacja liczy statystyki tylko z `X_train` i bez ponownego obliczania
  stosuje je do `X_val` oraz `X_test`. Gotowa asercja przechodzi.
- Oba warianty przyjmują `(N, 1, 28, 28)` i zwracają `(N, 10)`. Weryfikacja
  kształtów w sekcjach 4 i 6 jest obecna i poprawna.
- Wariant 2 różni się od wariantu 1 w nazwany i uzasadniony sposób.
- Porównanie obejmuje co najmniej dwa ziarna i raportuje accuracy razem z liczbą
  parametrów oraz czasem.
- Zbiór testowy pojawia się dopiero w sekcji 7, po wyborze na podstawie walidacji.
- Wnioski korzystają z konkretnych wyników, a nie z ogólników lub
  niezweryfikowanych założeń.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego w
`decisions/solutions/06_cnn/` i nie jest częścią repozytorium.
