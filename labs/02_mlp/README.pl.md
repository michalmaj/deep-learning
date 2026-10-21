Język: [English](README.md) | **Polski**

# Laboratorium 02. Klasyfikacja danych tabularnych za pomocą MLP

Przewidywany czas pracy wynosi około dwóch godzin. Laboratorium rozwija materiał
z [`lectures/02_mlp/lecture.md`](../../lectures/02_mlp/lecture.md) dotyczący
perceptronu, MLP i funkcji aktywacji. Wykorzystuje jednak inny zbiór danych oraz
własną implementację `nn.Module` zamiast `nn.Sequential`. Obejmuje też porównanie
architektur i funkcji aktywacji oraz analizę błędnych klasyfikacji.

## 1. Problem i dane

Zadanie polega na binarnej klasyfikacji punktów należących do dwóch
**współśrodkowych okręgów** z dodanym szumem gaussowskim. Zewnętrzny pierścień
tworzy klasę 0, a wewnętrzny okrąg klasę 1. Jest to inny rodzaj nieliniowej
separowalności niż półksiężyce z wykładu. Potrzebna granica decyzyjna ma postać
zamkniętej krzywej, a nie pojedynczej fali.

Dane są generowane lokalnie i dzielone na zbiory treningowy oraz testowy w
sekcji 1 notebooka. Ten kod jest gotowy i nie stanowi części zadania.

## 2. Metryki

W sekcji 2 należy uzupełnić funkcję, która na podstawie predykcji i etykiet
oblicza **TP, FP, TN i FN**, a następnie **accuracy, precision i recall**. Klasa 1,
czyli wewnętrzny okrąg, jest klasą pozytywną. Funkcja służy w dalszej części
notebooka również do oceny modelu bazowego.

## 3. Model bazowy

Gotowy model bazowy składa się z pojedynczej warstwy liniowej
`nn.Linear(2, 1)`. Powtarza zasadniczą obserwację z wykładu na innym zbiorze
danych. Model liniowy nie rozwiązuje problemu nieliniowego. Na okręgach wypada
wyraźnie gorzej niż na półksiężycach i zbliża się do losowego zgadywania. Wynika
to z geometrii danych, ponieważ żadna prosta nie oddzieli pierścienia od środka.

## 4. Implementacja MLP

W sekcji 4 należy zaimplementować klasę `MLP(nn.Module)`. W odróżnieniu od
przykładu z wykładu nie korzysta ona z `nn.Sequential`. Liczba neuronów w
warstwie ukrytej oraz funkcja aktywacji powinny być konfigurowalne, aby ta sama
klasa mogła posłużyć do porównania w sekcji 6.

Sekcja 5 wymaga uzupełnienia funkcji trenującej, która wykonuje kolejno
`forward`, obliczenie straty przez `BCEWithLogitsLoss`, `backward` oraz
`optimizer.step()`. Funkcja powinna przyjmować dowolny model `nn.Module`.

## 5. Eksperyment

Gotowy kod trenuje domyślną konfigurację MLP za pomocą uzupełnionej klasy i
funkcji, a następnie porównuje jej wynik z modelem bazowym.

## 6. Porównanie architektur i funkcji aktywacji

Do porównania należy wybrać **co najmniej trzy** konfiguracje MLP różniące się
liczbą neuronów w warstwie ukrytej lub funkcją aktywacji. Przynajmniej jedna z
nich powinna być celowo zbyt słaba, na przykład ze względu na bardzo małą liczbę
neuronów. Pozwala to sprawdzić coś więcej niż zależność między rozmiarem modelu
a wynikiem. Każdą konfigurację należy wytrenować, a metryki testowe zebrać w
jednej tabeli lub jednym wydruku.

## 7. Ocena wyników

W komórce Markdown należy zinterpretować tabelę z sekcji 6. Analiza powinna
wskazywać najlepszą konfigurację oraz ewentualny model o zbyt małej pojemności.
Uzasadnienie powinno odwoływać się do tego, jakie funkcje dana architektura może
reprezentować, oraz do dowodu przedstawionego na wykładzie.

## 8. Analiza błędnych klasyfikacji

Gotowy kod rysuje granicę decyzyjną najlepszego modelu i zaznacza błędnie
sklasyfikowane punkty testowe. W komórce Markdown należy opisać ich położenie
względem granicy między pierścieniem a okręgiem w 3–5 zdaniach. Warto
rozstrzygnąć, czy są to punkty trudne z powodu bliskości prawdziwej granicy i
szumu, czy oznaka systematycznego błędu modelu. Uzasadnienie powinno wynikać z
wykresu.

## 9. Wnioski

W ostatniej komórce Markdown należy odpowiedzieć na dwa pytania.

- Czy wynik eksperymentu z sekcji 6 jest zgodny z wnioskiem z dowodu o składaniu
  przekształceń afinicznych? Czym obserwacja empiryczna różni się od samego
  dowodu?
- Jaka jedna praktyczna zasada dotycząca projektowania MLP wynika z wyników tego
  laboratorium?

## Kryteria oceny

- Metryki, klasa `MLP` i funkcja trenująca działają, a notebook wykonuje się od
  uzupełnionej komórki do końca bez błędu.
- Sekcja 6 obejmuje co najmniej trzy rzeczywiście różne konfiguracje, w tym jedną
  wyraźnie słabszą.
- Interpretacja wyników łączy wartości liczbowe ze zdolnością reprezentacyjną
  architektury.
- Analiza błędów odwołuje się do położenia błędnie sklasyfikowanych punktów na
  wykresie.
- Wnioski są zwięzłe, ale zawierają uzasadnienie.

Pełne rozwiązanie wzorcowe znajduje się lokalnie u prowadzącego w
`decisions/solutions/02_mlp/` i nie jest częścią repozytorium.
