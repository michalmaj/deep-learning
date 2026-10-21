Język: [English](README.md) | **Polski**

# Laboratorium 03. Backpropagation od podstaw w NumPy

Przewidywany czas pracy wynosi około dwóch godzin. Laboratorium rozwija
wyprowadzenie z
[`lectures/03_backprop/lecture.md`](../../lectures/03_backprop/lecture.md), ale
wymaga samodzielnej implementacji w NumPy bez `loss.backward()`. Zamiast danych
użytych do weryfikacji wzorów pojawia się regresja jednej zmiennej. Funkcją
aktywacji jest tanh, a nie sigmoid. Jej pochodna została przedstawiona w
wykładzie 2 i tutaj zostanie użyta w propagacji wstecznej.

Celem nie jest zbudowanie uniwersalnego mechanizmu automatycznego
różniczkowania. Implementacja obejmuje przebieg w przód i wstecz dla jednej,
ustalonej architektury z pojedynczą warstwą ukrytą. Najważniejsze jest
zrozumienie mechanizmu, a nie stworzenie zamiennika PyTorch.

## 1. Problem i dane

Zadanie dotyczy regresji jednej zmiennej dla $x\in[-3,3]$ oraz
$y = \sin(2x) + 0{,}3x^2 + \varepsilon$. Funkcja jest wyraźnie nieliniowa, więc
model liniowy nie może jej dobrze odwzorować. Dane są generowane lokalnie z
ustalonym ziarnem. Kod odpowiedzialny za ich utworzenie i wizualizację jest
gotowy.

## 2. Propagacja w przód

Do zaimplementowania pozostaje funkcja `forward(X, W1, b1, W2, b2)` dla sieci
`Linear(1, hidden) → tanh → Linear(hidden, 1)`. Powinna zwracać wszystkie
wartości pośrednie $Z^{(1)}, A^{(1)}$ i $Z^{(2)}$, ponieważ są potrzebne w
propagacji wstecznej. Przed przejściem dalej należy sprawdzić kształty zwracanych
tablic.

## 3. Propagacja wsteczna

Głównym zadaniem jest implementacja `backward(X, Y, Z1, A1, Z2, W2)`, która
zwraca `(dW1, db1, dW2, db2)`.

1. Najpierw należy wyprowadzić pochodną straty MSE względem $Z^{(2)}$ dla
   batcha. Jest to wzór z sekcji *Wersja wsadowa* wykładu zapisany z oznaczeniami
   używanymi w notebooku.
2. Następnie należy zastosować pochodną $\tanh'(z) = 1 - \tanh^2(z)$ zamiast
   $\sigma(z)(1-\sigma(z))$. Ten krok odróżnia implementację od przykładu z
   wykładu 3.
3. Propagacja przez obie warstwy powinna korzystać ze wzorów macierzowych,
   uwzględniając transpozycje, mnożenie $\odot$ i sumowanie gradientów biasów po
   batchu.

Kształty `dW1, db1, dW2, db2` muszą dokładnie odpowiadać kształtom
`W1, b1, W2, b2` przed rozpoczęciem sekcji 4.

## 4. Weryfikacja gradientu

Funkcja `gradient_check(...)` powinna zbudować w PyTorch **ten sam** przebieg w
przód, używając tych samych wartości `W1, b1, W2, b2` przekazanych jako
`torch.tensor(..., requires_grad=True)`. Po wywołaniu `backward()` należy
porównać gradienty z implementacją NumPy i wypisać maksymalną różnicę dla każdej
z czterech tablic.

Trening można rozpocząć dopiero po przejściu tej kontroli. Oczekiwane różnice są
rzędu $10^{-10}$ lub mniejsze. NumPy domyślnie korzysta z `float64`, dlatego
ograniczenia precyzji omawiane na wykładzie nie powinny tu dominować.

## 5. Trening

Do uzupełnienia pozostaje `train(X, Y, hidden_size, lr, epochs)`. Funkcja losowo
inicjalizuje wagi, wywołuje `forward` i `backward`, ręcznie aktualizuje parametry
przez `W -= lr * dW` i zapisuje historię straty. `torch.optim` nie jest używany,
ponieważ mechanizm optymalizatorów stanowi temat wykładu 4. Końcową stratę
należy porównać z gotowym modelem bazowym przewidującym średnią wartość $y$.

## 6. Interpretacja

Gotowy kod rysuje krzywą straty oraz predykcję sieci na tle danych. Komórka
Markdown powinna zawierać analizę w 3–5 zdaniach. Należy ocenić kształt i tempo
spadku straty, a także jakość dopasowania w całym zakresie $x$. Warto wskazać
ewentualny obszar systematycznego odchylenia. Jeśli go nie ma, pozostały rozrzut
punktów należy odnieść do szumu użytego podczas generowania danych.

## 7. Błędny gradient

Notebook zawiera drugą, celowo błędną wersję `backward` i trenuje za jej pomocą
identyczną sieć na tych samych danych. Rodzaj błędu można rozpoznać dopiero w
kodzie. Wynik należy porównać z krzywą straty poprawnej implementacji.

Analiza w komórce Markdown powinna mieć 3–5 zdań. Należy wskazać pominięte
mnożenie lub sumowanie, odwołać się do odpowiedniego wzoru z wykładu i wyjaśnić,
dlaczego błąd zmienia przebieg straty, ale nie powoduje błędu wykonania.

## Kryteria oceny

- `forward`, `backward`, `gradient_check` i `train` działają, a notebook wykonuje
  się od uzupełnionej komórki do końca bez błędu.
- Gradienty z `backward` zgadzają się z PyTorch `autograd` w granicach
  $10^{-8}$–$10^{-10}$.
- Model trenowany własnymi gradientami osiąga wyraźnie niższą stratę niż model
  bazowy przewidujący średnią.
- Interpretacja z sekcji 6 odwołuje się do konkretnych wartości lub kształtu
  wykresów.
- Analiza z sekcji 7 wskazuje brakujący element wzoru propagacji wstecznej, a nie
  tylko objaw w postaci gorszej straty.

Pełne rozwiązanie wzorcowe znajduje się lokalnie u prowadzącego w
`decisions/solutions/03_backprop/` i nie jest częścią repozytorium.
