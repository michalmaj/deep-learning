Język: [English](README.md) | **Polski**

# Laboratorium 03: backpropagation od podstaw w NumPy

Czas: ok. 2 godziny. Bazuje na wyprowadzeniu z [`lectures/03_backprop/lecture.md`](../../lectures/03_backprop/lecture.md),
ale implementujesz je **samodzielnie w NumPy**, bez `loss.backward()`, dla
innego problemu niż wykład (regresja jednej zmiennej, nie losowe dane do
weryfikacji wzorów) i z inną aktywacją (**tanh**, nie sigmoid z wykładu —
wzór na pochodną tanh poznałeś/aś w wykładzie 2, teraz użyjesz go w
backpropagation).

Nie budujesz uniwersalnego silnika automatycznego różniczkowania — tylko
forward i backward dla **tej jednej, ustalonej architektury** (1 warstwa
ukryta). Celem jest zrozumienie mechanizmu, nie napisanie alternatywy dla
PyTorch.

## 1. Problem i dane (gotowe)

Regresja jednej zmiennej: $x\in[-3,3]$, $y = \sin(2x) + 0{,}3x^2 + \varepsilon$
— funkcja wyraźnie nieliniowa (model liniowy nie ma szans), generowana
lokalnie z ustalonym ziarnem. Kod generujący dane i wizualizujący je jest
gotowy.

## 2. Forward pass — Twoje zadanie

Zaimplementuj `forward(X, W1, b1, W2, b2)` dla sieci: `Linear(1, hidden) →
tanh → Linear(hidden, 1)`. Funkcja musi zwrócić **wszystkie** wartości
pośrednie ($Z^{(1)}, A^{(1)}, Z^{(2)}$), nie tylko końcową predykcję —
będą potrzebne do propagacji wstecznej w kolejnym kroku. Sprawdź kształty
tego, co zwracasz, zanim przejdziesz dalej.

## 3. Backward pass — Twoje zadanie

To jest rdzeń laboratorium. Zaimplementuj `backward(X, Y, Z1, A1, Z2, W2)`,
która zwraca `(dW1, db1, dW2, db2)`:

1. Wyprowadź (na kartce, zanim zaczniesz pisać kod) pochodną straty MSE
   względem $Z^{(2)}$ dla batcha — to jest dokładnie wzór z wykładu
   (sekcja *Wersja wsadowa*), tylko podstaw swoje oznaczenia.
2. Wyprowadź pochodną $\tanh'(z) = 1 - \tanh^2(z)$ (podaną w wykładzie 2) i
   użyj jej zamiast $\sigma(z)(1-\sigma(z))$ z wykładu 3 — to jest miejsce,
   w którym Twoja implementacja musi się różnić od wzoru z wykładu, nie
   kopiować go jeden do jednego.
3. Zaimplementuj propagację wsteczną przez obie warstwy, używając wzorów
   macierzowych z wykładu (transpozycje, $\odot$, sumowanie po batchu dla
   biasów).

Nie przechodź do sekcji 4, dopóki kształty `dW1, db1, dW2, db2` nie
zgadzają się dokładnie z kształtami `W1, b1, W2, b2`.

## 4. Weryfikacja gradientu — Twoje zadanie

Zaimplementuj `gradient_check(...)`, która buduje **ten sam** forward pass
w PyTorch (z tych samych wartości `W1, b1, W2, b2`, jako `torch.tensor(...,
requires_grad=True)`), woła `backward()`, i porównuje otrzymane gradienty
z Twoimi — wypisz maksymalną różnicę dla każdego z czterech gradientów.

**Nie przechodź do treningu, dopóki ta weryfikacja nie przejdzie** (różnice
rzędu $10^{-10}$ lub mniejsze — NumPy domyślnie liczy w `float64`, więc
nie powinieneś/aś napotkać ograniczeń precyzji z wykładu).

## 5. Trening — Twoje zadanie

Zaimplementuj `train(X, Y, hidden_size, lr, epochs)`: inicjalizuje wagi
losowo, w pętli woła Twój `forward` i `backward`, aktualizuje parametry
**ręcznie** (`W -= lr * dW`, bez `torch.optim` — ten mechanizm poznasz w
wykładzie 4), zbiera historię straty. Porównaj końcową stratę z
baseline'em "przewiduj średnią z $y$" (gotowe).

## 6. Interpretacja — Twoje zadanie

Notebook rysuje (gotowe): krzywą straty i wykres dopasowania (predykcja
Twojej sieci na tle prawdziwych danych). Odpowiedz (Markdown, 3–5 zdań):
czy krzywa straty wygląda tak, jak oczekujesz po wykładzie 1 i 3 (kształt,
tempo spadku)? Czy dopasowanie dobrze oddaje kształt funkcji w całym
przedziale $x$, czy widzisz region z systematycznym (nie losowym)
odchyleniem od danych — a jeśli nie, czy pozostały rozrzut punktów wokół
krzywej modelu da się wytłumaczyć samym szumem, z jakim dane zostały
wygenerowane?

## 7. Błędny gradient — Twoje zadanie

Notebook zawiera (gotowe) drugą wersję `backward`, z **celowo** wstawionym
błędem (zobaczysz który dopiero w kodzie — nie zdradzamy go tutaj) i
trenuje nią identyczną sieć na tych samych danych. Porównaj krzywe straty
obu wersji.

**Zadanie (Markdown, 3–5 zdań):** zidentyfikuj, czego brakuje w błędnej
wersji `backward`, odwołując się do konkretnego wzoru z wykładu (które
mnożenie/sumowanie zostało pominięte), i wyjaśnij, dlaczego akurat ten
brak prowadzi do zaobserwowanego zachowania krzywej straty (a nie np. do
błędu wykonania).

## Kryteria oceny

- `forward`, `backward`, `gradient_check` i `train` działają (notebook
  wykonuje się od danej komórki do końca bez błędu).
- Gradienty z `backward` zgadzają się z PyTorch `autograd` w granicach
  $10^{-8}$–$10^{-10}$ (nie $10^{-3}$ — to by znaczyło, że coś jest nie tak).
- Model trenowany własnymi gradientami bije baseline "średnia" o wyraźny
  margines, nie tylko nieznacznie.
- Interpretacja w sekcji 6 odwołuje się do konkretnego kształtu
  wykresów/liczb, nie do ogólników.
- Interpretacja błędu w sekcji 7 poprawnie wskazuje brakujący element
  wzoru backpropagation, nie tylko opisuje objaw ("strata jest gorsza").

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/03_backprop/`) i nie jest częścią tego repozytorium.
