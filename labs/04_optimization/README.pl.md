Język: [English](README.md) | **Polski**

# Laboratorium 04. Eksperymentalne porównanie strategii optymalizacji

Przewidywany czas pracy wynosi około 2 godzin. Laboratorium opiera się na
regułach aktualizacji SGD, momentum i Adama opisanych w
[`lectures/04_optimization/lecture.md`](../../lectures/04_optimization/lecture.md),
ale wykorzystuje inny problem. Zamiast dwuwymiarowej funkcji kwadratowej lub
klasyfikacji półksiężyców pojawia się regresja wielowymiarowa. Adam nie jest
implementowany od podstaw. Celem jest eksperymentalne porównanie i
diagnozowanie przebiegu optymalizacji.

## 1. Problem i dane

Zadanie dotyczy regresji z trzema cechami wejściowymi o celowo różnych
skalach. Pierwsza należy do przedziału $[-1,1]$, druga do $[-40,40]$, a
trzecia do $[-3,3]$. Wartość docelowa zależy od wszystkich trzech cech, w tym
nieliniowo od pierwszej. Dane powstają lokalnie i są dzielone na zbiór
treningowy (70%), walidacyjny (15%) i testowy (15%).

Każdy podział ma inną rolę.

- **Zbiór treningowy** służy do uczenia parametrów modelu.
- **Zbiór walidacyjny** służy do porównywania optymalizatorów i wyboru learning
  rate. Można korzystać z niego wielokrotnie.
- **Zbiór testowy** służy do jednokrotnej oceny wybranych konfiguracji na końcu
  laboratorium. Użycie go podczas podejmowania wcześniejszych decyzji
  pozbawiłoby wynik roli niezależnej miary jakości na nowych danych.

Różne skale cech są celowym elementem eksperymentu i wpływają na przebieg
optymalizacji.

## 2. Pętla treningowa

Należy zaimplementować funkcję
`train_model(opt_name, lr, epochs, batch_size, model_seed, data_seed, momentum)`.
Funkcja tworzy model za pomocą przygotowanego `make_model(seed)` oraz wybiera
odpowiedni optymalizator. Dostępne są `torch.optim.SGD` z momentum lub bez
niego oraz `torch.optim.Adam`. Adam nie powinien być implementowany od zera.

Trening korzysta z mini-batchy zwracanych przez `iterate_minibatches` i
zachowuje kolejność `zero_grad() -> forward -> strata (MSE) -> backward() ->
step()`. Po każdej epoce zapisywana jest strata treningowa i walidacyjna.
Obliczenie obu wartości powinno odbywać się w `torch.no_grad()`, aby nie
tworzyć niepotrzebnego grafu. Wynikiem funkcji jest
`(model, train_losses, val_losses)`.

Eksperyment pozostaje kontrolowany dzięki takim samym wartościom `model_seed`
i `data_seed` w każdym wywołaniu. Zapewniają one tę samą inicjalizację wag i
kolejność mini-batchy. Różnice między przebiegami wynikają więc z wyboru
optymalizatora i learning rate, a nie z losowości. Przygotowany baseline,
który zawsze przewiduje średnią ze zbioru treningowego, stanowi punkt
odniesienia.

## 3. Porównanie przy wspólnym learning rate

Funkcję `train_model` należy uruchomić dla `"sgd"`, `"momentum"` z domyślnym
`momentum=0.9` oraz `"adam"`. Każda konfiguracja korzysta z `lr=0.01` i
`epochs=20`. Krzywe straty treningowej i walidacyjnej dla wszystkich trzech
metod powinny znaleźć się na jednym wykresie, z osobną linią dla każdej
metody.

Przed przejściem dalej warto sprawdzić, czy każdy z optymalizatorów nauczył się
czegokolwiek użytecznego.

## 4. Przegląd learning rate

Dla SGD bez momentum należy przetestować `lr` z listy `[0.0001, 0.001, 0.01]`,
za każdym razem przez 20 epok. Dla momentum z `momentum=0.9` obowiązuje lista
`[0.00005, 0.0005, 0.005]`. Końcowe straty walidacyjne należy zebrać w prostej
tabeli lub czytelnym wydruku.

## 5. Diagnoza

W komórce Markdown należy zapisać diagnozę o długości 4–6 zdań. W sekcji 3
przynajmniej jeden optymalizator poza Adamem prawdopodobnie nie nauczy się
niczego sensownego przy `lr=0.01`, a strata może nawet osiągnąć `nan`.

Wyjaśnienie powinno łączyć wyniki przeglądu z sekcji 4 z różnymi skalami cech
i krzywizną funkcji straty omówioną na wykładzie. Samo stwierdzenie, że
learning rate był za duży, nie wystarcza. Potrzebne jest uzasadnienie, dlaczego
te dane powodują dużą wrażliwość SGD i momentum na wybór `lr`, podczas gdy
Adam jest znacznie mniej wrażliwy.

## 6. Finalna ewaluacja

Na podstawie straty walidacyjnej z sekcji 4 oraz wiedzy o Adamie należy wybrać
po jednej wartości `lr` dla każdego z trzech optymalizatorów. Każda wybrana
konfiguracja jest następnie trenowana przez 25 epok.

Dopiero po zakończeniu wyboru wszystkie trzy modele są oceniane na zbiorze
testowym, dokładnie raz. Tabela wyników powinna zawierać nazwę optymalizatora,
wybrany `lr` i stratę testową.

## 7. Wnioski

Odpowiedzi należy zapisać w komórce Markdown i uzasadnić wartościami z sekcji
6.

1. Który optymalizator osiągnął najniższą stratę testową? Czy wynik odpowiada
   wnioskom z wykładu dla problemu źle uwarunkowanego przez różne skale cech?
2. Czy po znormalizowaniu wszystkich cech do podobnego zakresu należałoby
   oczekiwać równie dużych różnic między optymalizatorami? Na czym opiera się
   ta prognoza?

## Zakres rdzeniowy i rozszerzenie

Sekcje 1–6 zajmują około 100 minut i tworzą rdzeń laboratorium. Orientacyjnie
wprowadzenie wraz z sekcją 2 wymaga około 30 minut, sekcje 3–4 około 35 minut,
sekcja 5 około 15 minut, a sekcja 6 około 20 minut. Sekcja 7 wymaga kolejnych
15–20 minut i stanowi obowiązkową interpretację wyników.

Opcjonalnym rozszerzeniem jest dodanie kolejnych wartości do przeglądu `lr`
albo sprawdzenie wartości `momentum` innej niż `0.9`.

## Kryteria oceny

- `train_model` zachowuje właściwą kolejność `zero_grad/backward/step`,
  poprawnie obsługuje mini-batche i zwraca historię strat.
- Porównanie przy wspólnym `lr` oraz przegląd z sekcji 4 pokazują różnice w
  zachowaniu metod, a nie trzy identyczne płaskie krzywe.
- Diagnoza z sekcji 5 wskazuje konkretną przyczynę związaną ze skalą cech i
  krzywizną straty, a nie tylko objaw w postaci zbyt dużego `lr`.
- Zbiór testowy jest używany dokładnie raz, po wyborze `lr` na podstawie
  walidacji.
- Wnioski z sekcji 7 odwołują się do rzeczywistych wartości z sekcji 6.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego w
`decisions/solutions/04_optimization/` i nie jest częścią repozytorium.
