Język: [English](README.md) | **Polski**

# Laboratorium 04: eksperymentalne porównanie strategii optymalizacji

Czas: ok. 2 godziny. Bazuje na [`lectures/04_optimization/lecture.md`](../../lectures/04_optimization/lecture.md)
(reguły aktualizacji SGD/momentum/Adam), ale na **innym problemie** niż
wykład (regresja wielowymiarowa, nie 2D miska ani klasyfikacja na
półksiężycach) i bez implementowania Adama od podstaw — matematykę
algorytmów masz z wykładu, tu uczysz się **eksperymentalnie porównywać i
diagnozować** optymalizację.

## 1. Problem i dane (gotowe)

Regresja: trzy cechy wejściowe o **celowo różnych skalach** (jedna w
zakresie $[-1,1]$, jedna w $[-40,40]$, jedna w $[-3,3]$), cel zależny od
wszystkich trzech (nieliniowo od pierwszej). Dane generowane lokalnie,
podzielone **trójstronnie**: trening (70%) / walidacja (15%) / test (15%).

**Role podziałów, których musisz przestrzegać przez całe laboratorium:**
- **trening** — do uczenia parametrów modelu,
- **walidacja** — do porównywania optymalizatorów i wyboru learning rate (możesz na niej "podglądać" wyniki tyle razy, ile chcesz),
- **test** — dotykasz go **dokładnie raz**, na samym końcu, dla już wybranej konfiguracji. Jeśli użyjesz testu do podejmowania decyzji po drodze, przestaje być uczciwą miarą tego, jak model radzi sobie z nowymi danymi.

Różne skale cech nie są przypadkiem — zobaczysz, dlaczego mają znaczenie.

## 2. Pętla treningowa — Twoje zadanie

Zaimplementuj `train_model(opt_name, lr, epochs, batch_size, model_seed, data_seed, momentum)`:
tworzy model (gotowa funkcja `make_model(seed)`), odpowiedni optymalizator
(`torch.optim.SGD` z/bez `momentum`, `torch.optim.Adam` — **nie**
implementujesz Adama od zera), i trenuje go, iterując po mini-batchach
(gotowa funkcja `iterate_minibatches`) z poprawną kolejnością: `zero_grad()
→ forward → strata (MSE) → backward() → step()`. Po każdej epoce zapisz
stratę treningową i walidacyjną (bez `torch.no_grad()` w tym miejscu
niepotrzebnie zużyjesz pamięć na graf, którego nie potrzebujesz).
Funkcja ma zwrócić `(model, train_losses, val_losses)`.

To jest **kontrolowany eksperyment**: ten sam `model_seed` i ten sam
`data_seed` przy każdym wywołaniu oznaczają tę samą inicjalizację wag i tę
samą kolejność mini-batchy — różnice w wynikach między wywołaniami będą
więc wynikać wyłącznie z wyboru optymalizatora/learning rate, nie z
przypadku. Baseline "przewiduj średnią z treningu" jest gotowy — to Twój
punkt odniesienia.

## 3. Porównanie przy wspólnym learning rate — Twoje zadanie

Wywołaj `train_model` dla `"sgd"`, `"momentum"` (domyślne `momentum=0.9`)
i `"adam"`, wszystkie z `lr=0.01`, `epochs=20`. Narysuj krzywe straty
treningowej i walidacyjnej dla wszystkich trzech na jednym wykresie
(każda metoda osobną linią).

**Zanim przejdziesz dalej:** przyjrzyj się wynikom. Czy wszystkie trzy
się nauczyły czegokolwiek?

## 4. Przegląd learning rate — Twoje zadanie

Dla **SGD** (bez momentum) przetestuj `lr` z listy `[0.0001, 0.001, 0.01]`
(`epochs=20` każdy), zbierz stratę walidacyjną końcową dla każdego.
Zrób to samo dla **momentum** (`momentum=0.9`) z listą
`[0.00005, 0.0005, 0.005]`. Przedstaw wyniki w prostej tabeli/wydruku.

## 5. Diagnoza — Twoje zadanie

**Zadanie (Markdown, 4–6 zdań):** w sekcji 3 przynajmniej jeden z
optymalizatorów (poza Adamem) prawdopodobnie nie nauczył się niczego
sensownego przy `lr=0,01` — strata mogła nawet wybuchnąć do `nan`.
Na podstawie wyników z sekcji 4 (przegląd `lr`) i tego, co wiesz z wykładu
o różnych skalach cech i krzywiźnie straty: zdiagnozuj, **dlaczego**
akurat `lr=0,01` zawiódł dla tego optymalizatora, podczas gdy Adam dał
sobie z nim radę bez problemu. Nie wystarczy napisać "lr było za duże" —
wyjaśnij **dlaczego** akurat ten problem (te konkretne dane) sprawia, że
SGD/momentum są tak wrażliwe na dobór `lr`, a Adam znacznie mniej.

## 6. Finalna ewaluacja — Twoje zadanie

Na podstawie przeglądu z sekcji 4 (i Twojej wiedzy o Adamie z wykładu)
wybierz **jeden, najlepszy wg straty walidacyjnej** `lr` dla każdego z
trzech optymalizatorów. Wytrenuj każdy z wybranym `lr` (`epochs=25`), a
następnie — **dopiero teraz, raz** — oceń każdy z trzech wytrenowanych
modeli na zbiorze **testowym**. Przedstaw wynik w tabeli: optymalizator,
wybrany `lr`, strata testowa.

## 7. Wnioski — Twoje zadanie

Odpowiedz (Markdown, uzasadnione realnymi liczbami z sekcji 6, nie
ogólnikami):

1. Który optymalizator osiągnął najniższą stratę testową? Czy ten wynik
   pasuje do tego, czego nauczył Cię wykład o tym konkretnym typie
   problemu (źle uwarunkowanym przez różne skale cech)?
2. Gdybyś miał/miała porównać te same trzy optymalizatory na problemie
   **bez** różnicy skal cech (wszystkie cechy znormalizowane do podobnego
   zakresu) — czy spodziewasz się podobnie dużej różnicy między nimi? Na
   czym opierasz tę przewidywanie?

## Zakres rdzeniowy vs. rozszerzenie (wskazówka czasowa)

Sekcje 1–6 (ok. 100 min: wprowadzenie i sekcja 2 ~30 min, sekcje 3–4
~35 min, sekcja 5 ~15 min, sekcja 6 ~20 min) to rdzeń — w pełni realizują
cel laboratorium. Sekcja 7 (~15–20 min) to krótka, ale obowiązkowa
interpretacja — **nie** jest rozszerzeniem do pominięcia, w przeciwieństwie
do ewentualnego dalszego poszerzania przeglądu `lr` o więcej wartości czy
o `momentum` inny niż `0,9`, co zostawiamy jako opcjonalne, jeśli zostanie
czas.

## Kryteria oceny

- `train_model` działa poprawnie (właściwa kolejność
  `zero_grad/backward/step`, poprawne mini-batche, zwraca historię).
- Porównanie przy wspólnym `lr` i przegląd `lr` z sekcji 4 faktycznie
  pokazują różnicę w zachowaniu (nie trzy identyczne, płaskie krzywe).
- Diagnoza w sekcji 5 odwołuje się do konkretnej przyczyny (skala cech /
  krzywizna straty), nie tylko do objawu ("było za duże").
- Finalna ewaluacja w sekcji 6 dotyka zbioru testowego dokładnie raz, po
  wyborze `lr` na podstawie walidacji — nie na odwrót.
- Wnioski w sekcji 7 odwołują się do rzeczywistych liczb z sekcji 6.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/04_optimization/`) i nie jest częścią tego
repozytorium.
