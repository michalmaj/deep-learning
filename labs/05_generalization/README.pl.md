Język: [English](README.md) | **Polski**

# Laboratorium 05. Rozpoznawanie przeuczenia i dobór regularyzacji

Przewidywany czas pracy wynosi około 2 godzin. Laboratorium wykorzystuje
pojęcia z
[`lectures/05_generalization/lecture.md`](../../lectures/05_generalization/lecture.md),
w tym ryzyko empiryczne i oczekiwane, L2, weight decay, AdamW, dropout oraz
early stopping. Problem różni się od wykładowego. Zamiast regresji
jednowymiarowej pojawia się klasyfikacja danych tabularnych z cechami
informacyjnymi i szumowymi.

## 1. Dane, podział i przeciek danych

Zbiór ma 14 cech. Cztery z nich zawierają informację o celu, a dziesięć jest
czystym szumem. Do treningu służy 100 przykładów, a zbiory walidacyjny i
testowy zawierają po 200 przykładów. Dwie cechy mają celowo znacznie większą
skalę niż pozostałe, co odpowiada częstej sytuacji w danych tabularnych i
uzasadnia normalizację.

Notebook zawiera przygotowany, ale **celowo błędny** kod. Średnia i odchylenie
standardowe do normalizacji są w nim obliczane na całym zbiorze przed
podziałem na trening, walidację i test.

Pierwsze zadanie wymaga zapisania w komórce Markdown diagnozy o długości 3–5
zdań. Należy wskazać źródło błędu, wyjaśnić, dlaczego stanowi on data leakage,
oraz określić, które dane nie powinny wpływać na statystyki normalizacji.

Następnie należy przygotować poprawną wersję kodu. Statystyki są obliczane
**wyłącznie** na `X_train`, po czym te same wartości stosuje się do `X_train`,
`X_val` i `X_test` bez ponownego dopasowania. Średnie i odchylenia z błędnej
oraz poprawnej wersji należy porównać dla pierwszej cechy. Różnica powinna być
wyraźnie niezerowa.

W dalszej części obowiązują wyłącznie poprawnie znormalizowane dane.

## 2. Baseline i model o nadmiarowej pojemności

Baseline oparty na klasie większościowej oraz MLP `[32,32]` o pojemności
wystarczającej do przeuczenia 100 przykładów są gotowe do uruchomienia. Nie
stanowią części implementacyjnej zadania.

## 3. Pętla treningowa

Należy zaimplementować
`train_model(model, epochs, lr, weight_decay=0.0, use_adamw=False)`. Cały zbiór
treningowy mieści się w jednym batchu, dlatego pętla po mini-batchach nie jest
potrzebna. Wymagana jest poprawna kolejność `zero_grad() -> forward -> strata
(BCEWithLogitsLoss) -> backward() -> step()` oraz przełączanie
`model.train()` i `model.eval()` podczas obliczania straty walidacyjnej.
Funkcja zwraca `(train_losses, val_losses)`.

Po treningu modelu `[32,32]` bez regularyzacji przez 1000 epok należy
przeanalizować przygotowany wykres. Odpowiedź w komórce Markdown o długości
3–5 zdań powinna podać dokładność treningową i walidacyjną oraz wskazać
konkretny kształt krzywych świadczący o przeuczeniu. Sama różnica między
stratami nie wystarcza.

## 4. Weight decay i dropout

Dla modelu `[32,32]` należy porównać:

- `AdamW` z `weight_decay` równym `0.0`, `0.01`, `0.1` i `0.3`,
- `nn.Dropout` z `p` równym `0.0`, `0.2`, `0.4` i `0.6`. Dropout jest dodawany
  między warstwami ukrytymi przez przygotowaną funkcję
  `make_model(hidden_sizes, dropout_p=...)`.

Dla każdej konfiguracji należy zebrać dokładność i stratę walidacyjną, a
wyniki przedstawić w tabeli lub czytelnym wydruku.

## 5. Early stopping

Przygotowany szkielet `train_with_early_stopping(model, epochs, patience,
min_delta)` zawiera działającą pętlę treningową, ale nie zapisuje ani nie
przywraca najlepszego stanu modelu. Uzupełniona funkcja ma zwracać model
**przywrócony** do stanu z najlepszej epoki oraz numer tej epoki.

Gotowa asercja sprawdza implementację. Strata walidacyjna obliczona po
wywołaniu funkcji musi dokładnie odpowiadać najlepszej zapamiętanej stracie.

## 6. Kontrolowane porównanie i finalna ewaluacja

Na podstawie wyników z sekcji 4–5 należy wybrać jedną konfigurację o
najniższej stracie walidacyjnej. Może to być brak regularyzacji, weight decay,
dropout, early stopping lub uzasadnione wynikami połączenie metod. Wybrana
konfiguracja jest trenowana ponownie z tym samym ziarnem.

Dopiero wtedy, po raz pierwszy w laboratorium, model jest oceniany na zbiorze
testowym. Należy podać dokładność i stratę testową.

## 7. Wnioski

Odpowiedzi w komórce Markdown powinny odwoływać się do wartości z sekcji 4–6.

1. Która metoda regularyzacji dała największą poprawę względem modelu bez
   regularyzacji? Czy zmieniła dokładność, stratę, czy obie miary? Czy to
   rozróżnienie ma znaczenie?
2. Czy błędna normalizacja przed podziałem miała duży czy mały wpływ na końcowy
   wynik? Dlaczego procedura pozostaje błędna niezależnie od wielkości efektu
   w pojedynczym przebiegu?
3. Które ograniczenie eksperymentu, na przykład rozmiar danych, ziarno
   losowości lub architektura, uniemożliwia uznanie jednej metody za zawsze
   najlepszą?

## Podział czasu

Podane wartości są orientacyjne i nie zostały zweryfikowane podczas zajęć.

| Blok | Czas |
|---|---|
| Sekcja 1, diagnoza przecieku i poprawka | 20 min |
| Sekcja 2, orientacja | 5 min |
| Sekcja 3, pętla treningowa i interpretacja krzywych | 25 min |
| Sekcja 4, weight decay i dropout | 25 min |
| Sekcja 5, early stopping | 20 min |
| Sekcja 6, porównanie i test | 15 min |
| Sekcja 7, wnioski | 10 min |

Sekcje 1–7 tworzą obowiązkowy rdzeń. Opcjonalnym rozszerzeniem jest dodanie
wartości do przeglądu `weight_decay` i `p` albo połączenie dropout z weight
decay.

## Kryteria oceny

- Statystyki normalizacji w sekcji 1 są obliczane wyłącznie z `X_train` i
  stosowane bez ponownego dopasowania do pozostałych zbiorów.
- `train_model` i `train_with_early_stopping` działają poprawnie, a notebook
  wykonuje się od odpowiedniej komórki do końca bez błędu.
- Early stopping przywraca najlepszy stan, a przygotowana asercja nie jest
  pominięta ani usunięta.
- Zbiór testowy pojawia się dokładnie raz, po wyborze konfiguracji na podstawie
  walidacji.
- Interpretacje odwołują się do wartości i kształtów krzywych z wykonanego
  przebiegu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego w
`decisions/solutions/05_generalization/` i nie jest częścią repozytorium.
