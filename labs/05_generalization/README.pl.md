Język: [English](README.md) | **Polski**

# Laboratorium 05: rozpoznawanie przeuczenia i dobór regularyzacji

Czas: ok. 2 godziny. Bazuje na [`lectures/05_generalization/lecture.md`](../../lectures/05_generalization/lecture.md)
(ryzyko empiryczne vs. oczekiwane, L2/weight decay, AdamW, dropout, early
stopping), ale na **innym problemie** niż wykład — klasyfikacja tabularna
z cechami informacyjnymi i szumowymi, nie regresja 1D.

## 1. Dane i podział (gotowe) + przeciek danych — Twoje zadanie

Dane: 14 cech ($4$ informacyjne, $10$ czysty szum — niezwiązane z celem),
$100$ przykładów treningowych, po $200$ walidacyjnych i testowych. Dwie
cechy mają celowo dużo większą skalę niż pozostałe (częsta sytuacja w
realnych danych tabularnych) — dlatego normalizacja ma tu znaczenie.

Notebook zawiera (gotowy, ale **celowo błędny**) kod, który liczy średnią
i odchylenie standardowe do normalizacji **na całym zbiorze danych, przed
podziałem** na trening/walidację/test.

**Zadanie (Markdown, 3–5 zdań, zanim przejdziesz dalej):** na czym
dokładnie polega błąd w tym kodzie? Dlaczego to jest data leakage, nawet
jeśli model "dobrze działa"? Jakie dane, w jakim momencie, nie powinny być
widoczne przy liczeniu tych statystyk?

**Zadanie (kod):** napisz poprawną wersję — statystyki normalizacji
policzone **wyłącznie** na `X_train`, zastosowane (tymi samymi, nie
przeliczonymi na nowo) do `X_train`, `X_val` i `X_test`. Porównaj
policzone średnie/odchylenia (błędne vs. poprawne) dla pierwszej cechy —
różnica powinna być wyraźnie niezerowa.

Od tego miejsca w notebooku pracujesz **wyłącznie** na poprawnie
znormalizowanych danych.

## 2. Model bazowy i model o nadmiarowej pojemności (gotowe)

Baseline (klasa większościowa) i MLP `[32,32]` o pojemności wystarczającej
do przeuczenia na $100$ przykładach są gotowe do uruchomienia — to nie
jest część zadania.

## 3. Pętla treningowa — Twoje zadanie

Zaimplementuj `train_model(model, epochs, lr, weight_decay=0.0,
use_adamw=False)`: poprawna pętla z mini-batchami **nie jest tu potrzebna**
(cały zbiór treningowy, $100$ przykładów, mieści się w jednym batchu) —
ale kolejność `zero_grad → forward → strata (BCEWithLogitsLoss) →
backward() → step()` oraz przełączanie `model.train()`/`model.eval()`
przy liczeniu straty walidacyjnej musi być poprawna. Zwraca
`(train_losses, val_losses)`.

**Zadanie (Markdown, 3–5 zdań):** uruchom `train_model` na modelu
`[32,32]` bez regularyzacji na $1000$ epok i narysuj krzywe (kod do
wykresu gotowy). Jaka jest dokładność treningowa vs. walidacyjna? W
którym miejscu krzywej widać dowód przeuczenia — nie "straty są różne",
tylko konkretny kształt, który to pokazuje?

## 4. Weight decay i dropout — Twoje zadanie

Dla modelu `[32,32]`:

- przetestuj `AdamW` z `weight_decay` z listy `[0.0, 0.01, 0.1, 0.3]`,
- przetestuj `nn.Dropout` (dodany między warstwami ukrytymi — gotowa
  funkcja `make_model(hidden_sizes, dropout_p=...)`) z `p` z listy
  `[0.0, 0.2, 0.4, 0.6]`.

Dla każdej konfiguracji zbierz dokładność i stratę walidacyjną. Przedstaw
wyniki w tabeli/wydruku.

## 5. Early stopping — Twoje zadanie

Dostajesz szkielet `train_with_early_stopping(model, epochs, patience,
min_delta)` z działającą pętlą treningową, ale **bez** logiki
zapamiętywania i przywracania najlepszego stanu modelu — to jest Twoje
zadanie. Funkcja ma, po zakończeniu, zwrócić model **przywrócony** do
stanu z najlepszej epoki (nie z ostatniej), oraz numer tej epoki.

Zweryfikuj swoją implementację: strata walidacyjna policzona na modelu
*po* wywołaniu funkcji musi dokładnie zgadzać się z zapamiętaną najlepszą
stratą (gotowa asercja to sprawdzi).

## 6. Kontrolowane porównanie i finalna ewaluacja — Twoje zadanie

Na podstawie wyników z sekcji 4–5 wybierz **jedną** konfigurację (która z
nich: brak regularyzacji / weight decay / dropout / early stopping — albo
kombinacja, jeśli masz ku temu podstawę z wyników) o najlepszej stracie
**walidacyjnej**. Wytrenuj ją raz jeszcze (dla pewności, z tym samym
ziarnem) i oceń — **po raz pierwszy w tym laboratorium** — na zbiorze
**testowym**. Wypisz dokładność i stratę testową.

## 7. Wnioski — Twoje zadanie

Odpowiedz (Markdown, uzasadnione rzeczywistymi liczbami z sekcji 4–6):

1. Która metoda regularyzacji dała największą poprawę względem baseline'u
   bez regularyzacji? Czy poprawiła tylko dokładność, tylko stratę, czy
   obie miary — i czy to rozróżnienie ma znaczenie?
2. Wróć do sekcji 1: czy błąd normalizacji przed podziałem dał w Twoim
   przebiegu duży, wyraźny efekt na wyniku końcowym, czy mały? Niezależnie
   od odpowiedzi — dlaczego sama procedura (liczenie statystyk na pełnym
   zbiorze) jest błędna, nawet jeśli w konkretnym przebiegu efekt okazał
   się mały?
3. Jakie jest co najmniej jedno ograniczenie tego eksperymentu (rozmiar
   danych, konkretne ziarno losowości, konkretna architektura), które
   sprawia, że nie powinieneś/aś uogólniać wniosków z sekcji 6 na "zawsze
   najlepsza jest metoda X"?

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---|
| Sekcja 1 (diagnoza przecieku + poprawka) | 20 min |
| Sekcja 2 (orientacja, gotowe) | 5 min |
| Sekcja 3 (pętla treningowa + interpretacja krzywych) | 25 min |
| Sekcja 4 (weight decay + dropout) | 25 min |
| Sekcja 5 (early stopping) | 20 min |
| Sekcja 6 (porównanie + test) | 15 min |
| Sekcja 7 (wnioski) | 10 min |

Sekcje 1–6 to rdzeń obowiązkowy. Jeśli zostanie czas: rozszerz sweep z
sekcji 4 o dodatkowe wartości `weight_decay`/`p`, albo spróbuj kombinacji
dropout + weight decay jednocześnie — to już rozszerzenie, nie rdzeń.

## Kryteria oceny

- Poprawka normalizacji w sekcji 1 faktycznie liczy statystyki tylko z
  `X_train` i stosuje je (bez przeliczania) do pozostałych zbiorów.
- `train_model` i `train_with_early_stopping` działają poprawnie
  (notebook wykonuje się od danej komórki do końca bez błędu).
- Early stopping rzeczywiście przywraca najlepszy stan — asercja w
  notebooku przechodzi, nie jest pominięta/usunięta.
- Zbiór testowy pojawia się w notebooku dokładnie raz, po wyborze
  konfiguracji na podstawie walidacji.
- Interpretacje i wnioski odwołują się do konkretnych liczb/kształtów
  krzywych z Twojego przebiegu, nie do ogólników.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/05_generalization/`) i nie jest częścią tego
repozytorium.
