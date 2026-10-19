Język: [English](README.md) | **Polski**

# Laboratorium 09: klasyfikacja wiadomości SMS (spam/ham) z RNN/LSTM/GRU

Czas: ok. 2 godziny. Bazuje na [`lectures/09_rnn/lecture.md`](../../lectures/09_rnn/lecture.md)
(rekurencja, $h_t$, LSTM/GRU, padding), ale na **innym problemie** niż wykład —
prawdziwa klasyfikacja tekstu (SMS Spam Collection), nie syntetyczne sekwencje.

## 0. Konfiguracja (gotowe)

Import bibliotek, urządzenie, ziarno losowości, `torch.set_num_threads(4)`.

## 1. Dane: SMS Spam Collection (gotowe + krótkie zadanie)

5574 wiadomości SMS, etykieta `ham`/`spam`, pobierane bezpośrednio (plik tekstowy
`label\ttext` na linię, zip ~200KB, SHA-256 zweryfikowany w kodzie — zobacz
[`data/registry.yml`](../../data/registry.yml)).

**Zadanie:** policz i wypisz rozkład klas (`ham`/`spam`). Zbiór jest zbalansowany czy
nie? Zanotuj — to ma znaczenie dla wyboru metryk w sekcji 10.

## 2. Tokenizacja — Twoje zadanie

Zaimplementuj `tokenize(text)`: prosty podział na tokeny (słowa + interpunkcja jako
osobne tokeny, małe litery). Gotowy wzorzec wyrażenia regularnego czeka na
dokończenie.

**Zadanie (Markdown, 2-3 zdania):** jakie są ograniczenia tak prostej tokenizacji?
Pomyśl o odmianie słów (np. "free" vs "freee" w spamie celowo zniekształconym), o
interpunkcji sklejonej ze słowami, i o tym, czego nauczymy się o tokenizacji
podslownej (*subword*) dopiero w module 11.

## 3. Podział train/val/test (gotowe)

70/15/15, ziarno ustalone — żeby wyniki były porównywalne między Twoimi
eksperymentami.

## 4. Słownik: tylko ze zbioru treningowego — Twoje zadanie

Zbuduj słownik **wyłącznie** z `train_texts` (nigdy z val/test — dokładnie ta sama
zasada co normalizacja w modułach 05-07, tylko tu dotyczy zbioru tokenów, nie
statystyk liczbowych). Dodaj dwa specjalne tokeny:

- `<PAD>` (id $0$) — wypełnienie do wspólnej długości w batchu, **nie** treść,
- `<UNK>` (id $1$) — token nieznany (słowo spoza słownika treningowego, np. rzadkie
  słowo widziane tylko w val/test).

Odrzuć tokeny występujące w treningu rzadziej niż `MIN_FREQ` razy (zmniejsza
słownik, zmniejsza przeuczenie na rzadkich słowach).

**Zadanie (kod):** zweryfikuj, że żaden token ze zbioru walidacyjnego/testowego, który
nie wystąpił w treningu, nie powoduje błędu (powinien zostać zamieniony na `<UNK>`,
nie rzucić wyjątkiem ani zostać pominięty po cichu).

## 5. Tekst → sekwencja identyfikatorów i batche o zmiennej długości (gotowe)

Funkcje `encode()` i `make_batch_tensors()` zamieniają listę tokenów na identyfikatory
i budują padded batch **razem z tensorem rzeczywistych długości** `lengths` — ten
tensor jest kluczowy dla poprawnej obsługi paddingu w sekcji 7.

Sprawdź: dwie wiadomości o różnej długości w tym samym batchu — jaki jest kształt
wynikowego tensora? Co się stało z krótszą wiadomością?

Tuż za tą sekcją notebook definiuje `train_eval(model_cls, vocab_size, epochs, lr,
seed, **model_kwargs)` (gotowe, używane przez wszystkie modele w dalszej części
laboratorium) — monitoruje stratę treningową i walidacyjną po każdej epoce i zwraca
wytrenowany model.

## 6. Baseline bez kolejności — Twoje zadanie

Zbuduj klasyfikator oparty na **uśrednieniu embeddingów tokenów** (bez żadnej
warstwy rekurencyjnej) — dokładnie jak w wykładzie, eksperyment C, ale tu uśrednienie
musi pomijać padding (zobacz wykład: padding to nie treść). Podpowiedź: potrzebujesz
maski `x != PAD_ID` przy liczeniu średniej, inaczej padding sztucznie "rozwadnia"
reprezentację krótszych wiadomości. Wytrenuj go od razu (`train_eval`, gotowe
wywołanie) — ta liczba będzie Twoim punktem odniesienia w sekcji 13.

## 7. Klasyfikator `nn.Embedding` + `nn.GRU`/`nn.LSTM` — Twoje zadanie

Zbuduj model: `nn.Embedding` → `nn.GRU` (lub `nn.LSTM`) → warstwa klasyfikacyjna.

**To jest najważniejsza decyzja projektowa w tym laboratorium:** reprezentacją całej
sekwencji do klasyfikacji musi być stan **po ostatnim rzeczywistym tokenie**, nie po
ostatniej pozycji tensora (która może być paddingiem). Użyj
`torch.nn.utils.rnn.pack_padded_sequence` przed przekazaniem do warstwy rekurencyjnej
— to mechanizm, który informuje PyTorch o rzeczywistych długościach, żeby warstwa nie
przetwarzała paddingu w ogóle.

Zweryfikuj swoją implementację (gotowa asercja czeka): **ten sam tekst**, zakodowany
osobno i zbatchowany razem z dłuższą wiadomością (więc inaczej dopełniony paddingiem),
musi dać **identyczną** reprezentację w obu przypadkach. Jeśli asercja nie przechodzi —
masz dokładnie ten błąd, który jest tematem sekcji diagnostycznej (sekcja 11) — ale tu
oczekujemy, że Twoja implementacja jest już poprawna.

## 8. Trening klasyfikatora z sekcji 7 (pętla z sekcji 5 — gotowe wywołanie)

Wytrenuj swój klasyfikator z sekcji 7 przez `train_eval` (ta sama funkcja, co w
sekcji 6 dla baseline'u).

## 9. Porównanie ≥2 konfiguracji — Twoje zadanie

Porównaj **co najmniej dwie** konfiguracje — do wyboru (uzasadnij w 2-3 zdaniach,
dlaczego akurat te): GRU vs LSTM, różny wymiar stanu ukrytego, obecność/brak
dropoutu, różna liczba epok. Użyj **tego samego** podziału danych i tych samych
metryk dla każdej.

## 10. Ewaluacja na zbiorze testowym — Twoje zadanie

Wybierz lepszą (według walidacji) konfigurację z sekcji 9. Oceń ją na zbiorze
testowym (**pierwszy raz** w tym laboratorium) przy użyciu:

- accuracy,
- **macro-F1** (zbiór jest niezbalansowany — sekcja 1 — accuracy sama w sobie myli:
  model, który zawsze przewiduje `ham`, miałby wysoką accuracy i zero użyteczności),
- macierzy pomyłek.

## 11. Zadanie diagnostyczne

Poniższy kod (w notebooku) **wygląda** na poprawny — nie rzuca wyjątku, trenuje się,
daje jakąś accuracy — ale ma błąd związany z obsługą paddingu.

**Zadanie:**
1. Zdiagnozuj błąd — na czym dokładnie polega, w której linii.
2. Wyjaśnij, dlaczego jest istotny (kiedy się ujawnia, a kiedy nie — podpowiedź:
   sprawdź dla batch_size=1 i dla batcha z sekwencjami o różnej długości).
3. Popraw kod.
4. Sprawdź zachowanie na sekwencjach różnej długości (gotowa asercja z sekcji 7
   powinna teraz przechodzić).

## 12. Analiza błędów — Twoje zadanie

Dla modelu z sekcji 10: wypisz 5-8 błędnie sklasyfikowanych wiadomości (tekst +
prawdziwa/przewidziana etykieta). Czy błędy koncentrują się po jednej stronie (więcej
`ham`→`spam` czy `spam`→`ham`)? Czy krótkie wiadomości są mylone częściej niż długie
(policz średnią długość błędnie i poprawnie sklasyfikowanych)?

## 13. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami z sekcji 6-12:

1. O ile model sekwencyjny (GRU/LSTM) przebija baseline bez kolejności? Czy ta
   różnica jest duża, czy skromna — i co to mówi o tym, na ile ten **konkretny**
   problem (klasyfikacja spamu) faktycznie wymaga rozumienia kolejności słów, w
   porównaniu z zadaniem z wykładu (eksperyment C), które było tak skonstruowane,
   żeby kolejność była jedyną rozstrzygającą informacją?
2. Która z dwóch porównanych konfiguracji (sekcja 9) wypadła lepiej na walidacji?
   Czy różnica jest wystarczająco duża, żeby zaufać jej na pojedynczym przebiegu?
3. Jakie jest co najmniej jedno ograniczenie tego eksperymentu (rozmiar danych,
   prosta tokenizacja, pojedyncze ziarno), które nie pozwala uogólnić wniosków?

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, eksploracja) | 10 min |
| Sekcja 2 (tokenizacja + ograniczenia) | 10 min |
| Sekcje 3-5 (słownik, PAD/UNK, kodowanie — gotowe/krótkie) | 15 min |
| Sekcja 6 (baseline bez kolejności) | 15 min |
| Sekcja 7 (klasyfikator GRU/LSTM + poprawny padding) | 25 min |
| Sekcja 8 (trening) | 5 min |
| Sekcja 9 (porównanie ≥2 konfiguracji) | 10 min |
| Sekcja 10 (ewaluacja testowa) | 10 min |
| Sekcja 11 (zadanie diagnostyczne) | 15 min |
| Sekcja 12 (analiza błędów) | 10 min |
| Sekcja 13 (wnioski) | 10 min |

Sekcje 0-12 to rdzeń obowiązkowy. Trening na tym zbiorze trwa pojedyncze sekundy na
CPU — budżet czasowy ma iść w poprawną implementację obsługi sekwencji i analizę, nie
w czekanie na trening.

**Rozszerzenie (opcjonalne, jeśli zostanie czas):** dwukierunkowy GRU/LSTM
(`bidirectional=True`) — czy pomaga na tym zadaniu? Albo: osobna analiza accuracy dla
krótkich (<10 tokenów) i długich (>30 tokenów) wiadomości.

## Kryteria oceny

- Słownik zbudowany wyłącznie z `train_texts`; `<UNK>` poprawnie obsługuje tokeny
  spoza słownika bez błędu.
- Klasyfikator z sekcji 7 używa `pack_padded_sequence` (albo równoważnego poprawnego
  mechanizmu) — asercja "ten sam tekst, ta sama reprezentacja niezależnie od
  paddingu" przechodzi.
- Baseline z sekcji 6 poprawnie pomija padding przy uśrednianiu (nie tylko przy
  klasyfikatorze rekurencyjnym).
- Ewaluacja testowa (sekcja 10) używa accuracy **i** macro-F1 **i** confusion matrix,
  nie samej accuracy.
- Zbiór testowy pojawia się dopiero w sekcji 10, po wyborze konfiguracji na
  podstawie walidacji.
- Zadanie diagnostyczne (sekcja 11) wskazuje konkretną przyczynę, nie ogólnik.
- Wnioski odwołują się do konkretnych liczb z Twojego przebiegu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/09_rnn/`) i nie jest częścią tego repozytorium.
