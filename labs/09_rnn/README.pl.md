Język: [English](README.md) | **Polski**

# Laboratorium 09. Klasyfikacja wiadomości SMS za pomocą RNN, LSTM i GRU

Przewidywany czas wynosi około 2 godzin. Laboratorium korzysta z rekurencji, stanów ukrytych, LSTM, GRU i paddingu opisanych w [`lecture.md`](../../lectures/09_rnn/lecture.md). Zamiast danych syntetycznych analizowany jest rzeczywisty zbiór SMS Spam Collection.

## 0. Konfiguracja

Importy, wybór urządzenia, ziarno losowości i `torch.set_num_threads(4)` są gotowe.

## 1. Dane SMS Spam Collection

Zbiór zawiera 5574 wiadomości z etykietami `ham` i `spam`. Dane są pobierane jako archiwum o rozmiarze około 200 KB. Każdy wiersz pliku tekstowego ma postać `label\ttext`. Szczegóły znajdują się w [`data/registry.yml`](../../data/registry.yml).

Gotowy mechanizm cache zapisuje pobrane i rozpakowane dane w plikach tymczasowych. Weryfikuje SHA-256 archiwum oraz wypakowanego elementu, a następnie atomowo podmienia pliki docelowe. Przerwane pobieranie nie zastępuje więc poprawnego cache częściowym plikiem.

**Zadanie.** Należy policzyć rozkład klas `ham` i `spam` oraz określić, czy zbiór jest zbalansowany. Wynik będzie potrzebny przy wyborze metryk w sekcji 10.

## 2. Tokenizacja

Należy dokończyć `tokenize(text)`. Prosty tokenizer powinien zamieniać tekst na małe litery oraz rozdzielać słowa i interpunkcję. W notebooku znajduje się rozpoczęty wzorzec wyrażenia regularnego.

**Zadanie w Markdown, 2 do 3 zdań.** Trzeba opisać ograniczenia takiej tokenizacji. Warto uwzględnić odmianę i celowo zniekształcone słowa, na przykład `free` oraz `freee`, interpunkcję połączoną ze słowami i brak tokenizacji podwyrazowej omawianej w module 11.

## 3. Podział train, validation i test

Gotowy podział ma proporcje 70/15/15 i stałe ziarno. Dzięki temu wyniki kolejnych eksperymentów pozostają porównywalne.

## 4. Słownik z danych treningowych

Słownik należy zbudować wyłącznie z `train_texts`. Włączenie zbioru walidacyjnego lub testowego spowodowałoby przeciek danych. Jest to odpowiednik zasady stosowanej przy normalizacji w modułach 05-07, lecz dotyczy zbioru tokenów zamiast statystyk liczbowych.

Wymagane są dwa tokeny specjalne.

- `<PAD>` o identyfikatorze $0$ wypełnia batch do wspólnej długości i nie jest treścią.
- `<UNK>` o identyfikatorze $1$ zastępuje token spoza słownika treningowego.

Tokeny występujące w treningu rzadziej niż `MIN_FREQ` razy należy odrzucić. Ogranicza to rozmiar słownika i przeuczenie na rzadkich słowach.

**Zadanie w kodzie.** Token niewidziany podczas treningu ma zostać zamieniony na `<UNK>`. Nie może powodować wyjątku ani zostać pominięty bez informacji.

## 5. Kodowanie i batche o zmiennej długości

Gotowe funkcje `encode()` i `make_batch_tensors()` zamieniają tokeny na identyfikatory. Tworzą również dopełniony batch oraz tensor rzeczywistych długości `lengths`, potrzebny do poprawnej obsługi paddingu w sekcji 7.

Należy sprawdzić kształt tensora dla dwóch wiadomości o różnej długości i wyjaśnić, co stało się z krótszą wiadomością.

Po tej sekcji zdefiniowana jest funkcja `train_eval(model_cls, vocab_size, epochs, lr, seed, **model_kwargs)`. Monitoruje stratę treningową i walidacyjną po każdej epoce, a następnie zwraca wytrenowany model. Wszystkie dalsze modele korzystają z tej samej funkcji.

## 6. Baseline bez kolejności

Należy zbudować klasyfikator uśredniający embeddingi tokenów, bez warstwy rekurencyjnej. Padding nie może uczestniczyć w średniej. Potrzebna jest maska `x != PAD_ID`, ponieważ padding sztucznie osłabia reprezentację krótszych wiadomości.

Gotowe wywołanie `train_eval` trenuje baseline. Jego wynik będzie punktem odniesienia w sekcji 13.

## 7. Klasyfikator `nn.Embedding` z `nn.GRU` lub `nn.LSTM`

Model powinien zawierać `nn.Embedding`, warstwę `nn.GRU` albo `nn.LSTM` oraz warstwę klasyfikacyjną.

Reprezentacją całej sekwencji ma być stan po ostatnim rzeczywistym tokenie, a nie po ostatniej pozycji tensora, która może zawierać padding. Przed warstwą rekurencyjną należy zastosować `torch.nn.utils.rnn.pack_padded_sequence`. Mechanizm przekazuje rzeczywiste długości i zapobiega przetwarzaniu paddingu.

Gotowa asercja porównuje ten sam tekst zakodowany osobno oraz w batchu z dłuższą wiadomością. Obie reprezentacje muszą być identyczne. Błąd w tej asercji zwykle oznacza problem analizowany ponownie w sekcji 11.

## 8. Trening klasyfikatora

Klasyfikator z sekcji 7 należy wytrenować za pomocą gotowego wywołania `train_eval`, używanego wcześniej dla baseline'u.

## 9. Porównanie co najmniej dwóch konfiguracji

Należy porównać przynajmniej dwie konfiguracje. Możliwe warianty obejmują GRU i LSTM, różne wymiary stanu ukrytego, obecność dropoutu lub różną liczbę epok. Wybór wymaga uzasadnienia w 2 do 3 zdaniach. Wszystkie konfiguracje muszą korzystać z tego samego podziału danych i tych samych metryk.

## 10. Ewaluacja na zbiorze testowym

Lepszą konfigurację wybiera się na podstawie walidacji. Dopiero potem, pierwszy raz w laboratorium, wykonywana jest ewaluacja na zbiorze testowym. Wymagane są trzy wyniki.

- accuracy,
- macro-F1,
- macierz pomyłek.

Macro-F1 jest istotne przy niezbalansowanym zbiorze. Model przewidujący zawsze `ham` mógłby uzyskać wysoką accuracy, choć nie byłby użyteczny.

## 11. Zadanie diagnostyczne

Kod w notebooku nie zgłasza wyjątku i może się trenować, ale błędnie obsługuje padding.

Należy:

1. wskazać konkretny błąd i linię, w której występuje,
2. wyjaśnić, kiedy błąd się ujawnia, porównując `batch_size=1` z batchem sekwencji o różnych długościach,
3. poprawić kod,
4. sprawdzić sekwencje o różnych długościach za pomocą asercji z sekcji 7.

## 12. Analiza błędów

Dla modelu z sekcji 10 należy wypisać od 5 do 8 błędnie sklasyfikowanych wiadomości wraz z prawdziwą i przewidzianą etykietą. Analiza powinna określić, czy częściej występują błędy `ham` na `spam`, czy `spam` na `ham`. Porównanie z długością wiadomości jest rozszerzeniem opcjonalnym.

## 13. Wnioski

Odpowiedzi powinny opierać się na liczbach z sekcji 6-12.

1. O ile GRU lub LSTM przewyższa baseline bez kolejności? Wielkość różnicy należy odnieść do roli kolejności w klasyfikacji spamu i porównać z eksperymentem C z wykładu.
2. Która konfiguracja z sekcji 9 wypadła lepiej na walidacji? Czy różnica z pojedynczego przebiegu jest wystarczająca do sformułowania mocnego wniosku?
3. Jakie ograniczenie, na przykład rozmiar danych, prosta tokenizacja lub pojedyncze ziarno, utrudnia uogólnienie wyników?

## Orientacyjny podział czasu

| Blok | Czas |
|---|---|
| Sekcje 0-1, konfiguracja i eksploracja | 10 min |
| Sekcja 2, tokenizacja i ograniczenia | 10 min |
| Sekcje 3-5, słownik, PAD/UNK i kodowanie | 10 min |
| Sekcja 6, baseline bez kolejności | 15 min |
| Sekcja 7, klasyfikator i padding | 25 min |
| Sekcja 8, trening | 5 min |
| Sekcja 9, porównanie konfiguracji | 10 min |
| Sekcja 10, ewaluacja testowa | 10 min |
| Sekcja 11, diagnostyka paddingu | 10 min |
| Sekcja 12, analiza błędów | 5 min |
| Sekcja 13, wnioski | 10 min |

Sekcje 0-13 stanowią obowiązkowy rdzeń. Trening na CPU trwa pojedyncze sekundy, więc większość czasu powinna zostać przeznaczona na poprawną obsługę sekwencji i analizę.

Opcjonalne rozszerzenia obejmują porównanie długości poprawnie i błędnie sklasyfikowanych wiadomości, dwukierunkowy GRU lub LSTM z `bidirectional=True` oraz osobne accuracy dla krótkich, poniżej 10 tokenów, i długich, powyżej 30 tokenów, wiadomości.

## Kryteria oceny

- Słownik powstaje wyłącznie z `train_texts`, a `<UNK>` poprawnie obsługuje tokeny spoza słownika.
- Klasyfikator z sekcji 7 używa `pack_padded_sequence` albo równoważnego mechanizmu. Asercja niezależności reprezentacji od paddingu przechodzi.
- Baseline z sekcji 6 pomija padding podczas uśredniania.
- Ewaluacja testowa zawiera accuracy, macro-F1 i macierz pomyłek.
- Zbiór testowy jest używany dopiero po wyborze konfiguracji na podstawie walidacji.
- Diagnoza z sekcji 11 wskazuje konkretną przyczynę błędu.
- Wnioski odwołują się do liczb uzyskanych podczas eksperymentu.

Pełne rozwiązanie wzorcowe znajduje się lokalnie u prowadzącego w `decisions/solutions/09_rnn/` i nie jest częścią repozytorium.
