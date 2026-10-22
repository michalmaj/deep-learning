Język: [English](README.md) | **Polski**

# Laboratorium 10. Mały Transformer do klasyfikacji sekwencji

Przewidywany czas wynosi około 2 godzin. Obowiązkowy rdzeń zajmuje 100-105 minut, a bufor 15-20 minut. Laboratorium wykorzystuje scaled dot-product attention, multi-head attention, maskowanie, positional encoding i blok Transformer z [`lecture.md`](../../lectures/10_attention/lecture.md).

Zadanie jest trudniejsze od przykładu wykładowego. Sekwencje mają różne długości i zawierają prawdziwy padding oraz losowe dystraktory.

Sekcje 0-10 stanowią obowiązkowy rdzeń. Sekcje R1-R4 są opcjonalne i nie wpływają na ocenę rdzenia.

## Zadanie

Każda sekwencja ma losową długość od 10 do 40. Zawiera tokeny `A` i `B` w losowych pozycjach oraz dystraktory na pozostałych miejscach. Etykieta określa, czy `A` występuje przed `B`.

Długość jest losowana niezależnie od etykiety. Model nie powinien więc rozwiązać zadania za pomocą zależności między klasą a długością.

## 0. Konfiguracja

Importy, urządzenie, ziarno losowości i `torch.set_num_threads(4)` są gotowe.

## 1. Generator danych

Gotowa funkcja `make_dataset(n, seed)` zwraca `(X, lengths, y)`. Tensor `X` ma kształt `(n, MAX_LEN)` i jest wypełniony `PAD_ID` poza rzeczywistą długością. `lengths` zawiera długości, a `y` etykiety.

**Zadanie.** Na zbiorze treningowym należy sprawdzić:

1. balans klas, oczekiwany w pobliżu 50/50,
2. średnią długość w obu klasach i brak istotnej różnicy między nimi.

Drugi punkt wyklucza prostą heurystykę opartą na długości zamiast kolejności `A` i `B`.

## 2. Baseline bez kolejności

Należy zbudować klasyfikator uśredniający embeddingi tokenów bez attention. Padding musi zostać pominięty. Gotowy helper `masked_mean_pool(x, real_mask, lengths)` realizuje zasadę znaną z modułu 9. Trzeba użyć go w odpowiednim miejscu `MeanPoolBaseline.forward`.

Gotowa pętla trenuje baseline. Wynik będzie punktem odniesienia w sekcji 10.

## 3. Positional encoding

Funkcja `sinusoidal_positional_encoding(seq_len, d_model)` jest gotowa. Należy zdecydować, gdzie dodać jej wynik w architekturze z sekcji 6.

## 4. Maska paddingu

Z `x` i `PAD_ID` należy zbudować `key_padding_mask`. Dla `nn.MultiheadAttention` wartość `True` oznacza padding, który ma zostać zignorowany. Jest to konwencja odwrotna do boolowskiego `attn_mask` w `F.scaled_dot_product_attention`.

Sekwencja bez paddingu powinna mieć maskę złożoną wyłącznie z `False`. Odwrócenie konwencji prowadzi do błędu analizowanego w sekcji 9.

## 5. Blok Transformer

`TinyTransformerBlock` powinien zawierać:

1. `nn.MultiheadAttention` z maską z sekcji 4,
2. połączenie rezydualne i LayerNorm,
3. FFN zbudowaną jako Linear, ReLU, Linear,
4. drugie połączenie rezydualne i LayerNorm.

Jest to wariant Post-LN. Architektura ma pozostać widoczna, dlatego nie należy zastępować jej pojedynczym `nn.TransformerEncoderLayer`. Projekcje Q, K i V zapewnia `nn.MultiheadAttention`.

## 6. Agregacja i klasyfikator

`Classifier` powinien łączyć `nn.Embedding`, opcjonalne positional encoding, `TinyTransformerBlock`, maskowaną agregację i warstwę klasyfikacyjną.

Agregacja używa `masked_mean_pool`. `real_mask` jest negacją `key_padding_mask`, a mianownik powinien pochodzić z `lengths`, nie z `MAX_LEN`.

Gotowa asercja porównuje tę samą sekwencję przetworzoną osobno oraz w batchu z dłuższą sekwencją. Reprezentacje muszą być identyczne. Różnica wskazuje błąd w masce albo agregacji.

## 7. Trening z positional encoding i bez niego

Gotowa funkcja `train_eval(model, epochs, lr, batch_size, seed)` obsługuje wszystkie konfiguracje. Należy wytrenować `Classifier` z `use_pos_enc=True` oraz `use_pos_enc=False` przy tym samym budżecie. Modele powinny zostać zapisane jako `model_with_pe` i `model_without_pe`.

`model_with_pe` jest później używany w sekcji 8 oraz rozszerzeniach R1 i R2. Nie wymaga ponownego treningu.

Gotowa pętla powtarza obie konfiguracje dla ziaren 0, 1 i 2. Interpretacja w sekcji 10 powinna określić, czy wynik bez PE na poziomie losowym jest systematyczny.

## 8. Ewaluacja testowa

`model_with_pe` należy ocenić na zbiorze testowym pierwszy raz w tej sekcji. Wymagane są accuracy oraz rozkład długości błędnie sklasyfikowanych przykładów.

## 9. Zadanie diagnostyczne

`BuggyClassifier` definiuje maskę i trenuje się bez wyjątku, ale stosuje odwróconą konwencję maski.

Należy:

1. wskazać błędną linię i wyjaśnić problem,
2. uruchomić trening i sprawdzić stratę, zwracając uwagę na sekwencje długości `MAX_LEN`,
3. wyjaśnić zachowanie softmaxu, gdy żaden klucz nie jest dozwolony,
4. poprawić konwencję maski,
5. potwierdzić za pomocą `torch.isfinite`, że strata jest skończona, oraz porównać accuracy z poprawnym modelem.

Dla sekwencji bez paddingu odwrócona maska blokuje wszystkie pozycje. Softmax otrzymuje wtedy wiersz samych wartości $-\infty$, co w jawnej implementacji prowadzi do `NaN`.

## 10. Wnioski

Odpowiedzi powinny korzystać z wyników sekcji 2 i 7-9.

1. Które elementy odpowiadają za mieszanie informacji, pozycję oraz redukcję sekwencji do jednego wektora?
2. Jaka jest różnica accuracy między baseline'em a Transformerem z PE i co mówi o znaczeniu kolejności?
3. Czy wynik bez PE jest stabilny dla trzech ziaren?
4. Czy asercja z sekcji 6 potwierdza poprawne pomijanie paddingu?
5. Co eksperyment pokazuje o attention, a czego nie dowodzi? Wysoka accuracy nie jest sama w sobie dowodem rozumienia relacji `A` i `B`.
6. Jakie ograniczenie utrudnia uogólnienie wyników na inne zadania sekwencyjne?

## Rozszerzenia opcjonalne

Wszystkie rozszerzenia wykorzystują `model_with_pe` bez ponownego treningu.

- **R1.** Accuracy walidacyjna w co najmniej trzech przedziałach długości, na przykład $[10,19]$, $[20,29]$ i $[30,40]$.
- **R2.** Wagi attention dla poprawnego przykładu i zapytania w pozycji `A`. Duża waga skierowana na `B` nie stanowi dowodu rozumienia relacji.
- **R3.** Stos dwóch bloków `TinyTransformerBlock` i porównanie accuracy lub stabilności.
- **R4.** Zmiana Post-LN na Pre-LN i porównanie stabilności treningu.

## Orientacyjny podział czasu

| Blok | Czas |
|---|---:|
| Sekcje 0-1, konfiguracja, dane i balans | 8 min |
| Sekcja 2, baseline | 8 min |
| Sekcje 3-4, positional encoding i maska | 12 min |
| Sekcja 5, blok Transformer | 20 min |
| Sekcja 6, agregacja, klasyfikator i asercja | 13 min |
| Sekcja 7, trening i trzy ziarna | 12 min |
| Sekcja 8, ewaluacja testowa | 6 min |
| Sekcja 9, diagnostyka | 13 min |
| Sekcja 10, wnioski | 8 min |
| **Rdzeń** | **100 min** |
| Bufor | 15-20 min |
| **Całość** | **około 120 min** |

Przy braku czasu priorytet mają sekcje 4-6 oraz 9. Pełne rozwiązanie wraz z trzema ziarnami i diagnostyką wykonuje się na CPU w czasie krótszym niż minuta. Czas zajęć powinien zostać przeznaczony głównie na implementację maskowania, agregację i analizę.

## Kryteria oceny

- Balans klas i niezależność długości od etykiety są potwierdzone liczbowo.
- Baseline pomija padding podczas uśredniania.
- `key_padding_mask` używa konwencji `True` oznaczającej padding.
- Blok jawnie zawiera attention, residual, LayerNorm, FFN, residual i LayerNorm.
- Agregacja wyklucza padding, a asercja z sekcji 6 przechodzi.
- Porównanie z PE i bez PE obejmuje kilka ziaren.
- Zbiór testowy jest używany dopiero w sekcji 8 i wyłącznie do oceny.
- Diagnoza wskazuje odwróconą maskę i całkowicie zamaskowany wiersz dla sekwencji bez paddingu.
- Wnioski odwołują się do wyników i określają także granice eksperymentu.

Pełne rozwiązanie wzorcowe znajduje się lokalnie u prowadzącego w `decisions/solutions/10_attention/` i nie jest częścią repozytorium.
