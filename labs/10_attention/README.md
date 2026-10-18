# Laboratorium 10: mały Transformer do klasyfikacji sekwencji zależnych od kolejności

Czas: ok. 2 godziny (**100-105 minut rdzenia obowiązkowego + 15-20 minut
bufora**). Bazuje na
[`lectures/10_attention/lecture.md`](../../lectures/10_attention/lecture.md)
(scaled dot-product attention, multi-head attention, maskowanie, positional
encoding, blok Transformer), ale na **innym, trudniejszym zadaniu** niż wykład:
sekwencje o **zmiennej długości** (prawdziwy padding, czego wykładowy eksperyment D
nie wymagał) z **dystraktorami** (losowymi tokenami niosącymi zero informacji).

**Sekcje 0-10 to rdzeń obowiązkowy. Sekcje R1-R4 (na końcu notebooka) są
opcjonalnymi rozszerzeniami** — wykonaj je tylko, jeśli zostanie czas po
zakończeniu rdzenia. Żadne kryterium oceny rdzenia nie wymaga ukończenia
rozszerzeń.

## Zadanie

Każda sekwencja (długość losowa z zakresu $[10,40]$) zawiera dokładnie dwa
wyróżnione tokeny, `A` i `B`, w losowych pozycjach, oraz same losowe tokeny
rozpraszające (dystraktory) na resztę pozycji. **Etykieta = czy `A` występuje przed
`B`.** Długość sekwencji jest losowana **niezależnie** od etykiety (zweryfikuj to w
sekcji 1) — model nie może "oszukać" zadania heurystyką ignorującą kolejność (np.
"krótsze sekwencje = klasa 0").

## 0. Konfiguracja (gotowe)

Import bibliotek, urządzenie, ziarno losowości, `torch.set_num_threads(4)`.

## 1. Dane: generator zadania (gotowe + krótkie zadanie)

`make_dataset(n, seed)` (gotowe) zwraca `(X, lengths, y)`: `X` to tensor
`(n, MAX_LEN)` wypełniony `PAD_ID` poza rzeczywistą długością każdej sekwencji,
`lengths` to rzeczywiste długości, `y` to etykiety.

**Zadanie:** zweryfikuj na zbiorze treningowym:
1. balans klas (ok. 50/50),
2. że średnia długość sekwencji **nie różni się** istotnie między klasami (gdyby się
   różniła, model mógłby "rozwiązywać" zadanie samą długością, nie kolejnością `A`/`B`
   — dokładnie pułapka, przed którą ostrzega brief tego laboratorium).

## 2. Baseline bez kolejności — Twoje zadanie

Zbuduj klasyfikator oparty na uśrednieniu embeddingów tokenów (bez attention) —
**musi pomijać padding** przy uśrednianiu. Dany jest helper `masked_mean_pool(x,
real_mask, lengths)` — dokładnie ta sama zasada jak w module 9, sekcja 6 (już
opanowana, nie reimplementujemy jej od zera) — Ty decydujesz, **gdzie i jak** jej
użyć w `MeanPoolBaseline.forward`. Wytrenuj go od razu (pętla treningowa jest
gotowa) — to Twój punkt odniesienia w sekcji 10.

## 3. Positional encoding (gotowe)

Funkcja `sinusoidal_positional_encoding(seq_len, d_model)` z wykładu — nie
reimplementujemy jej (już w pełni zweryfikowana tam), tylko **decydujemy, jak i
gdzie** ją dołączyć w architekturze (sekcja 6).

## 4. Maska paddingu — Twoje zadanie

Zbuduj `key_padding_mask` z `x` i `PAD_ID`. **Krytyczne:** używasz
`nn.MultiheadAttention`, więc zgodnie z wykładem (sekcja "Ważna uwaga
implementacyjna") `key_padding_mask` musi mieć konwencję **`True` = to jest padding,
zignoruj** — **nie** `True` = dozwolone (ta konwencja jest **odwrotna** względem
`F.scaled_dot_product_attention`). Pomyl te konwencje i trafisz prosto w zadanie
diagnostyczne z sekcji 9 — zanim do niego dojdziesz, zweryfikuj własną maskę: jedna
sekwencja bez paddingu powinna mieć maskę całkowicie `False`.

## 5. Blok Transformer — Twoje zadanie

Zbuduj `TinyTransformerBlock`: self-attention (`nn.MultiheadAttention` — projekcje
Q/K/V nie trzeba pisać ręcznie, to już zweryfikowane na wykładzie) **+ połączenie
rezydualne + LayerNorm + feed-forward (Linear→ReLU→Linear) + drugie połączenie
rezydualne + drugi LayerNorm** (Post-LN, jak w `lecture.md`). To **Ty** łączysz te
elementy — nie chowaj architektury za jednym wywołaniem `nn.TransformerEncoderLayer`.
Pamiętaj przekazać `key_padding_mask` z sekcji 4 do `self.mha(...)`.

## 6. Agregacja i klasyfikator — Twoje zadanie

Zbuduj `Classifier`: `nn.Embedding` (+ positional encoding z sekcji 3, jeśli
`use_pos_enc=True`) → `TinyTransformerBlock` z sekcji 5 → **agregacja do jednego
wektora na sekwencję** (użyj `masked_mean_pool` z sekcji 2 — ta sama zasada: real_mask
to negacja `key_padding_mask`, dzielenie przez `lengths`, nie przez `MAX_LEN`) →
warstwa klasyfikacyjna.

**Weryfikacja (gotowa asercja czeka w notebooku):** ta sama sekwencja, przetworzona
**osobno** i **zbatchowana** z inną, dłuższą sekwencją (więc inaczej dopełniona
paddingiem), musi dać **identyczną** reprezentację w obu przypadkach. Jeśli nie
przechodzi — masz błąd w masce (sekcja 4) albo w agregacji (ta sekcja), zanim
dojdziesz do treningu.

## 7. Trening (gotowe) i porównanie z vs bez positional encoding — Twoje zadanie

`train_eval(model, epochs, lr, batch_size, seed)` jest gotowe (ta sama pętla dla
wszystkich konfiguracji w dalszej części laboratorium). Wytrenuj **dwie**
konfiguracje Twojego `Classifier` — `use_pos_enc=True` i `use_pos_enc=False` — tym
samym budżetem treningowym, i zapisz je jako `model_with_pe`/`model_without_pe`.
**`model_with_pe` jest używany ponownie w sekcji 8 i w rozszerzeniach R1/R2 —
nie trenujesz go drugi raz.**

Dana (gotowa) jest też krótka pętla po **3 ziarnach** (0, 1, 2), uruchamiająca
`train_eval` dla obu wariantów — trening trwa pojedyncze sekundy na CPU, więc
uruchomienie jej 6× zajmuje mniej czasu niż jej napisanie od zera. **Twoje zadanie
to interpretacja** wyniku (sekcja 10, pytanie 3): czy "bez PE = poziom przypadku"
jest systematyczne, czy to przypadek jednego nieudanego przebiegu?

## 8. Ewaluacja na zbiorze testowym — Twoje zadanie

Oceń `model_with_pe` z sekcji 7 (**pierwszy raz** w tym laboratorium używasz tu
zbioru testowego do **oceny**, nie do ponownego treningu): accuracy + rozkład
długości błędnie sklasyfikowanych przykładów (czy błędy koncentrują się przy
konkretnych długościach?).

## 9. Zadanie diagnostyczne

Poniższy kod (w notebooku, `BuggyClassifier`) **wygląda** poprawnie — definiuje
maskę, buduje blok, trenuje się bez wyjątku — ale ma błąd związany z **konwencją
maski**, dokładnie tą, przed którą ostrzega sekcja 4.

**Zadanie:**
1. Zdiagnozuj błąd — która linia, dlaczego.
2. Uruchom trening i zaobserwuj, co się dzieje ze stratą (podpowiedź: sprawdź, czy w
   zbiorze treningowym są sekwencje **bez żadnego paddingu**, tj. o długości
   `MAX_LEN` — i co się dzieje z maskowaniem attention, gdy **wszystkie** pozycje
   takiej sekwencji wypadają "zablokowane" przez odwróconą konwencję).
3. Wyjaśnij matematycznie, nawiązując do sekcji "Wiersz z wszystkimi kluczami
   zamaskowanymi" w `lecture.md`: co dokładnie dzieje się z `softmax`, gdy dla danego
   zapytania zero kluczy jest dozwolonych?
4. Popraw kod (odwróć konwencję maski).
5. Zweryfikuj, że po poprawce strata jest zawsze skończona (`torch.isfinite`) i że
   model trenuje się do accuracy zbliżonej do oryginalnej, poprawnej implementacji.

## 10. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami z Twojego przebiegu (sekcje 2, 7-9):

1. Czy potrafisz ręcznie wskazać, które elementy architektury (sekcja 5-6) odpowiadają
   za: (a) mieszanie informacji między pozycjami, (b) wprowadzenie informacji o
   pozycji, (c) redukcję sekwencji do jednego wektora?
2. Jak duża jest różnica accuracy między baseline'em (sekcja 2) a Twoim
   Transformerem **z** PE (sekcja 7) — i co to mówi o tym, czy zadanie faktycznie
   wymaga kolejności?
3. Czy wynik "bez PE = poziom przypadku" jest stabilny na kilku ziarnach (sekcja 7,
   gotowa pętla), czy to przypadek jednego przebiegu?
4. Czy maskowanie jest poprawne — czy masz dowód (asercja z sekcji 6), że model nie
   korzysta z informacji zawartej w paddingu?
5. Co dokładnie pokazuje (a czego **nie** pokazuje) ten eksperyment o mechanizmie
   attention? Czy wysoka accuracy sama w sobie dowodzi, że model "rozumie" relację
   porządkową `A`/`B`, czy tylko że skutecznie minimalizuje stratę na tym konkretnym
   zbiorze?
6. Jakie jest co najmniej jedno ograniczenie tego eksperymentu, które nie pozwala
   uogólnić wniosków na dowolne zadania sekwencyjne?

## Rozszerzenia (opcjonalne — R1-R4, jeśli zostanie czas)

Wszystkie reużywają `model_with_pe` z sekcji 7 — **bez** ponownego treningu.

- **R1.** Accuracy na zbiorze walidacyjnym w co najmniej 3 przedziałach długości
  (np. $[10,19]$, $[20,29]$, $[30,40]$) — czy skuteczność jest stabilna, czy spada
  dla dłuższych/krótszych sekwencji?
- **R2.** Dla jednego poprawnie sklasyfikowanego przykładu: wagi attention
  (zapytanie = pozycja `A`, dowolna głowa). **Nie interpretuj** wysokiej wagi w
  stronę `B` jako "dowód", że model "rozumie" relację porządkową.
- **R3.** Ułóż **dwa** `TinyTransformerBlock` jeden po drugim (stos) — czy poprawia
  to accuracy lub stabilność na dłuższych sekwencjach?
- **R4.** Zamień Post-LN na Pre-LN (norma przed podwarstwą, nie po) i sprawdź, czy
  coś się zmienia w stabilności treningu.

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---:|
| Sekcje 0-1 (konfiguracja, dane, weryfikacja balansu) | 8 min |
| Sekcja 2 (baseline bez kolejności) | 8 min |
| Sekcje 3-4 (positional encoding — przegląd, maska paddingu) | 12 min |
| Sekcja 5 (blok Transformer) | 20 min |
| Sekcja 6 (agregacja + klasyfikator + weryfikacja) | 13 min |
| Sekcja 7 (trening, porównanie z/bez PE, multi-seed) | 12 min |
| Sekcja 8 (ewaluacja testowa) | 6 min |
| Sekcja 9 (zadanie diagnostyczne) | 13 min |
| Sekcja 10 (wnioski) | 8 min |
| **Rdzeń (sekcje 0-10)** | **100 min** |
| Bufor | 15-20 min |
| **Całość** | **ok. 120 min** |

Priorytet, jeśli czasu zabraknie nawet w rdzeniu: sekcje 4-6 (rdzeń architektury) i
9 (diagnostyka) — to są elementy, które nie mają sensownego skróconego wariantu.
Trening trwa pojedyncze sekundy na CPU na jeden przebieg (zweryfikowane: pełne
rozwiązanie wzorcowe, łącznie z pętlą po 3 ziarnach i diagnostyką, wykonuje się w
poniżej minuty) — budżet czasowy idzie w poprawną implementację maskowania/
agregacji i analizę, nie w czekanie na trening.

**Rozszerzenia R1-R4 są opcjonalne** i nie są wliczone do rdzenia — zobacz sekcję
wyżej.

## Kryteria oceny

Wszystkie poniższe kryteria dotyczą **rdzenia (sekcje 0-10)** — żadne nie wymaga
ukończenia rozszerzeń R1-R4.

- Balans klas i niezależność długości sekwencji od etykiety zweryfikowane liczbowo
  (sekcja 1).
- Baseline (sekcja 2) poprawnie pomija padding przy uśrednianiu.
- `key_padding_mask` (sekcja 4) używa poprawnej konwencji dla `nn.MultiheadAttention`
  (`True` = padding).
- Blok Transformer (sekcja 5) zbudowany z widocznych elementów (attention + residual
  + LayerNorm + FFN + residual + LayerNorm) — nie jedno wywołanie
  `nn.TransformerEncoderLayer`.
- Agregacja (sekcja 6) wyklucza padding — asercja "ta sama sekwencja, ten sam wynik
  niezależnie od paddingu w batchu" przechodzi.
- Porównanie z/bez PE (sekcja 7) sprawdzone na kilku ziarnach (gotowa pętla), nie
  jednym przebiegu.
- Zbiór testowy używany dopiero w sekcji 8, do oceny, nie do ponownego treningu.
- Zadanie diagnostyczne (sekcja 9) wskazuje konkretną przyczynę (odwrócona
  konwencja maski + wiersz w pełni zamaskowany dla sekwencji bez paddingu), nie
  ogólnik.
- Wnioski odwołują się do konkretnych liczb z Twojego przebiegu i wprost
  odpowiadają, czego eksperyment **nie** dowodzi, nie tylko co dowodzi.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/10_attention/`) i nie jest częścią tego repozytorium.
