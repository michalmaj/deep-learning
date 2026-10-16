# Laboratorium 10: mały Transformer do klasyfikacji sekwencji zależnych od kolejności

Czas: ok. 2 godziny. Bazuje na
[`lectures/10_attention/lecture.md`](../../lectures/10_attention/lecture.md)
(scaled dot-product attention, multi-head attention, maskowanie, positional
encoding, blok Transformer), ale na **innym, trudniejszym zadaniu** niż wykład:
sekwencje o **zmiennej długości** (prawdziwy padding, czego wykładowy eksperyment D
nie wymagał) z **dystraktorami** (losowymi tokenami niosącymi zero informacji).

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
**musi pomijać padding** przy uśrednianiu (maska `x != PAD_ID`, dzielenie przez
`lengths`, nie przez `MAX_LEN` — dokładnie ta sama zasada co w module 9, sekcja 6).
Wytrenuj go od razu — to Twój punkt odniesienia w sekcji 13.

## 3. Positional encoding (gotowe)

Funkcja `sinusoidal_positional_encoding(seq_len, d_model)` z wykładu — nie
reimplementujemy jej (już w pełni zweryfikowana tam), tylko **decydujemy, jak i
gdzie** ją dołączyć w architekturze (sekcja 5).

## 4. Maska paddingu — Twoje zadanie

Zbuduj `key_padding_mask` z `x` i `PAD_ID`. **Krytyczne:** używasz
`nn.MultiheadAttention`, więc zgodnie z wykładem (sekcja "Ważna uwaga
implementacyjna") `key_padding_mask` musi mieć konwencję **`True` = to jest padding,
zignoruj** — **nie** `True` = dozwolone (ta konwencja jest **odwrotna** względem
`F.scaled_dot_product_attention`). Pomyl te konwencje i trafisz prosto w zadanie
diagnostyczne z sekcji 12 — zanim do niego dojdziesz, zweryfikuj własną maskę: jedna
sekwencja bez paddingu powinna mieć maskę całkowicie `False`.

## 5. Blok Transformer — Twoje zadanie

Zbuduj `TinyTransformerBlock`: self-attention (`nn.MultiheadAttention` — projekcje
Q/K/V nie trzeba pisać ręcznie, to już zweryfikowane na wykładzie) **+ połączenie
rezydualne + LayerNorm + feed-forward (Linear→ReLU→Linear) + drugie połączenie
rezydualne + drugi LayerNorm** (Post-LN, jak w `lecture.md`). To **Ty** łączysz te
elementy — nie chowaj architektury za jednym wywołaniem `nn.TransformerEncoderLayer`.
Pamiętaj przekazać `key_padding_mask` z sekcji 4 do `self.mha(...)`.

## 6. Agregacja do klasyfikacji — Twoje zadanie

Zbuduj `Classifier`: `nn.Embedding` (+ positional encoding z sekcji 3, jeśli
`use_pos_enc=True`) → `TinyTransformerBlock` z sekcji 5 → **agregacja do jednego
wektora na sekwencję** → warstwa klasyfikacyjna. Agregacja (uśrednienie po
pozycjach) musi **wykluczać padding** (ta sama zasada co w sekcji 2) — inaczej
reprezentacja krótszej sekwencji zostanie "rozwodniona" przez embeddingi paddingu,
nawet jeśli sama attention poprawnie je ignorowała.

**Weryfikacja (gotowa asercja czeka w notebooku):** ta sama sekwencja, przetworzona
**osobno** i **zbatchowana** z inną, dłuższą sekwencją (więc inaczej dopełniona
paddingiem), musi dać **identyczną** reprezentację w obu przypadkach. Jeśli nie
przechodzi — masz błąd w masce (sekcja 4) albo w agregacji (ta sekcja), zanim
dojdziesz do treningu.

## 7. Trening na CPU (gotowe)

`train_eval(model, epochs, lr, batch_size, seed)` — ta sama pętla dla wszystkich
konfiguracji w dalszej części laboratorium.

## 8. Porównanie: z vs bez positional encoding — Twoje zadanie

Wytrenuj **dwie** konfiguracje Twojego `Classifier` — `use_pos_enc=True` i
`use_pos_enc=False` — tym samym budżetem treningowym. Sprawdź na **kilku ziarnach**
(nie jednym), czy wynik bez PE jest systematycznie bliski poziomowi przypadku, czy to
przypadek jednego nieudanego przebiegu.

## 9. Zachowanie dla różnych długości sekwencji — Twoje zadanie

Dla modelu **z** positional encoding: policz accuracy na zbiorze walidacyjnym w co
najmniej 3 przedziałach długości (np. $[10,19]$, $[20,29]$, $[30,40]$). Czy
skuteczność jest stabilna, czy spada dla dłuższych/krótszych sekwencji?

## 10. Analiza wybranych wag attention — Twoje zadanie

Dla jednego poprawnie sklasyfikowanego przykładu: wypisz sekwencję, pozycje tokenów
`A` i `B`, oraz wagi attention (zapytanie = pozycja tokenu `A`, dowolna głowa).
**Nie interpretuj** wysokiej wagi w stronę `B` jako "dowód", że model "rozumie"
relację porządkową — opisz tylko to, co faktycznie widzisz w liczbach.

## 11. Ewaluacja na zbiorze testowym — Twoje zadanie

Oceń model z positional encoding (wybrany na podstawie walidacji) na zbiorze
testowym (**pierwszy raz** w tym laboratorium): accuracy + rozkład długości błędnie
sklasyfikowanych przykładów (czy błędy koncentrują się przy konkretnych długościach?).

## 12. Zadanie diagnostyczne

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

## 13. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami z Twojego przebiegu (sekcje 2, 8-12):

1. Czy potrafisz ręcznie wskazać, które elementy architektury (sekcja 5-6) odpowiadają
   za: (a) mieszanie informacji między pozycjami, (b) wprowadzenie informacji o
   pozycji, (c) redukcję sekwencji do jednego wektora?
2. Jak duża jest różnica accuracy między baseline'em (sekcja 2) a Twoim
   Transformerem **z** PE — i co to mówi o tym, czy zadanie faktycznie wymaga
   kolejności (w porównaniu z tym, co baseline bez PE osiąga w sekcji 8)?
3. Czy wynik "bez PE = poziom przypadku" jest stabilny na kilku ziarnach, czy to
   przypadek jednego przebiegu?
4. Czy maskowanie jest poprawne — czy masz dowód (asercja z sekcji 6), że model nie
   korzysta z informacji zawartej w paddingu?
5. Co dokładnie pokazuje (a czego **nie** pokazuje) ten eksperyment o mechanizmie
   attention? Czy wysoka accuracy sama w sobie dowodzi, że model "rozumie" relację
   porządkową `A`/`B`, czy tylko że skutecznie minimalizuje stratę na tym konkretnym
   zbiorze?
6. Jakie jest co najmniej jedno ograniczenie tego eksperymentu, które nie pozwala
   uogólnić wniosków na dowolne zadania sekwencyjne?

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, dane, weryfikacja balansu) | 10 min |
| Sekcja 2 (baseline bez kolejności) | 15 min |
| Sekcja 3 (positional encoding — gotowe, krótki przegląd) | 5 min |
| Sekcja 4 (maska paddingu) | 15 min |
| Sekcja 5 (blok Transformer) | 25 min |
| Sekcja 6 (agregacja + klasyfikator + weryfikacja) | 15 min |
| Sekcje 7-8 (trening, porównanie z/bez PE) | 15 min |
| Sekcja 9 (zachowanie dla różnych długości) | 10 min |
| Sekcja 10 (analiza wag attention) | 10 min |
| Sekcja 11 (ewaluacja testowa) | 10 min |
| Sekcja 12 (zadanie diagnostyczne) | 15 min |
| Sekcja 13 (wnioski) | 10 min |

Sekcje 0-12 to rdzeń obowiązkowy (suma orientacyjna przekracza 120 min — jak w
module 9, czas na poszczególne sekcje zależy od wcześniejszego opanowania
materiału; priorytet mają sekcje 4-6, rdzeń architektury). Trening trwa pojedyncze
sekundy na CPU na jeden przebieg — budżet czasowy idzie w poprawną implementację
maskowania/agregacji i analizę, nie w czekanie na trening.

**Rozszerzenie (opcjonalne, jeśli zostanie czas):** ułóż **dwa** `TinyTransformerBlock`
jeden po drugim (stos) — czy poprawia to accuracy lub stabilność na dłuższych
sekwencjach? Albo: zamień Post-LN na Pre-LN (norma przed podwarstwą, nie po) i
sprawdź, czy coś się zmienia w stabilności treningu.

## Kryteria oceny

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
- Porównanie z/bez PE (sekcja 8) sprawdzone na kilku ziarnach, nie jednym przebiegu.
- Zbiór testowy używany dopiero w sekcji 11.
- Zadanie diagnostyczne (sekcja 12) wskazuje konkretną przyczynę (odwrócona
  konwencja maski + wiersz w pełni zamaskowany dla sekwencji bez paddingu), nie
  ogólnik.
- Wnioski odwołują się do konkretnych liczb z Twojego przebiegu i wprost
  odpowiadają, czego eksperyment **nie** dowodzi, nie tylko co dowodzi.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/10_attention/`) i nie jest częścią tego repozytorium.
