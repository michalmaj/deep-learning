Język: [English](lecture.en.md) | **Polski**

# Wykład 10 — Attention i architektura Transformer

## Cele wykładu

1. Rozumieć attention jako ważone agregowanie informacji z różnych pozycji sekwencji
   — bezpośredni dostęp, bez przepychania informacji przez łańcuch stanów ukrytych.
2. Umieć policzyć scaled dot-product attention ręcznie i wyjaśnić role $Q$, $K$, $V$.
3. Rozumieć multi-head attention jako kilka niezależnych projekcji działających
   równolegle, nie jako jeden większy mechanizm.
4. Poprawnie stosować maskowanie (padding i przyczynowe) — w tym rozumieć, że różne
   API PyTorch interpretują maski boolowskie **różnie**, nie zamiennie.
5. Rozumieć, dlaczego self-attention bez informacji o pozycji nie rozróżnia kolejności
   tokenów, i jak positional encoding to naprawia.
6. Znać strukturę bloku Transformer (attention + FFN + residual + LayerNorm) i
   rozróżniać: pojedynczą głowę, warstwę multi-head, blok, cały model.

## Dlaczego to jest ważne

W module 9 stan ukryty $h_t$ musiał skompresować całą dotychczasową sekwencję do
wektora o stałym rozmiarze — informacja z odległej przeszłości docierała do bieżącego
kroku dopiero po przejściu przez wiele pośrednich aktualizacji (i, jak zmierzyliśmy,
mogła się po drodze zanikać). Attention odwraca tę zasadę: zamiast polegać wyłącznie
na skompresowanym stanie, każda pozycja może **bezpośrednio** "zajrzeć" do
reprezentacji dowolnej innej pozycji, ważąc ją tym, jak bardzo jest istotna. Zobaczymy
to dosłownie w eksperymencie porównawczym z GRU: na zadaniu, gdzie dwa istotne tokeny
są odległe o $64$ pozycje, GRU traci większość skuteczności (spada blisko poziomu
przypadku), a mały Transformer radzi sobie wyraźnie lepiej (choć też nie idealnie).

## Intuicja

Wyobraź sobie, że dla każdej pozycji w sekwencji zadajesz pytanie: "które inne pozycje
są dla mnie istotne, i ile informacji powinienem od nich wziąć?". Attention
formalizuje to pytanie: każda pozycja emituje **zapytanie** (query), każda pozycja
(łącznie z nią samą) udostępnia **klucz** (key, "czego dotyczę") i **wartość**
(value, "jaką informację niosę"). Podobieństwo zapytania do klucza decyduje o wadze, z
jaką odpowiadająca wartość wchodzi do wyniku.

❌ "Attention i Transformer to to samo."

✅ Attention to **mechanizm** (sposób ważonego agregowania informacji) — Transformer
to **architektura** zbudowana z tego mechanizmu plus feed-forward network, połączeń
rezydualnych i normalizacji, ułożonych w powtarzalne bloki. Sam mechanizm attention
istniał i był używany (np. w modelach encoder-decoder z RNN) zanim powstał Transformer
— zobaczymy niżej, że Transformer to konkretny sposób złożenia attention z innymi,
równie istotnymi elementami.

## Formalny model: Query, Key, Value

Dla sekwencji reprezentacji wejściowych $x_1,\ldots,x_T \in \mathbb R^{d_{\text{model}}}$
(np. embeddingi tokenów z modułu 8, ewentualnie plus positional encoding — niżej),
budujemy trzy macierze przez projekcje liniowe:

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

gdzie $X\in\mathbb R^{T\times d_{\text{model}}}$ (wiersze = pozycje), $W^Q, W^K \in
\mathbb R^{d_{\text{model}}\times d_k}$, $W^V\in\mathbb R^{d_{\text{model}}\times d_v}$
(w podstawowej wersji $d_k=d_v=d_{\text{model}}$, w multi-head — niżej — mniejsze).

**Uwaga o notacji.** $Q$, $K$, $V$ to **ustandaryzowana w całej literaturze** notacja
(query/key/value) — zachowujemy ją, mimo że $K$ w tym kursie oznacza też "liczbę
warstw" (moduł 2) i "rozmiar kernela" (moduł 6). To **inny byt**: $K$ tutaj to zawsze
**macierz** (pojawia się razem z $Q$ i $V$), nigdy pojedyncza liczba — kontekst
jednoznacznie rozróżnia, podobnie jak w module 8 rozróżniliśmy dwa znaczenia $k$
(k-NN vs. rozmiar kernela). Wagi attention oznaczamy $\alpha$ (nieużywane wcześniej w
kursie) — **nie** $A$, które w module 3 oznacza batchowane aktywacje warstwy
($A^{(l)}$).

## Matematyka: scaled dot-product attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Rozbijmy to krok po kroku:

1. **$QK^\top$** $\in\mathbb R^{T\times T}$ — iloczyn skalarny każdego zapytania z
   każdym kluczem: wiersz $i$, kolumna $j$ to $q_i\cdot k_j$, miara "jak bardzo
   zapytanie $i$ pasuje do klucza $j$" (większy iloczyn skalarny = bardziej zgodne
   kierunki, przy porównywalnych normach — dokładnie ta intuicja z modułu 8, cosine
   similarity, tylko tu bez jawnej normalizacji L2).
2. **Skalowanie przez $\sqrt{d_k}$**: dla losowych $q,k$ o współrzędnych
   niezależnych, wariancja $q\cdot k$ rośnie **liniowo** z $d_k$ (suma $d_k$
   niezależnych iloczynów) — bez skalowania, przy dużym $d_k$, wartości $QK^\top$
   mają dużą wariancję, co pcha softmax w rejon bardzo małych gradientów (nasycenie,
   moduł 2). Dzielenie przez $\sqrt{d_k}$ sprowadza wariancję z powrotem do rzędu $1$,
   niezależnie od $d_k$.
3. **Softmax wzdłuż ostatniego wymiaru** (po kolumnach, dla każdego wiersza/zapytania
   osobno) — zamienia każdy wiersz w rozkład prawdopodobieństwa nad dostępnymi
   kluczami: $\alpha_{ij}\geq 0$, $\sum_j \alpha_{ij}=1$. To właśnie **wagi
   attention**: ile uwagi zapytanie $i$ poświęca kluczowi (a więc i wartości) $j$.
4. **$\cdot V$**: każda pozycja wyjściowa to ważona (wagami $\alpha_{i\cdot}$) suma
   wszystkich wartości — agregacja informacji z całej sekwencji, w jednym kroku, bez
   przechodzenia przez pośrednie stany.

**Wymiary — jawnie.** Dla pojedynczej sekwencji: $Q,K\in\mathbb R^{T\times d_k}$,
$V\in\mathbb R^{T\times d_v}$, $QK^\top\in\mathbb R^{T\times T}$, wyjście
$\in\mathbb R^{T\times d_v}$. Dla batcha rozmiaru $B$: dokładamy wymiar z przodu,
$Q,K\in\mathbb R^{B\times T\times d_k}$ itd., $QK^\top\in\mathbb R^{B\times T\times T}$
— macierz wag attention jest **osobna dla każdego przykładu w batchu**.

**Uwaga o notacji.** Rozmiar batcha oznaczamy tu $B$, nie $N$ jak w modułach
1-3/6-7 — świadomie, bo w tym module $N$ jest już zajęte (niżej: liczba bloków
Transformer ułożonych jeden po drugim). $B$ to jedyny tu symbol dla "liczby
przykładów naraz", bez kolizji w obrębie tego dokumentu.

## Przykład rozwiązany: 3 pozycje, $d_k=4$

$$Q=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&0&0\end{pmatrix}\quad K=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&1&1\end{pmatrix}\quad V=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$$

**1. $QK^\top$:**
$$QK^\top=\begin{pmatrix}2&0&2\\0&2&2\\1&1&2\end{pmatrix}$$

**2. Po skalowaniu ($\div\sqrt4=\div2$):**
$$\begin{pmatrix}1{,}0&0{,}0&1{,}0\\0{,}0&1{,}0&1{,}0\\0{,}5&0{,}5&1{,}0\end{pmatrix}$$

**3. Wagi po softmax (każdy wiersz sumuje się do $1$):**
$$\begin{pmatrix}0{,}4223&0{,}1554&0{,}4223\\0{,}1554&0{,}4223&0{,}4223\\0{,}2741&0{,}2741&0{,}4519\end{pmatrix}$$

**4-5. Ważona kombinacja $V$ = końcowa reprezentacja:**
$$\begin{pmatrix}3{,}000&4{,}000\\3{,}534&4{,}534\\3{,}356&4{,}356\end{pmatrix}$$

Zweryfikowane w granicach tolerancji numerycznej (`atol=1e-6`) względem
`F.scaled_dot_product_attention` w notebooku.

## Multi-head attention

Zamiast jednej projekcji $Q,K,V$ o pełnym wymiarze $d_{\text{model}}$, multi-head
attention dzieli na $H$ **niezależnych** głów, każda o mniejszym wymiarze
$d_k = d_{\text{model}}/H$. **Uwaga o notacji:** $H$ tutaj to liczba głów attention —
**inne znaczenie** niż $H$ w module 6 (wysokość obrazu w konwencji NCHW). Ten sam
symbol, zupełnie inna domena (sekwencje vs obrazy) — rozróżnia go kontekst, tak jak
$k$ w module 6 (rozmiar kernela) i $k$ w module 8 (k-NN):

$$\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i), \qquad i=1,\ldots,H$$
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)\,W^O$$

Każda głowa ma **własne** $W^Q_i, W^K_i, W^V_i$ (różne macierze, nie ta sama
projekcja powielona $H$ razy) — wyniki konkatenowane wzdłuż wymiaru cech i
przepuszczone przez jedną wspólną projekcję wyjściową $W^O\in\mathbb
R^{d_{\text{model}}\times d_{\text{model}}}$. W praktyce (i w `nn.MultiheadAttention`)
projekcje $Q,K,V$ wszystkich głów liczy się **jedną** macierzą
$W^{QKV}\in\mathbb R^{d_{\text{model}}\times 3d_{\text{model}}}$ (złączoną), a dopiero
potem dzieli wynik na $H$ głów — matematycznie równoważne $H$ osobnym projekcjom,
zweryfikowane w granicach tolerancji numerycznej (`atol=1e-5`) w notebooku przez
ręczne rozcięcie `in_proj_weight`.

❌ "Każda głowa uczy się innej, rozpoznawalnej relacji (np. 'głowa składniowa',
'głowa semantyczna')."

✅ Różne głowy **mogą** wyspecjalizować się w różnych wzorcach — to realna, udokumentowana
obserwacja w dużych, wytrenowanych modelach — ale nie jest to **gwarancja**
architektury, tylko możliwy (i niejednoznaczny do zinterpretowania) efekt treningu.
Losowo zainicjalizowane głowy nie mają żadnej z góry ustalonej specjalizacji — w
eksperymencie B zobaczymy, że same kształty i niezależne macierze wag niczego o
treści nie przesądzają.

## Maskowanie

**Padding mask**: zapytania nie powinny "zaglądać" do pozycji wypełnienia (moduł 9:
padding to nie treść). **Causal mask** (maska przyczynowa): w modelach
autoregresywnych (generowanie tekstu token po tokenie) pozycja $i$ nie może korzystać
z pozycji $j>i$ — inaczej model "widziałby przyszłość" podczas treningu, czego nie ma
podczas faktycznego generowania.

❌ "Causal mask załatwia też padding, albo odwrotnie — to w gruncie rzeczy to samo."

✅ To **dwa niezależne** ograniczenia, które się **łączą** (logiczne AND), nie
zastępują: causal mask ogranicza na podstawie **pozycji względem zapytania**
(zawsze ten sam trójkątny wzorzec, niezależny od konkretnej sekwencji), padding mask
ogranicza na podstawie **treści danego przykładu** (które pozycje są realne, zależy
od długości konkretnej sekwencji w batchu). Model dekodera ze zmienną długością
sekwencji w batchu potrzebuje **obu naraz** — zweryfikowane w notebooku: połączona
maska to iloczyn (logiczny AND) obu masek z osobna.

### Padding w RNN a padding w attention — to nie ten sam mechanizm

Moduł 9 uczy `pack_padded_sequence` jako sposobu na ignorowanie paddingu w
RNN/LSTM/GRU — łatwo stąd wynieść, że maska paddingu w attention "robi to samo".
**Nie robi.** To dwa różne mechanizmy, nie dwie nazwy jednej rzeczy:

| | `pack_padded_sequence` (RNN, moduł 9) | `key_padding_mask` (attention) |
|---|---|---|
| Co dzieje się z pozycją paddingu | **Nie jest przetwarzana w ogóle** — warstwa rekurencyjna fizycznie pomija ten krok | **Jest przetwarzana normalnie** przez embedding, FFN, LayerNorm — zerowana jest tylko jej **waga w sumie attention** |
| Pozycja paddingu jako zapytanie | Nie istnieje (sekwencja się już skończyła) | **Wciąż produkuje jakiś wynik** — nie jest pomijana jako krok obliczeniowy, tylko później ignorowana |

Konsekwencja: samo zamaskowanie attention **nie wystarcza**, żeby padding był
całkowicie bez wpływu na wynik — jeśli reprezentacja całej sekwencji powstaje przez
uśrednienie *wszystkich* pozycji (nie tylko rzeczywistych), padding wpłynie na
wynik, nawet gdy attention poprawnie go zignorowało przy ważeniu. Potrzebna jest
**jeszcze** poprawna agregacja na końcu — zobacz zadanie diagnostyczne w
laboratorium tego modułu.

### Maska boolowska a addytywna

Maskę można zakodować na dwa sposoby: **boolowską** (gdzie jedna wartość oznacza
"dozwolone", druga "zablokowane" — ale **która jest która, zależy od API!**, zobacz
niżej) albo **addytywną** (tensor liczb rzeczywistych dodawany do wyniku $QK^\top$
przed softmax: $0$ dla dozwolonych pozycji, $-\infty$ dla zablokowanych — po dodaniu
$-\infty$, `softmax` nadaje tej pozycji wagę $0$).

### Wiersze z wszystkimi kluczami zamaskowanymi

Jeśli dla jakiegoś zapytania **żaden** klucz nie jest dozwolony (np. błąd w
konstrukcji maski), wiersz $QK^\top$ po maskowaniu to same $-\infty$ — `softmax`
liczy $e^{-\infty}/\sum e^{-\infty} = 0/0$. **Zweryfikowane w notebooku:** naiwna,
ręczna implementacja (`masked_fill(-\infty)` + `softmax`) daje w tej sytuacji `NaN`.
Zoptymalizowane jądro `F.scaled_dot_product_attention` ma specjalny przypadek i
zwraca $0$ zamiast `NaN` — **ale nie licz na to przy własnej implementacji "from
scratch"**: sprawdź, że każda pozycja zapytania, której wynik faktycznie
wykorzystujesz, ma przynajmniej jeden dozwolony klucz (np. zapytania odpowiadające
realnym, nie-paddingowym tokenom zawsze mają dostępny przynajmniej siebie samych przy
causal mask).

**Dodatkowa pułapka, specyficzna dla `nn.MultiheadAttention`** (zweryfikowana
empirycznie): to, czy wiersz w pełni zamaskowany daje `NaN` czy bezpieczne zero,
zależy od parametru `need_weights`. **Domyślną wartością jest `need_weights=True`**
(potrzebne, gdy chcemy **obejrzeć** same wagi attention — jak w notebooku i
laboratorium) — wymusza to jawną ścieżkę z ręcznym `softmax`, i wtedy wiersz w
pełni zamaskowany daje `NaN`, identycznie jak w naiwnej implementacji "from
scratch" wyżej. Trzeba **jawnie ustawić** `need_weights=False`, żeby
`nn.MultiheadAttention` mogło użyć tej samej, zoptymalizowanej ścieżki co
`F.scaled_dot_product_attention` — bezpieczne zero. Wniosek: nawet w ramach
**jednego** API zachowanie numeryczne może zależeć od argumentu niezwiązanego z
samym maskowaniem — kolejny powód, by nie zakładać zachowania bez testu.

### Ważna uwaga implementacyjna: maski NIE są przenośne między API

**Zweryfikowane eksperymentalnie, nie z dokumentacji** (bezpośrednia inspekcja wag
attention przy kontrolowanym przykładzie):

| API | parametr | `True` oznacza |
|---|---|---|
| `F.scaled_dot_product_attention` | `attn_mask` (bool) | **dozwolone** (attend) |
| `nn.MultiheadAttention` | `attn_mask` (bool) | **zablokowane** (ignoruj) |
| `nn.MultiheadAttention` | `key_padding_mask` (bool) | **to jest padding** (ignoruj) |

`nn.MultiheadAttention` jest **wewnętrznie spójne** (`attn_mask` i
`key_padding_mask` obie: `True`=zablokuj), ale **odwrotne** względem
`F.scaled_dot_product_attention`. Skopiowanie maski boolowskiej z jednego API do
drugiego bez odwrócenia da **po cichu błędny wynik** — żadnego wyjątku, model się
wytrenuje, ale będzie "widział" dokładnie te pozycje, które miał ignorować (i
odwrotnie). Zweryfikowany test w notebooku sprawdza to wprost dla obu API, nie
ufając samej dokumentacji.

## Pozycja w sekwencji: dlaczego self-attention jej potrzebuje

Wzór $\text{Attention}(Q,K,V)$ **nie odwołuje się nigdzie do indeksu pozycji** $i$ czy
$j$ poza tym, które konkretne wiersze $X$ bierze — jeśli permutujemy wiersze wejścia
$X$, wyjście permutuje się **dokładnie tak samo** (własność nazywana
**równoważnością permutacyjną**, *permutation equivariance*). Zweryfikowane w
notebooku: `output(permutuj(X))` zgadza się z `permutuj(output(X))` w granicach
tolerancji numerycznej (`atol=1e-6`).

Konsekwencja: jeśli zbudujemy model klasyfikujący sekwencję przez self-attention +
agregację **niewrażliwą na kolejność** (np. średnią po pozycjach), całość jest
**niezmiennicza permutacyjnie** (*permutation invariant*) — permutacja wejścia **nie
zmienia** predykcji. Zweryfikowane: dla zadania z modułu 9 (etykieta zależy wyłącznie
od kolejności dwóch tokenów), model self-attention + mean-pooling **bez** positional
encoding osiąga dokładnie poziom przypadku — identyczny wynik na każdym z 3 ziaren
(bo nie jest to kwestia "pecha" przy inicjalizacji — model matematycznie **nie może**
reprezentować funkcji zależnej od kolejności).

❌ "Self-attention automatycznie 'wie', która pozycja jest która, bo przetwarza
sekwencję."

✅ Sam mechanizm attention jest ślepy na pozycję — "wie" tylko tyle, ile **wprost**
dostanie w reprezentacji wejściowej. Stąd **positional encoding**: dodajemy do
embeddingu tokenu dodatkowy wektor zależny od jego pozycji, **zanim** wejdzie do
attention.

**Positional encoding — wersja sinusoidalna** (oryginalny Transformer):

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Każda para współrzędnych koduje pozycję falą sinusoidalną o innej częstotliwości —
deterministyczne, bez dodatkowych parametrów do wytrenowania, działa (w ograniczonym
stopniu) dla sekwencji dłuższych niż widziane podczas treningu. Alternatywa: **uczony
embedding pozycji** (`nn.Embedding(max_len, d_model)`, dokładnie jak embedding
tokenów z modułu 8, tylko indeksowany pozycją zamiast tożsamością tokenu) — prostszy,
ale nie ekstrapoluje poza `max_len` widziane podczas treningu.

❌ "Dowolna z tych dwóch metod kodowania pozycji działa tak samo dobrze w każdej
sytuacji."

✅ Sinusoidalne kodowanie jest deterministyczne i (częściowo) generalizuje na dłuższe
sekwencje; uczone embeddingi pozycji mogą dopasować się lepiej do konkretnego zadania,
ale tylko w zakresie długości widzianych podczas treningu — wybór to kompromis, nie
oczywistość.

## Architektura bloku Transformer

$$x' = \text{LayerNorm}\big(x + \text{MultiHead}(x)\big), \qquad \text{out} = \text{LayerNorm}\big(x' + \text{FFN}(x')\big)$$

(wariant **Post-LN**, jak w oryginalnym Transformerze — normalizacja **po** dodaniu
połączenia rezydualnego). FFN to prosta sieć **stosowana niezależnie do każdej
pozycji** (*position-wise*): $\text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$,
zwykle z wewnętrznym wymiarem $4\times$ większym niż $d_{\text{model}}$.

- **Połączenia rezydualne** ($x + \text{podwarstwa}(x)$): dokładnie ten sam mechanizm
  co w ResNet (moduł 7) — ułatwiają propagację gradientu przez głębokie stosy bloków,
  z tych samych powodów (gradient ma "skrót" omijający podwarstwę).
- **LayerNorm**: normalizuje każdą pozycję **niezależnie**, po wymiarze cech (nie po
  batchu, jak BatchNorm z modułu 6-7) — zweryfikowane w notebooku: średnia $\approx0$,
  odchylenie $\approx1$ dla każdej pozycji osobno.
- **Dropout**: stosowany typowo po attention i po FFN, przed dodaniem do połączenia
  rezydualnego — jak w module 5, ale uwaga na pułapkę niżej.

**Pre-LN** (wariant używany w wielu nowszych modelach): normalizacja **przed**
podwarstwą, nie po: $x' = x + \text{MultiHead}(\text{LayerNorm}(x))$. Różnica wpływa
na stabilność treningu bardzo głębokich stosów — nie rozwijamy tu pełnej analizy,
tylko odnotowujemy, że to realny, świadomy wybór projektowy, nie szczegół bez
znaczenia.

**Cztery różne poziomy — nie mylić:**

| Poziom | Co to jest |
|---|---|
| Pojedyncza głowa | jedno $\text{Attention}(XW^Q_i,XW^K_i,XW^V_i)$ |
| Warstwa multi-head | $H$ głów + konkatenacja + projekcja $W^O$ |
| Blok Transformer | multi-head + residual + LN + FFN + residual + LN |
| Model | $N$ bloków ułożonych jeden po drugim (plus embedding + głowica na wejściu/wyjściu) |

❌ "Zbudowanie Transformera to w zasadzie jedno wywołanie `nn.Transformer`."

✅ `nn.Transformer`/`nn.TransformerEncoder` **ukrywają** wszystkie powyższe warstwy
za jednym wywołaniem — wygodne w praktyce, ale nieprzydatne dydaktycznie: w tym
module budujemy blok jawnie, warstwa po warstwie, żeby widzieć, co dzieje się w
środku.

## Encoder-only, decoder-only, encoder-decoder

Oryginalny Transformer (Vaswani i in., 2017) to architektura **encoder-decoder**:
encoder przetwarza całą sekwencję wejściową naraz (self-attention bez maski
przyczynowej — każda pozycja widzi całą resztę), decoder generuje sekwencję wyjściową
token po tokenie (self-attention **z** maską przyczynową, plus **cross-attention** do
wyjścia encodera — $Q$ z decodera, $K,V$ z encodera).

- **Cross-attention a self-attention**: self-attention liczy $Q,K,V$ z **tej samej**
  sekwencji; cross-attention liczy $Q$ z jednej sekwencji (np. decodera), a $K,V$ z
  **innej** (np. encodera) — mechanizm matematyczny identyczny, różni się tylko,
  **skąd** pochodzą wejścia.
- **Encoder-only** (np. architektury typu BERT): tylko encoder, self-attention bez
  maski przyczynowej — dobre do zadań, gdzie cała sekwencja jest dostępna naraz
  (klasyfikacja, jak w naszym labie).
- **Decoder-only** (np. architektury typu GPT): tylko decoder, self-attention z
  maską przyczynową, bez cross-attention (nic do czego by się odwoływać) — dobre do
  generowania tekstu token po tokenie.

❌ "Encoder-only i decoder-only to w gruncie rzeczy ta sama architektura pod inną
nazwą."

✅ Różnią się **czy i jak** maskują self-attention — encoder-only nie maskuje
(dwukierunkowy dostęp do całej sekwencji), decoder-only maskuje przyczynowo
(jednokierunkowy, zgodny z generowaniem). To realna różnica funkcjonalna, nie tylko
nazewnicza — model encoder-only nie nadaje się wprost do generowania tekstu token po
tokenie, bo podczas treningu "widział" całą sekwencję naraz.

## Co obserwujemy

Wszystkie liczby niżej pochodzą z rzeczywistego wykonania `examples.ipynb` (nie z
szacunków) — zobacz notebook po pełne wyjście.

- **Eksperyment A.** Ręczna implementacja scaled dot-product attention zgadza się,
  w granicach tolerancji numerycznej (`atol=1e-6`), z
  `F.scaled_dot_product_attention` na przykładzie liczbowym z 3 pozycjami. Zmiana
  jednego zapytania zmienia **tylko** odpowiadający mu wiersz
  wyniku; zmiana jednego klucza/wartości zmienia **wszystkie** wiersze (każde
  zapytanie może do niego "zajrzeć"). Zweryfikowano wprost: `nn.MultiheadAttention`
  interpretuje maskę boolowską (`attn_mask`, `key_padding_mask`) **odwrotnie** niż
  `F.scaled_dot_product_attention` — ta sama logiczna intencja zapisana z błędną
  konwencją daje odwrotny, cichy wynik. Wiersz z wszystkimi kluczami zamaskowanymi:
  naiwna implementacja daje `NaN`, `F.scaled_dot_product_attention` — zero.
  `model.eval()` nie wyłącza dropoutu przekazywanego jako stały argument funkcji
  (potwierdzone: dwa wywołania `NaiveAttention.eval()` dają różne wyniki).
- **Eksperyment B.** Ręczna konstrukcja multi-head attention (rozcięcie
  `in_proj_weight`, podział na głowy, konkatenacja, `out_proj`) zgadza się, w
  granicach tolerancji numerycznej (`atol=1e-5`), z `nn.MultiheadAttention`, zarówno
  output jak i wagi per głowa. Liczba parametrów dla $d_{\text{model}}=8$, $H=2$:
  $288$, zgodna ze wzorem
  $3d_{\text{model}}^2+3d_{\text{model}}+d_{\text{model}}^2+d_{\text{model}}$.
- **Eksperyment C.** Równoważność permutacyjna self-attention i niezmienniczość
  reprezentacji uśrednionej (oraz niezmienniczość reprezentacji stylu CLS na
  permutację pozostałych tokenów) — potwierdzone w granicach tolerancji numerycznej
  (`atol=1e-6`). Na zadaniu zależnym od kolejności (8-elementowy słownik, etykieta =
  kolejność tokenów 0 i 1): **z**
  positional encoding, `val_acc=test_acc=1,000`; **bez** PE, `val_acc=0,475`,
  `test_acc=0,485` — **identyczne na każdym z 3 testowanych ziaren** (0, 1, 2), co
  potwierdza, że to ograniczenie reprezentacyjne, nie nieszczęśliwa inicjalizacja.
  Permutacja wejścia **z** PE zmienia predykcję (logity $[5{,}84,-4{,}49]\to[6{,}01,-4{,}66]$
  po permutacji w sprawdzonym przykładzie) — zgodnie z oczekiwaniem, bo kolejność
  faktycznie się zmienia.
- **Eksperyment D.** Mały Transformer (2514 parametrów przy $T=8$) rozwiązuje zadanie
  porządkowe bez błędu (`val_acc=1,000`, 0/200 błędów). Porównanie z GRU (1922
  parametry) na trudniejszym wariancie zadania (dystraktory dołączone za stałą,
  wczesną parą decydujących tokenów — wymusza pamięć na całą długość "ogona"):
  $T=8,16,32$ — oba modele `val_acc=1,000`; **$T=64$ — Transformer $0{,}860$, GRU
  $0{,}480$** (blisko poziomu przypadku) — **zaobserwowana różnica accuracy**,
  zmierzona z jednego przebiegu (w przeciwieństwie do porównania z/bez PE wyżej w
  tym samym eksperymencie, to konkretne porównanie nie jest sprawdzone na wielu
  ziarnach). Moduł 9 (Eksperyment B) **zmierzył niezależnie**, na innym zadaniu, że
  norma gradientu GRU względem odległego wejścia zanika wykładniczo z długością
  sekwencji. Spadek accuracy GRU przy $T=64$ jest **zgodny z hipotezą**, że ten sam
  mechanizm przyczynił się do wyniku — GRU musi przenieść informację o pozycji
  tokenów 0/1 przez $56$ kroków samego szumu, Transformer ma do nich bezpośredni
  dostęp niezależnie od odległości — ale w **tym** eksperymencie nie zmierzyliśmy
  normy gradientu GRU na tym konkretnym zadaniu: to wiarygodna, ale niepotwierdzona
  bezpośrednio hipoteza, nie udowodniona przyczyna. Transformer **też** się pogarsza
  przy $T=64$ (nie jest to „zawsze wygrywa”) — przy tym samym, niezmienionym budżecie
  treningowym. Czas treningu: Transformer szybszy niż GRU przy większym $T$ (np.
  $1{,}22$s vs $2{,}41$s przy $T=64$), ale to mikrobenchmark na bardzo krótkich
  sekwencjach — nie wyciągamy z tego daleko idących wniosków o rzeczywistej
  wydajności.

## Częste błędne przekonania

(Dodatkowe, nieujęte wyżej)

❌ "Większa waga attention to dowód, że dany token przyczynowo wpłynął na predykcję."

✅ Waga attention mówi, ile dana wartość **wniosła** do ważonej sumy na tym etapie —
to **opisowa** własność jednego kroku obliczeń, nie dowód przyczynowości względem
finalnej predykcji modelu (która przechodzi jeszcze przez FFN, kolejne bloki,
głowicę). Wysoka waga nie gwarantuje, że usunięcie tego tokenu zmieniłoby wynik —
odwrotnie, niska waga nie gwarantuje braku wpływu (informacja mogła dotrzeć inną
drogą, np. przez wcześniejszy blok).

❌ "Attention eliminuje wszystkie problemy związane z długimi sekwencjami."

✅ Attention daje **bezpośredni** dostęp do każdej pozycji (zamiast przez łańcuch
stanów), co realnie pomaga przy długozasięgowych zależnościach (zobacz porównanie z
GRU niżej) — ale koszt pamięci i obliczeń rośnie **kwadratowo** z długością sekwencji
($T\times T$ macierz wag), więc bardzo długie sekwencje są kosztowne z zupełnie
innego powodu niż zanikający gradient.

❌ "Mały Transformer zawsze rozwiąże zadanie sekwencyjne lepiej niż RNN/LSTM/GRU."

✅ Zmierzone wprost (eksperyment porównawczy, sekcja "Co obserwujemy" wyżej): na
krótkich sekwencjach ($T=8$) GRU radzi sobie równie dobrze; przy $T=64$, przy tym
samym, niezmienionym budżecie
treningowym, **nasz mały Transformer też nie jest idealny** (nie 100%) — wielkość i
budżet treningu mają znaczenie, architektura sama w sobie nie gwarantuje sukcesu.

❌ "Sama obecność warstwy attention oznacza, że model poprawnie rozumie treść
sekwencji."

✅ Attention to mechanizm obliczeniowy — czy wytrenowany model faktycznie "rozumie"
treść w jakimkolwiek sensowym znaczeniu, to osobne pytanie, nierozstrzygnięte przez
samą architekturę. Rozróżniamy: matematyczne działanie mechanizmu (co dokładnie
liczy), możliwości architektury (co **mogłaby** reprezentować) i zachowanie
konkretnego wytrenowanego modelu (co **faktycznie** się nauczył) — trzy różne
poziomy, żaden nie implikuje automatycznie pozostałych.

## Implikacje inżynierskie

- **Koszt $T\times T$.** Macierz wag attention ma rozmiar $T\times T$ (na głowę, na
  przykład w batchu) — liczba porównań pozycji rośnie **kwadratowo** z długością
  sekwencji, w przeciwieństwie do RNN, gdzie koszt rośnie **liniowo**. Dla krótkich
  sekwencji (jak w naszych eksperymentach) to nieistotne; dla bardzo długich
  (tysiące tokenów) staje się dominującym kosztem pamięci i czasu.
- **Pamięć.** Prosta implementacja (jak nasza, w notebooku) **materializuje** całą
  macierz $T\times T\times H$ (dla $H$ głów) w pamięci — zoptymalizowane jądra (np.
  FlashAttention, tylko wspominamy, nie implementujemy) unikają jawnego
  materializowania pełnej macierzy, co daje **inny** profil kosztu pamięci niż nasza
  dydaktyczna wersja, mimo identycznego wyniku matematycznego.
- **Równoległość a rzeczywisty czas.** Self-attention może przetwarzać **wszystkie**
  pozycje sekwencji równolegle podczas treningu (nie ma zależności $h_t$ od
  $h_{t-1}$, jak w RNN) — ale to **nie oznacza automatycznie** niższego czasu
  wykonania w każdym zastosowaniu: przy krótkich sekwencjach, małych modelach i na
  CPU (gdzie "równoległość" i tak ogranicza liczba rdzeni) różnica bywa niewielka lub
  odwrotna, zmierzone wprost w eksperymencie porównawczym.
- **Liczba głów i wymiar reprezentacji.** Więcej głów przy tym samym
  $d_{\text{model}}$ oznacza **mniejszy** $d_k$ na głowę (bo $d_k=d_{\text{model}}/H$)
  — nie więcej parametrów, inny podział tej samej puli.
- **Trening vs inferencja.** Podczas treningu encoder/self-attention bez maski
  przyczynowej przetwarza całą sekwencję naraz. Model decoder-only podczas
  **generowania** (inferencji) produkuje jeden token na raz — w praktyce cache'uje
  już policzone $K,V$ dla wcześniejszych pozycji (*KV-cache*), żeby nie przeliczać
  ich od nowa przy każdym nowym tokenie — temat rozwijany w module 11.

## Co warto zapamiętać

- $\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$ — ważona agregacja
  wartości, wagi z podobieństwa zapytań do kluczy, skalowanie zapobiega nasyceniu
  softmaxu.
- Multi-head = $H$ niezależnych projekcji równolegle, nie gwarancja interpretowalnej
  specjalizacji każdej głowy.
- Maski boolowskie **nie są przenośne** między `F.scaled_dot_product_attention` i
  `nn.MultiheadAttention` — sprawdzaj semantykę za każdym razem.
- Self-attention bez positional encoding jest równoważne permutacyjnie — nie
  rozróżnia kolejności tokenów bez jawnie dodanej informacji o pozycji.
- Blok Transformer = multi-head attention + FFN + dwa połączenia rezydualne + dwie
  normalizacje — cztery odrębne poziomy (głowa/warstwa/blok/model), nie jeden byt.
- Transformer nie musi wygrywać z RNN/LSTM/GRU na każdym zadaniu, zwłaszcza małym i
  krótkim — przewaga ujawnia się przy długozasięgowych zależnościach, zmierzone.

## Pytania dla studentów

1. Masz zapytanie $q$ identyczne z jednym z kluczy $k_j$ i bardzo różne od
   pozostałych. Jak (w przybliżeniu) będzie wyglądał wiersz wag attention dla tego
   zapytania po softmax?
2. Dlaczego dzielimy przez $\sqrt{d_k}$, a nie np. przez $d_k$ albo wcale? Co by się
   stało z wagami attention przy bardzo dużym $d_k$ bez tego skalowania?
3. Masz model encoder-only (bez maski przyczynowej) wytrenowany do klasyfikacji
   sekwencji przez uśrednienie wyjść attention po pozycjach. Czy permutacja kolejności
   tokenów wejściowych zmieni predykcję? Uzasadnij, odwołując się do równoważności
   permutacyjnej.
4. Masz batch z sekwencjami o różnej długości i chcesz zbudować model
   autoregresywny (decoder-only). Jakich **dwóch** masek potrzebujesz jednocześnie i
   jak się łączą?

## Bibliografia

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in
  Neural Information Processing Systems 30 (NeurIPS 2017). arXiv:1706.03762.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*.
  arXiv:1607.06450.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image
  Recognition*. CVPR. (połączenia rezydualne — zobacz też moduł 7)
- PyTorch documentation: *torch.nn.functional.scaled_dot_product_attention*,
  `https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html`.
- PyTorch documentation: *torch.nn.MultiheadAttention*,
  `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
