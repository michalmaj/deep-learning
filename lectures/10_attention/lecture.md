Język: [English](lecture.en.md) | **Polski**

# Wykład 10. Attention i architektura Transformer

## Cele wykładu

1. opisać attention jako ważoną agregację informacji z pozycji sekwencji,
2. obliczyć scaled dot-product attention i wyjaśnić role $Q$, $K$ oraz $V$,
3. odróżnić pojedynczą głowę, multi-head attention, blok Transformer i cały model,
4. poprawnie stosować maskę paddingu i maskę przyczynową,
5. wyjaśnić potrzebę kodowania pozycji,
6. przedstawić attention, FFN, połączenia rezydualne i LayerNorm jako elementy bloku Transformer.

## Dlaczego to jest ważne

W module 9 stan $h_t$ kompresował dotychczasową sekwencję do wektora o stałym rozmiarze. Informacja z odległej pozycji musiała przejść przez wiele aktualizacji i mogła po drodze zanikać.

Attention daje każdej pozycji bezpośredni dostęp do reprezentacji innych pozycji. Ich udział zależy od wyuczonych wag. Eksperyment porówna ten mechanizm z GRU w zadaniu, w którym istotne tokeny dzieli długi ciąg dystraktorów.

## Intuicja

Dla każdej pozycji powstaje pytanie o to, które elementy sekwencji są istotne. Zapytanie, czyli *query*, opisuje poszukiwaną informację. Klucz, czyli *key*, pozwala ocenić zgodność pozycji z zapytaniem. Wartość, czyli *value*, zawiera informację przekazywaną do wyniku.

Podobieństwo zapytania do klucza wyznacza wagę odpowiadającej wartości.

❌ Attention i Transformer oznaczają to samo.

✅ Attention jest mechanizmem agregacji. Transformer jest architekturą łączącą attention z FFN, połączeniami rezydualnymi i normalizacją.

## Query, Key i Value

Dla $X\in\mathbb R^{T\times d_{\text{model}}}$, którego wiersze odpowiadają pozycjom, tworzone są trzy projekcje liniowe.

$$Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V$$

Macierze $W^Q,W^K\in\mathbb R^{d_{\text{model}}\times d_k}$, a $W^V\in\mathbb R^{d_{\text{model}}\times d_v}$. Wejścia mogą być embeddingami tokenów z modułu 8 uzupełnionymi kodowaniem pozycji.

$Q$, $K$ i $V$ są standardowymi symbolami query, key i value. Litera $K$ oznacza tutaj macierz, nie liczbę warstw ani rozmiar kernela. Wagi attention są oznaczane przez $\alpha$, aby nie kolidowały z aktywacjami $A^{(l)}$ z modułu 3.

## Scaled dot-product attention

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Obliczenie składa się z czterech etapów.

1. $QK^\top$ zawiera iloczyny skalarne wszystkich zapytań i kluczy. Element $(i,j)$ mierzy zgodność $q_i$ z $k_j$.
2. Dzielenie przez $\sqrt{d_k}$ stabilizuje wariancję wyników. Bez skalowania większe $d_k$ sprzyja nasyceniu softmaxu i małym gradientom.
3. Softmax działa osobno w każdym wierszu. Otrzymane $\alpha_{ij}$ są nieujemne i spełniają $\sum_j\alpha_{ij}=1$.
4. Mnożenie przez $V$ tworzy ważoną sumę wartości dla każdego zapytania.

Dla jednej sekwencji $Q,K\in\mathbb R^{T\times d_k}$, $V\in\mathbb R^{T\times d_v}$, $QK^\top\in\mathbb R^{T\times T}$, a wynik należy do $\mathbb R^{T\times d_v}$. Dla batcha rozmiaru $B$ pierwszy wymiar ma rozmiar $B$, a każdy przykład ma własną macierz wag $T\times T$.

Symbol $B$ oznacza rozmiar batcha. $N$ zostaje zarezerwowane dla liczby bloków Transformer.

## Przykład z trzema pozycjami

Dla $d_k=4$ przyjmowane są

$$Q=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&0&0\end{pmatrix},\quad K=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&1&1&1\end{pmatrix},\quad V=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}.$$

Iloczyny zapytań i kluczy wynoszą

$$QK^\top=\begin{pmatrix}2&0&2\\0&2&2\\1&1&2\end{pmatrix}.$$

Po podzieleniu przez $\sqrt4=2$ otrzymywane są

$$\begin{pmatrix}1{,}0&0{,}0&1{,}0\\0{,}0&1{,}0&1{,}0\\0{,}5&0{,}5&1{,}0\end{pmatrix}.$$

Softmax daje wagi

$$\begin{pmatrix}0{,}4223&0{,}1554&0{,}4223\\0{,}1554&0{,}4223&0{,}4223\\0{,}2741&0{,}2741&0{,}4519\end{pmatrix}.$$

Końcowa ważona kombinacja $V$ ma postać

$$\begin{pmatrix}3{,}000&4{,}000\\3{,}534&4{,}534\\3{,}356&4{,}356\end{pmatrix}.$$

Notebook potwierdza zgodność z `F.scaled_dot_product_attention` przy `atol=1e-6`.

## Multi-head attention

Multi-head attention korzysta z $H$ niezależnych głów. Przy $d_k=d_{\text{model}}/H$ każda głowa działa w mniejszej przestrzeni.

$$\operatorname{head}_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V),\qquad i=1,\ldots,H$$

$$\operatorname{MultiHead}(X)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O$$

Każda głowa ma własne projekcje. Wyniki są sklejane i przechodzą przez wspólną macierz $W^O\in\mathbb R^{d_{\text{model}}\times d_{\text{model}}}$. Symbol $H$ oznacza tutaj liczbę głów, a nie wysokość obrazu z modułu 6.

`nn.MultiheadAttention` łączy projekcje wszystkich głów w jednej macierzy $W^{QKV}\in\mathbb R^{d_{\text{model}}\times3d_{\text{model}}}$. Jest to równoważne osobnym projekcjom. Ręczne rozcięcie `in_proj_weight` zgadza się z implementacją PyTorch przy `atol=1e-5`.

❌ Każda głowa musi nauczyć się osobnej, łatwej do nazwania relacji.

✅ Specjalizacja może pojawić się podczas treningu, ale nie wynika automatycznie z architektury. Losowe głowy nie mają z góry przypisanych ról.

## Maskowanie

Maska paddingu blokuje dostęp do pozycji wypełnienia. Maska przyczynowa blokuje pozycje $j>i$, aby model autoregresywny nie korzystał z przyszłych tokenów.

❌ Jedna z tych masek może zastąpić drugą.

✅ Są to niezależne ograniczenia. Maska przyczynowa zależy od relacji pozycji, a maska paddingu od rzeczywistej długości konkretnej sekwencji. Decoder z batchami o różnych długościach potrzebuje obu.

### Padding w RNN i attention

`pack_padded_sequence` z modułu 9 oraz `key_padding_mask` służą podobnej intencji, ale działają inaczej.

| Mechanizm | Zachowanie pozycji paddingu | Padding jako zapytanie |
|---|---|---|
| `pack_padded_sequence` | krok nie jest przetwarzany przez RNN | nie istnieje po końcu sekwencji |
| `key_padding_mask` | pozycja nadal przechodzi przez embedding, FFN i LayerNorm; blokowana jest jej waga w attention | nadal tworzy wynik |

Sama maska attention nie usuwa całego wpływu paddingu. Końcowa agregacja także musi pomijać dopełnienie. W przeciwnym razie średnia reprezentacji nadal zależy od liczby pozycji `PAD`.

### Maski boolowskie i addytywne

Maska boolowska używa dwóch wartości logicznych, lecz znaczenie `True` zależy od API. Maska addytywna jest dodawana do $QK^\top$ przed softmaxem. Dozwolone pozycje otrzymują $0$, a zablokowane $-\infty$, dzięki czemu ich waga po softmaxie wynosi $0$.

### Wszystkie klucze zamaskowane

Jeśli całe zapytanie ma zablokowane klucze, naiwny softmax otrzymuje same wartości $-\infty$ i oblicza $0/0$. Ręczna implementacja w notebooku zwraca wtedy `NaN`. `F.scaled_dot_product_attention` obsługuje ten przypadek i zwraca zero.

W `nn.MultiheadAttention` wynik zależy również od `need_weights`. Domyślne `need_weights=True` korzysta z jawnej ścieżki softmaxu i może dać `NaN`. Ustawienie `need_weights=False` pozwala użyć zoptymalizowanej ścieżki zwracającej zero. Własna implementacja powinna zapewnić co najmniej jeden dozwolony klucz dla każdego wykorzystywanego zapytania.

### Semantyka masek w API PyTorch

Maski boolowskie nie są przenośne między interfejsami.

| API | Parametr | Znaczenie `True` |
|---|---|---|
| `F.scaled_dot_product_attention` | `attn_mask` | pozycja dozwolona |
| `nn.MultiheadAttention` | `attn_mask` | pozycja zablokowana |
| `nn.MultiheadAttention` | `key_padding_mask` | padding, pozycja zablokowana |

`nn.MultiheadAttention` jest wewnętrznie spójne, ale stosuje konwencję odwrotną do `F.scaled_dot_product_attention`. Błędnie przeniesiona maska nie musi wywołać wyjątku. Może po cichu odwrócić zbiór dostępnych pozycji.

## Informacja o pozycji

Wzór attention nie korzysta bezpośrednio z indeksów $i$ i $j$. Permutacja wierszy $X$ powoduje taką samą permutację wyjścia. Jest to równoważność permutacyjna. Notebook potwierdza ją przy `atol=1e-6`.

Połączenie self-attention z agregacją nieczułą na kolejność, na przykład średnią, jest niezmiennicze permutacyjnie. Taki model nie może rozwiązać zadania, którego etykieta zależy wyłącznie od kolejności tokenów.

❌ Self-attention automatycznie zna pozycje elementów sekwencji.

✅ Informacja o pozycji musi zostać dodana do wejścia. Positional encoding uzupełnia embedding tokenu przed warstwą attention.

Sinusoidalne kodowanie z oryginalnego Transformera ma postać

$$PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).$$

Każda para współrzędnych wykorzystuje inną częstotliwość. Kodowanie jest deterministyczne, nie dodaje parametrów i może częściowo działać dla długości niewidzianych podczas treningu.

Alternatywą jest uczony embedding pozycji `nn.Embedding(max_len, d_model)`. Może lepiej dopasować się do zadania, ale nie obsługuje indeksów poza `max_len`.

## Blok Transformer

W wariancie Post-LN stosowanym w oryginalnym Transformerze

$$x'=\operatorname{LayerNorm}\big(x+\operatorname{MultiHead}(x)\big),$$

$$\operatorname{out}=\operatorname{LayerNorm}\big(x'+\operatorname{FFN}(x')\big).$$

FFN działa niezależnie na każdej pozycji.

$$\operatorname{FFN}(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2$$

Jej wymiar wewnętrzny jest zwykle około cztery razy większy niż $d_{\text{model}}$.

- Połączenia rezydualne tworzą krótszą drogę dla gradientu, podobnie jak w ResNet z modułu 7.
- LayerNorm normalizuje każdą pozycję osobno po wymiarze cech, a nie po batchu.
- Dropout jest zwykle stosowany po attention i FFN, przed dodaniem ścieżki rezydualnej.

W wariancie Pre-LN normalizacja poprzedza podwarstwę, na przykład $x'=x+\operatorname{MultiHead}(\operatorname{LayerNorm}(x))$. Wybór wpływa na stabilność głębokich modeli.

| Poziom | Zawartość |
|---|---|
| pojedyncza głowa | jedno $\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V)$ |
| warstwa multi-head | $H$ głów, konkatenacja i $W^O$ |
| blok Transformer | multi-head, FFN, dwa residuale i dwie normalizacje |
| model | $N$ bloków, embeddingi oraz głowica wejściowa lub wyjściowa |

Wysokopoziomowe klasy PyTorch ukrywają te elementy. W tym module blok jest budowany jawnie, aby pokazać przepływ danych.

## Encoder, decoder i cross-attention

Oryginalny Transformer jest architekturą encoder-decoder. Encoder przetwarza całe wejście bez maski przyczynowej. Decoder generuje kolejne tokeny z maską przyczynową i korzysta z cross-attention do wyjść encodera.

Self-attention oblicza $Q$, $K$ i $V$ z jednej sekwencji. W cross-attention zapytania pochodzą z jednej sekwencji, a klucze i wartości z drugiej.

- **Encoder-only** korzysta z dwukierunkowego self-attention i pasuje do zadań takich jak klasyfikacja.
- **Decoder-only** korzysta z przyczynowego self-attention i pasuje do generowania token po tokenie.
- **Encoder-decoder** łączy oba elementy za pomocą cross-attention.

❌ Encoder-only i decoder-only różnią się tylko nazwą.

✅ Maska przyczynowa zmienia dostęp do informacji. Encoder widzi całą sekwencję, a decoder tylko bieżącą i wcześniejsze pozycje.

## Eksperymenty i obserwacje

Wyniki pochodzą z wykonanego [`examples.ipynb`](examples.ipynb).

### A. Scaled dot-product attention i maski

Ręczna implementacja zgadza się z `F.scaled_dot_product_attention` przy `atol=1e-6`. Zmiana jednego zapytania wpływa tylko na odpowiadający mu wiersz. Zmiana klucza lub wartości może wpłynąć na wszystkie zapytania.

Notebook potwierdza odmienne znaczenie masek boolowskich w dwóch API. Pokazuje także `NaN` w naiwnej implementacji dla całkowicie zamaskowanego wiersza i zero w zoptymalizowanej funkcji.

Dropout przekazany jako stały argument do funkcji nie jest wyłączany przez `model.eval()`. Dwa wywołania `NaiveAttention.eval()` mogą więc dać różne wyniki.

### B. Multi-head attention

Ręczne projekcje, podział na głowy, konkatenacja i `out_proj` zgadzają się z `nn.MultiheadAttention` przy `atol=1e-5`, zarówno dla wyjścia, jak i wag głów.

Dla $d_{\text{model}}=8$ i $H=2$ model ma 288 parametrów. Wynik zgadza się ze wzorem $3d_{\text{model}}^2+3d_{\text{model}}+d_{\text{model}}^2+d_{\text{model}}$.

### C. Kolejność i positional encoding

Self-attention jest równoważne permutacyjnie, a reprezentacja uśredniona jest niezmiennicza na permutacje. To samo dotyczy reprezentacji typu CLS, jeśli permutowane są pozostałe tokeny. Zgodność numeryczna jest sprawdzana przy `atol=1e-6`.

W zadaniu z ośmioma tokenami etykieta zależy od kolejności tokenów 0 i 1. Model z positional encoding osiąga `val_acc=test_acc=1,000`. Bez PE wyniki wynoszą `val_acc=0,475` i `test_acc=0,485`. Wynik powtarza się dla ziaren 0, 1 i 2.

Po dodaniu PE permutacja wejścia zmienia predykcję. W sprawdzonym przykładzie logity zmieniają się z $[5{,}84,-4{,}49]$ na $[6{,}01,-4{,}66]$.

### D. Transformer i GRU

Mały Transformer ma 2514 parametrów przy $T=8$ i osiąga `val_acc=1,000`, bez błędów w 200 przykładach.

W trudniejszej wersji zadania tokeny decydujące występują wcześnie, a za nimi dodawane są dystraktory. GRU ma 1922 parametry.

| $T$ | Transformer | GRU |
|---:|---:|---:|
| 8 | 1,000 | 1,000 |
| 16 | 1,000 | 1,000 |
| 32 | 1,000 | 1,000 |
| 64 | 0,860 | 0,480 |

Wynik dla $T=64$ pochodzi z jednego przebiegu. Zmierzono accuracy, nie normę gradientu. Spadek GRU jest zgodny z hipotezą opartą na niezależnym eksperymencie z modułu 9, lecz nie dowodzi, że zanikający gradient był jedyną przyczyną. Stały budżet 30 epok i pojedyncze ziarno pozostają czynnikami zakłócającymi.

Transformer także pogarsza się przy $T=64$. Czas treningu wynosi około 1,22 s dla Transformera i 2,41 s dla GRU, ale jest to mikrobenchmark krótkich sekwencji, a nie ogólne porównanie wydajności.

## Częste błędne przekonania

❌ Duża waga attention dowodzi przyczynowego wpływu tokenu na predykcję.

✅ Waga opisuje udział wartości w jednej ważonej sumie. Predykcja przechodzi jeszcze przez FFN, kolejne bloki i głowicę. Ocena przyczynowości wymaga osobnego eksperymentu.

❌ Attention usuwa wszystkie problemy długich sekwencji.

✅ Skraca drogę informacji, ale macierz $T\times T$ powoduje kwadratowy koszt pamięci i obliczeń.

❌ Mały Transformer zawsze przewyższa RNN, LSTM i GRU.

✅ Dla $T=8$, $16$ i $32$ oba modele osiągają taki sam wynik. Architektura nie gwarantuje przewagi przy każdym zadaniu i budżecie.

❌ Warstwa attention sama oznacza rozumienie treści.

✅ Mechanizm obliczeniowy, możliwości architektury i zachowanie wytrenowanego modelu są osobnymi poziomami opisu.

## Implikacje inżynierskie

- **Koszt.** Liczba porównań pozycji rośnie kwadratowo z $T$, podczas gdy koszt rekurencji rośnie liniowo.
- **Pamięć.** Prosta implementacja materializuje macierz $T\times T\times H$. Zoptymalizowane jądra, takie jak FlashAttention, mogą uniknąć jej pełnego zapisu bez zmiany wyniku matematycznego.
- **Równoległość.** Pozycje mogą być przetwarzane równolegle podczas treningu, lecz nie gwarantuje to krótszego czasu dla małych modeli i CPU.
- **Liczba głów.** Przy stałym $d_{\text{model}}$ więcej głów oznacza mniejsze $d_k$ na głowę, a nie automatycznie więcej parametrów.
- **Generowanie.** Decoder produkuje tokeny kolejno. KV-cache przechowuje wcześniejsze $K$ i $V$, aby nie obliczać ich ponownie. Mechanizm zostanie rozwinięty w module 11.

## Co warto zapamiętać

- Attention oblicza $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$.
- Multi-head attention korzysta z niezależnych projekcji, ale nie gwarantuje interpretowalnej specjalizacji.
- Semantyka maski boolowskiej zależy od używanego API.
- Self-attention bez informacji o pozycji jest równoważne permutacyjnie.
- Blok Transformer łączy multi-head attention, FFN, residuale i LayerNorm.
- Bezpośredni dostęp do pozycji pomaga przy długich zależnościach, ale koszt attention rośnie kwadratowo.

## Pytania

1. Jak będzie wyglądał rozkład wag, jeśli zapytanie jest bardzo podobne do jednego klucza i wyraźnie różne od pozostałych?
2. Dlaczego stosowane jest skalowanie przez $\sqrt{d_k}$? Co stałoby się bez niego dla dużego $d_k$?
3. Czy permutacja tokenów zmieni predykcję modelu bez positional encoding, jeśli wyniki self-attention są uśredniane?
4. Jakie dwie maski są potrzebne w autoregresywnym batchu sekwencji o różnych długościach?

## Bibliografia

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- PyTorch documentation. *torch.nn.functional.scaled_dot_product_attention*. `https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html`.
- PyTorch documentation. *torch.nn.MultiheadAttention*. `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
