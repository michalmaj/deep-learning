Język: [English](lecture.en.md) | **Polski**

# Wykład 09. Modelowanie sekwencji za pomocą RNN, LSTM i GRU

## Cele wykładu

1. wyjaśnić znaczenie kolejności w danych sekwencyjnych,
2. opisać stan ukryty i współdzielenie parametrów w czasie,
3. powiązać BPTT z zanikającym i eksplodującym gradientem,
4. przedstawić LSTM i GRU jako mechanizmy łagodzące problem długich zależności,
5. poprawnie obsługiwać padding, różne długości i warianty wyjścia modelu.

## Dlaczego to jest ważne

Tekst, szeregi czasowe, audio i logi zdarzeń zawierają informację w kolejności obserwacji. Zdania „pies ugryzł człowieka” i „człowiek ugryzł psa” używają tych samych słów, lecz mają inne znaczenie. Model nieczuły na kolejność nie może ich rozróżnić.

CNN wykorzystuje strukturę przestrzenną obrazu, ale nie przetwarza go krok po kroku. RNN wprowadza stan aktualizowany po każdym elemencie sekwencji. Eksperyment pokaże zadanie, w którym uśrednianie tokenów traci rozstrzygającą informację, a model rekurencyjny ją zachowuje.

## Intuicja

Element $x_t$ powinien być interpretowany w kontekście $x_1,\ldots,x_{t-1}$. Sekwencje mają też różne długości, więc nie zawsze dają się bezpośrednio zapisać jako batch o wspólnym kształcie.

Stan ukryty $h_t$ kompresuje istotną część historii do wektora o stałym rozmiarze. Ta sama funkcja aktualizuje go na każdym kroku.

❌ Stan ukryty jest bezstratną pamięcią całej sekwencji.

✅ Rozmiar $h_t$ nie zależy od $t$. Ten sam wektor musi podsumować trzy lub trzysta elementów, więc kompresja traci część informacji. Długa ścieżka rekurencji utrudnia także przekazywanie gradientu do początku sekwencji.

## Vanilla RNN

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b), \qquad t = 1,\ldots,T$$

- $x_t\in\mathbb R^d$ jest wejściem w kroku $t$,
- $h_t\in\mathbb R^{d_h}$ jest stanem ukrytym,
- $W_x\in\mathbb R^{d_h\times d}$ i $W_h\in\mathbb R^{d_h\times d_h}$ są macierzami wag,
- $b\in\mathbb R^{d_h}$ jest biasem,
- $h_0$ jest zwykle wektorem zerowym.

Symbol $h_t$ rozszerza oznaczenie reprezentacji ukrytej z modułu 2 o indeks czasu. Dolne indeksy w $W_x$ i $W_h$ nie oznaczają numerów warstw. Są to te same macierze używane wielokrotnie.

### Współdzielenie parametrów

RNN korzysta z jednego zestawu $W_x,W_h,b$ na wszystkich krokach. Liczba parametrów nie zależy od długości $T$, dlatego ten sam model może przetwarzać sekwencje długości 5 i 500.

❌ Każdy krok czasowy ma osobny zestaw wag, podobnie jak warstwy MLP.

✅ Współdzielenie wag w czasie jest cechą definiującą RNN. Rozwinięcie zawiera wiele użyć tej samej komórki, a nie wiele niezależnych sieci.

## Rozwinięcie w czasie i BPTT

Rekurencję można rozwinąć jako złożenie $T$ kopii funkcji $f(x_t,h_{t-1};\theta)$. Struktura przypomina sieć feedforward o głębokości $T$, lecz wszystkie kroki współdzielą parametry.

BPTT, czyli *Backpropagation Through Time*, jest zwykłą propagacją wsteczną zastosowaną do takiego rozwinięcia. Dla $\delta_t=\partial L/\partial h_t$ jeden krok wstecz ma postać

$$\delta_{t-1}=W_h^\top\,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\,\delta_t.$$

Cofnięcie się o $k$ kroków wymaga iloczynu $k$ Jacobianów.

$$\delta_{t-k}=\Big(\prod_{i=0}^{k-1}W_h^\top\,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\Big)\delta_t$$

Jeśli typowe wzmocnienie czynników jest mniejsze od 1, gradient zanika wykładniczo. Jeśli jest większe od 1, gradient eksploduje. Pochodna $\tanh$ należy do $(0,1]$, więc nasycenie aktywacji dodatkowo sprzyja zanikaniu.

## LSTM

LSTM dodaje stan komórki $c_t$ i trzy bramki sterujące przepływem informacji.

$$
\begin{aligned}
i_t &= \sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi}) \\
f_t &= \sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf}) \\
g_t &= \tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg}) \\
o_t &= \sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho}) \\
c_t &= f_t\odot c_{t-1}+i_t\odot g_t \\
h_t &= o_t\odot\tanh(c_t)
\end{aligned}
$$

$i_t$ jest bramką wejściową, $f_t$ bramką zapomnienia, a $o_t$ bramką wyjściową. Kandydat $g_t$ dostarcza nową treść. Stan $c_t$ pełni rolę pamięci, a $h_t$ jest jej widocznym wyjściem przekazywanym dalej.

Aktualizacja $c_t$ zawiera ścieżkę addytywną. Gradient nie musi na każdym kroku przechodzić przez pełną macierz rekurencyjną, co łagodzi problem długich zależności.

PyTorch przechowuje dwa biasy na bramkę i skleja cztery macierze w kolejności $[i,f,g,o]$. Notebook ręcznie rozcina `weight_ih_l0` i `weight_hh_l0`, uzyskując zgodność z `nn.LSTM` przy `atol=1e-6`.

❌ LSTM całkowicie eliminuje zanikający gradient.

✅ Bramki także korzystają z $\sigma$ i $\tanh$, które mogą się nasycać. LSTM zwykle spowalnia zanikanie, ale nie gwarantuje zachowania dowolnie odległej informacji.

## GRU

GRU korzysta z jednego stanu i dwóch bramek.

$$
\begin{aligned}
r_t &= \sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr}) \\
z_t &= \sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz}) \\
n_t &= \tanh\big(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn})\big) \\
h_t &= (1-z_t)\odot n_t+z_t\odot h_{t-1}
\end{aligned}
$$

Bramka $z_t$ określa, ile poprzedniego stanu pozostaje, a $r_t$ kontroluje jego wpływ na nowego kandydata. Dla $z_t\to1$ stan przechodzi prawie bez zmian.

GRU ma trzy zestawy wag, a LSTM cztery. Przy tych samych wymiarach oznacza to odpowiednio dokładnie $3\times$ i $4\times$ więcej parametrów niż vanilla RNN. PyTorch przechowuje bramki GRU w kolejności $[r,z,n]$.

## Przykład trzech kroków RNN

Dla $d=d_h=2$ i $T=3$ przyjmowane są

$$W_x=\begin{pmatrix}0{,}5&-0{,}3\\0{,}1&0{,}8\end{pmatrix},\quad W_h=\begin{pmatrix}0{,}2&0{,}0\\-0{,}1&0{,}4\end{pmatrix},\quad b=\begin{pmatrix}0{,}1\\-0{,}1\end{pmatrix},\quad h_0=\begin{pmatrix}0\\0\end{pmatrix}.$$

Sekwencja zawiera $x_1=(1,0)$, $x_2=(0,1)$ i $x_3=(1,1)$.

$$h_1=\begin{pmatrix}0{,}5370\\0{,}0000\end{pmatrix},\qquad h_2=\begin{pmatrix}-0{,}0923\\0{,}5692\end{pmatrix},\qquad h_3=\begin{pmatrix}0{,}2743\\0{,}7767\end{pmatrix}.$$

Po przypisaniu tych samych wag do `nn.RNN` wyniki zgadzają się przy `atol=1e-6`. `bias_hh_l0` jest ustawiony na zero, aby suma biasów PyTorch odpowiadała $b$. Jeden zestaw wag jest używany trzy razy.

## Rodzaje wyjścia i kierunkowość

- **Many-to-one** mapuje całą sekwencję na jedną etykietę i zwykle korzysta z końcowego stanu.
- **Many-to-many** tworzy wyjście dla każdego kroku, na przykład podczas tagowania słów.
- **Model jednokierunkowy** korzysta tylko z $x_1,\ldots,x_t$ i nadaje się do przetwarzania strumieniowego.
- **Model dwukierunkowy** czyta sekwencję w obu kierunkach i łączy dwa stany.

❌ Model dwukierunkowy jest zawsze lepszy.

✅ Wymaga dostępu do całej sekwencji, więc nie może działać przyczynowo w transmisji na żywo. Kierunkowość wynika z wymagań zadania.

## Sekwencje o zmiennej długości

Batch wymaga wspólnego kształtu, dlatego krótsze sekwencje są uzupełniane tokenem `PAD`.

❌ Padding jest neutralny i model sam nauczy się go ignorować.

✅ Komórka rekurencyjna aktualizuje stan także dla paddingu. `out[:, -1, :]` odpowiada ostatniemu prawdziwemu tokenowi tylko w sekwencji bez dopełnienia. Poprawne rozwiązanie korzysta z `pack_padded_sequence` albo wybiera stan $h_{\ell_i}$ dla rzeczywistej długości $\ell_i$.

Notebook pokazuje, że naiwny ostatni stan tego samego tekstu zmienia się zależnie od długości innych elementów batcha. Poprawna obsługa długości daje identyczną reprezentację.

## Eksperymenty

Notebook [`examples.ipynb`](examples.ipynb) zawiera cztery części.

1. Ręczna komórka RNN jest porównana z `nn.RNN`.
2. Norma $\partial L/\partial x_1$ jest mierzona dla $T\in\{2,5,10,20,40,80\}$ w RNN, LSTM i GRU. Osobno analizowany jest gradient względem $W_h$.
3. Zadanie zależne wyłącznie od kolejności porównuje mean pooling z RNN i GRU.
4. Równania LSTM, liczba parametrów i czas forward passu są porównywane między architekturami.

## Obserwacje

Ręczna komórka i `nn.RNN` dają te same stany. Zmiana $x_1$ wpływa na wszystkie późniejsze $h_t$, co bezpośrednio wynika z rekurencji.

### Gradient względem pierwszego wejścia

| $T$ | RNN | LSTM | GRU |
|---:|---:|---:|---:|
| 2 | $4{,}8\times10^{-1}$ | $6{,}1\times10^{-2}$ | $1{,}6\times10^{-1}$ |
| 10 | $9{,}8\times10^{-4}$ | $1{,}7\times10^{-3}$ | $2{,}0\times10^{-3}$ |
| 20 | $2{,}8\times10^{-7}$ | $2{,}5\times10^{-6}$ | $1{,}1\times10^{-5}$ |
| 40 | $2{,}7\times10^{-12}$ | $2{,}6\times10^{-10}$ | $3{,}0\times10^{-9}$ |
| 80 | $0$ | $1{,}5\times10^{-17}$ | $1{,}2\times10^{-17}$ |

Przy $T=80$ gradient praktycznie zanika we wszystkich architekturach. LSTM i GRU spowalniają ten proces, ale go nie eliminują. Przy $T=2$ RNN ma nawet większy gradient niż LSTM. Dla pięciu ziaren przy $T=40$ kolejność GRU, LSTM, RNN występuje w większości, lecz nie we wszystkich przebiegach.

Gradient względem współdzielonego $W_h$ zachowuje się inaczej. W RNN pozostaje rzędu $1$ do $3$ dla badanych długości, ponieważ sumuje wkłady ze wszystkich kroków. Bliskie kroki nie muszą propagować sygnału przez całą sekwencję. Nie należy więc utożsamiać gradientu względem parametrów z gradientem względem $x_1$.

### Zadanie zależne od kolejności

Mean pooling osiąga accuracy walidacyjną i testową $0{,}45/0{,}51$, czyli poziom losowy. RNN i GRU osiągają $1{,}00/1{,}00$. RNN ma 498 parametrów, a GRU 1330. Przewaga wynika z dostępu do kolejności, a nie z większej pojemności.

### Parametry i czas

Równania LSTM zgadzają się z `nn.LSTM` przy `atol=1e-6`. LSTM ma dokładnie $4\times$, a GRU $3\times$ więcej parametrów niż RNN. Dla $d_{in}=32$, $d_h=64$, $T=50$ i batcha 64 czasy wynoszą około 1,3 ms dla RNN, 2,0 ms dla GRU i 3,4 ms dla LSTM. Czas nie rośnie proporcjonalnie do liczby parametrów.

## Częste błędne przekonania

❌ Dłuższa sekwencja zawsze poprawia predykcję.

✅ Więcej kontekstu oznacza również dłuższą ścieżkę gradientu i więcej okazji do utraty wczesnej informacji.

❌ `nn.Embedding` sam rozumie znaczenie tekstu.

✅ Początkowe wektory są losowe. Struktura powstaje podczas treningu na konkretnym celu.

❌ Wyższa accuracy modelu sekwencyjnego dowodzi wykorzystania kolejności.

✅ Różnica może wynikać z pojemności lub budżetu treningu. Wiarygodny eksperyment musi odebrać modelowi bez kolejności dostęp do rozstrzygającej informacji.

❌ RNN, LSTM i GRU są najlepszym wyborem dla każdego problemu sekwencyjnego.

✅ Rekurencja utrudnia równoległość i dostęp do bardzo długiego kontekstu. Ograniczenia te motywują mechanizm uwagi z modułu 10.

## Implikacje inżynierskie

- **Czas.** Koszt rośnie liniowo z $T$, a kolejne kroki jednej sekwencji nie mogą być wykonane równolegle.
- **Pamięć BPTT.** Trening przechowuje wszystkie stany $h_t$, a w LSTM także $c_t$. Zużycie pamięci rośnie liniowo z długością.
- **Inferencja strumieniowa.** Model musi zachowywać stan między kolejnymi wywołaniami.
- **Reset stanu.** Przeniesienie $h_t$ między niezależnymi przykładami powoduje przeciek informacji. Każdy nowy przykład wymaga poprawnego $h_0$.
- **Batching.** Równoległość w wymiarze batcha wymaga paddingu i prawidłowej obsługi długości.
- **Koszt bramek.** LSTM i GRU mają więcej parametrów oraz większy koszt kroku niż RNN.
- **Attention.** Bezpośredni dostęp do wcześniejszych pozycji skraca drogę informacji i ogranicza zależność od jednego skompresowanego stanu.

## Co warto zapamiętać

- RNN współdzieli $W_x,W_h,b$ na wszystkich krokach.
- BPTT mnoży Jacobiany w czasie, co prowadzi do zanikającego lub eksplodującego gradientu.
- LSTM i GRU łagodzą problem, ale go nie eliminują.
- Padding wymaga `pack_padded_sequence` albo jawnego wyboru ostatniego prawdziwego kroku.
- Rekurencja nadal bywa użyteczna w małych i strumieniowych zadaniach, choć ma ograniczenia motywujące attention.

## Pytania dla studentów

1. Czy RNN z $d_h=16$ wytrenowany na sekwencjach długości 10 może przetworzyć sekwencję długości 100? Co pozostaje stałe, a co się zmienia?
2. Dlaczego gradient względem $W_h$ może nie zanikać tak samo jak gradient względem $x_1$?
3. Jakie błędy powoduje użycie `out[:, -1, :]` dla sekwencji długości 3 i 8 dopełnionych do wspólnego rozmiaru?
4. Jak zaprojektować zadanie, w którym mean pooling nie może przekroczyć losowej accuracy, a model sekwencyjny może?

## Bibliografia

- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science, 14(2), 179-211.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP, 1724-1734.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. ICML, PMLR 28(3), 1310-1318.
- PyTorch documentation. *torch.nn.RNN*, *torch.nn.LSTM*, *torch.nn.GRU*. `https://pytorch.org/docs/stable/generated/torch.nn.RNN.html`.
- PyTorch documentation. *torch.nn.utils.rnn.pack_padded_sequence*. `https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html`.
