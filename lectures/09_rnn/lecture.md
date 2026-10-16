# Wykład 09 — Modelowanie sekwencji: od prostego RNN do LSTM i GRU

## Cele wykładu

1. Zrozumieć, czym różnią się dane sekwencyjne od danych, które widzieliśmy do tej
   pory, i dlaczego kolejność obserwacji bywa rozstrzygająca.
2. Rozumieć rekurencję: stan ukryty jako skompresowaną reprezentację dotychczas
   przetworzonej sekwencji, aktualizowaną tą samą, współdzieloną funkcją na każdym
   kroku.
3. Rozumieć propagację wsteczną przez czas (BPTT) jako konsekwencję rozwinięcia
   rekurencji — i dlaczego prowadzi to do zanikającego/eksplodującego gradientu.
4. Znać architektury LSTM i GRU jako złagodzenie (nie rozwiązanie) tego problemu —
   i umieć to złagodzenie zmierzyć, nie tylko zadeklarować.
5. Poprawnie obsługiwać sekwencje o zmiennej długości (padding, maskowanie) i
   rozumieć, czym różnią się warianty many-to-one/many-to-many oraz jedno-/dwukierunkowe.

## Dlaczego to jest ważne

Wszystkie modele z modułów 1-8 przetwarzały dane, w których kolejność cech wejściowych
nie miała ustalonego znaczenia porządkowego (piksele owszem mają strukturę 2D, ale CNN
z modułu 6 wykorzystuje tę strukturę przez lokalność splotu, nie przez sekwencyjny
"krok po kroku"). Tekst, szeregi czasowe, sygnały audio, logi zdarzeń — w tych danych
**kolejność niesie informację, której zbiór cech sam w sobie nie oddaje**: "pies
ugryzł człowieka" i "człowiek ugryzł psa" mają identyczny zestaw słów. Model, który
nie potrafi wykorzystać kolejności, z definicji nie może rozróżnić takich przypadków —
zweryfikujemy to dosłownie w eksperymencie C, nie tylko stwierdzimy.

## Intuicja

Dane sekwencyjne: tekst (ciąg tokenów), szeregi czasowe (ciąg pomiarów), sygnały
(ciąg próbek), zdarzenia (ciąg akcji użytkownika) — łączy je to, że **element $t$
powinien być interpretowany w kontekście elementów $1,\ldots,t-1$**, nie w izolacji.
Sekwencje mają też zwykle **zmienną długość** — różne zdania mają różną liczbę słów —
co wymaga reprezentacji innej niż ustalony wektor cech (wrócimy do tego w sekcji o
paddingu).

**Stan ukryty** $h_t$ to próba skompresowania "tego, co istotne z $x_1,\ldots,x_t$" do
wektora o **stałym** rozmiarze, niezależnym od $t$. Intuicja: zamiast pamiętać całą
historię, model na bieżąco **aktualizuje** jedno podsumowanie, tą samą funkcją na
każdym kroku.

❌ "Stan ukryty to bezstratna pamięć całej dotychczasowej sekwencji."

✅ $h_t$ ma **ustalony, skończony** rozmiar, niezależnie od tego, czy $t=3$ czy
$t=300$ — musi więc *kompresować*, a kompresja z definicji traci informację. Im
dłuższa sekwencja, tym więcej musi się "zmieścić" w tym samym wektorze — zobaczymy w
eksperymencie B, że to ma bezpośredni, mierzalny skutek dla propagacji gradientu.

## Formalny model: vanilla RNN

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b), \qquad t = 1,\ldots,T$$

- $x_t \in \mathbb R^{d}$ — wejście w kroku $t$ (np. embedding tokenu z modułu 8,
  $d$ = wymiar embeddingu).
- $h_t \in \mathbb R^{d_h}$ — stan ukryty w kroku $t$ ($d_h$ = wymiar stanu ukrytego,
  hiperparametr niezależny od $d$).
- $W_x \in \mathbb R^{d_h\times d}$, $W_h\in\mathbb R^{d_h\times d_h}$ — macierze
  rekurencji, $b\in\mathbb R^{d_h}$ — bias.
- $h_0$ — stan początkowy, zwykle wektor zerowy (domyślnie w `torch.nn.RNN`, jeśli nie
  podano inaczej).

**Uwaga o notacji.** $h_t$ rozszerza $h$ z modułu 2 (tam: wyjście pojedynczej warstwy
ukrytej MLP) o indeks czasowy — ta sama rola (reprezentacja pośrednia), teraz
aktualizowana w czasie zamiast w głębi sieci. **Nie mylić** z bezindeksowym $h$ z
modułu 3 (krok różnic skończonych przy `gradient checking`) — tam $h$ nigdy nie ma
indeksu $t$, kontekst jest jednoznaczny. $W_x, W_h$ zapisane indeksem **dolnym**, w
odróżnieniu od $W^{(l)}$ z MLP (indeks **górny** = numer warstwy, moduł 2-3): tu nie
ma "kolejnych warstw" w sensie głębokości — $W_x$ i $W_h$ to **te same dwie macierze,
używane wielokrotnie**, raz na każdy krok czasowy.

**Kluczowa własność: współdzielenie parametrów.** W przeciwieństwie do MLP, gdzie
każda warstwa ma własne $W^{(l)}$, RNN używa **dokładnie tych samych** $W_x, W_h, b$
na kroku $1$, $2$, …, $T$ — to nie $T$ niezależnych sieci, tylko $T$-krotne
zastosowanie **jednej** funkcji z aktualizowanym stanem. Liczba parametrów RNN **nie
zależy od długości sekwencji** $T$ — model o tych samych wagach przetworzy sekwencję
długości 5 i długości 500.

❌ "RNN ma osobny zestaw wag dla każdego kroku czasowego, podobnie jak MLP ma osobne
wagi dla każdej warstwy."

✅ Dokładnie odwrotnie — to współdzielenie wag w czasie jest definiującą cechą RNN,
zweryfikowaną wprost w przykładzie rozwiązanym niżej.

## Matematyka: rozwinięcie w czasie i BPTT

**Rozwinięcie (unrolling).** $h_t = f(x_t, h_{t-1};\theta)$ zastosowane $T$ razy to,
po rozpisaniu, długie złożenie funkcji — dokładnie taka struktura, dla której mamy już
narzędzie z modułu 3: regułę łańcuchową. "Rozwinięta w czasie" sieć RNN wygląda
strukturalnie jak bardzo głęboka sieć feedforward o $T$ "warstwach", z tą różnicą, że
**wszystkie warstwy współdzielą te same wagi** $W_x, W_h$.

**BPTT (Backpropagation Through Time)** to dokładnie backpropagation z modułu 3,
zastosowany do tego rozwinięcia — nie powtarzamy tu całego wyprowadzenia. Oznaczmy
$\delta_t := \partial L/\partial h_t$ (rozszerzenie $\delta^{(l)}$ z modułu 3: tam
indeks górny = warstwa, tu indeks dolny = krok czasowy — ten sam koncept "gradientu
napływającego", inny wymiar propagacji). Z reguły łańcuchowej, propagacja $\delta$
wstecz przez jeden krok rekurencji:

$$\delta_{t-1} = \frac{\partial L}{\partial h_{t-1}} = \Big(\frac{\partial h_t}{\partial h_{t-1}}\Big)^{\!\top} \delta_t = W_h^\top \,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\, \delta_t$$

(Jacobian $\partial h_t/\partial h_{t-1}$ to iloczyn macierzy rekurencyjnej $W_h$ i
pochodnej $\tanh$ po współrzędnych — dokładnie ten sam wzorzec Jacobianu dla warstwy
liniowej + aktywacji elementwise, co w module 3, teraz zastosowany między krokami
czasowymi zamiast między warstwami.) Żeby cofnąć się o $k$ kroków, **mnożymy przez $k$
takich Jacobianów z rzędu**:

$$\delta_{t-k} = \Big(\prod_{i=0}^{k-1} W_h^\top\,\mathrm{diag}\big(1-\tanh^2(\cdot)\big)\Big)\,\delta_t$$

**To jest źródło zanikającego/eksplodującego gradientu.** Iloczyn $k$ macierzy: jeśli
typowa "wzmacniająca moc" każdego czynnika (z grubsza: największa wartość własna
$W_h$, przeskalowana przez pochodną $\tanh\in(0,1]$) jest $<1$, iloczyn **zanika**
wykładniczo z $k$; jeśli $>1$, **eksploduje** wykładniczo. Dla długich sekwencji ($T$
rzędu dziesiątek/setek) obie sytuacje są niemal nieuniknione przy losowej
inicjalizacji — zweryfikujemy to liczbowo w eksperymencie B, dla kilku długości
sekwencji i bez zakładania z góry, która sytuacja wystąpi.

## LSTM: pamięć komórki i bramki

LSTM wprowadza **drugi** stan, $c_t$ (stan komórki, *cell state*), obok $h_t$ (stan
ukryty) — $c_t$ pełni rolę "taśmy pamięci", do której dostęp jest kontrolowany przez
trzy **bramki** (wartości w $(0,1)$, interpretowane jako "ile przepuścić"):

$$
\begin{aligned}
i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) &&\text{(bramka wejściowa)}\\
f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) &&\text{(bramka zapomnienia)}\\
g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) &&\text{(kandydat na nową treść)}\\
o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) &&\text{(bramka wyjściowa)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t &&\text{(aktualizacja pamięci)}\\
h_t &= o_t \odot \tanh(c_t) &&\text{(stan ukryty = "odczyt" z pamięci)}
\end{aligned}
$$

($\odot$ = mnożenie po współrzędnych, jak w module 3.) **Różnica $h_t$ vs $c_t$:**
$c_t$ to wewnętrzna pamięć, aktualizowana przez **sumę** ($f_t\odot c_{t-1} +
i_t\odot g_t$) — ścieżka, wzdłuż której gradient może płynąć **bez** powtarzanego
mnożenia przez macierz wag na każdym kroku (w przeciwieństwie do $h_{t-1}\to h_t$ w
vanilla RNN). $h_t$ to "widoczna na zewnątrz" wersja tej pamięci, przepuszczona przez
bramkę wyjściową — to $h_t$ (nie $c_t$) trafia do głowicy klasyfikacyjnej czy do
następnej warstwy.

**Zgodność z `torch.nn.LSTM`.** PyTorch używa **dwóch osobnych biasów** na bramkę
($b_{i\cdot}$ i $b_{h\cdot}$, sumowanych) zamiast jednego $b$ — matematycznie
równoważne pojedynczemu biasowi (suma dwóch stałych to stała), tylko inny podział
parametrów. Wagi `weight_ih_l0`/`weight_hh_l0` to **cztery macierze sklejone w
jedną**, w ustalonej kolejności $[i,f,g,o]$ — zweryfikowane w notebooku przez ręczne
rozcięcie tensora wag i porównanie z `nn.LSTM` bit-exact.

❌ "LSTM całkowicie eliminuje zanikający gradient i gwarantuje zapamiętanie dowolnie
odległych informacji."

✅ LSTM **łagodzi** problem (ścieżka $c_{t-1}\to c_t$ addytywna, nie multiplikatywna
przez $W_h$ na każdym kroku), ale bramki same są funkcjami $\sigma$/$\tanh$, które też
się nasycają (moduł 2) — przy złej inicjalizacji lub bardzo długich sekwencjach LSTM
również może mieć zanikający gradient, tylko wolniej. Zmierzymy to wprost w
eksperymencie B — bez zakładania przewagi LSTM z góry.

## GRU: uproszczenie z dwiema bramkami

GRU łączy $h_t$ i $c_t$ w jeden stan i używa dwóch bramek zamiast trzech:

$$
\begin{aligned}
r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) &&\text{(bramka reset)}\\
z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) &&\text{(bramka update)}\\
n_t &= \tanh\big(W_{in} x_t + b_{in} + r_t\odot(W_{hn} h_{t-1} + b_{hn})\big) &&\text{(kandydat)}\\
h_t &= (1-z_t)\odot n_t + z_t\odot h_{t-1} &&\text{(interpolacja)}
\end{aligned}
$$

Intuicja: $z_t$ kontroluje, **ile** poprzedniego stanu zostaje (gdy $z_t\to 1$,
$h_t\approx h_{t-1}$ — stan "przechodzi" niemal bez zmian, analogicznie do ścieżki
addytywnej w LSTM); $r_t$ kontroluje, ile z $h_{t-1}$ wpływa na **nowego** kandydata
$n_t$ (gdy $r_t\to 0$, kandydat ignoruje przeszłość). GRU nie ma osobnego $c_t$ —
mniej parametrów (3 bramki zamiast 4 zestawów wag LSTM — zweryfikowane:
$3\times$ parametrów vanilla RNN, dokładnie, nie w przybliżeniu), prostszy mechanizm,
czasem porównywalna jakość — **nie zawsze**, zależy od zadania (eksperyment D).

**Zgodność z `torch.nn.GRU`:** kolejność bramek w tensorach wag to $[r,z,n]$,
zweryfikowana bit-exact w notebooku analogicznie do LSTM.

## Przykład rozwiązany: 3 kroki vanilla RNN

$d=2$ (wejście), $d_h=2$ (stan ukryty), $T=3$:

$$W_x=\begin{pmatrix}0{,}5&-0{,}3\\0{,}1&0{,}8\end{pmatrix}\quad W_h=\begin{pmatrix}0{,}2&0{,}0\\-0{,}1&0{,}4\end{pmatrix}\quad b=\begin{pmatrix}0{,}1\\-0{,}1\end{pmatrix}\quad h_0=\begin{pmatrix}0\\0\end{pmatrix}$$

Sekwencja: $x_1=(1,0)$, $x_2=(0,1)$, $x_3=(1,1)$.

$$h_1 = \tanh(W_x x_1 + W_h h_0 + b) = \tanh\begin{pmatrix}0{,}6\\0{,}0\end{pmatrix} = \begin{pmatrix}0{,}5370\\0{,}0000\end{pmatrix}$$

$$h_2 = \tanh(W_x x_2 + W_h h_1 + b) = \begin{pmatrix}-0{,}0923\\0{,}5692\end{pmatrix}, \qquad h_3 = \begin{pmatrix}0{,}2743\\0{,}7767\end{pmatrix}$$

**Zweryfikowane bit-exact względem `nn.RNN`** (notebook): te same $W_x,W_h,b$
wstawione jako `rnn.weight_ih_l0`/`weight_hh_l0`/`bias_ih_l0` (z `bias_hh_l0=0`, żeby
sumaryczny bias zgadzał się z $b$ powyżej — zobacz uwagę o dwóch biasach PyTorcha),
`nn.RNN` daje te same trzy stany co do szóstej cyfry po przecinku. **Te same $W_x,
W_h, b$** użyte trzy razy — nie trzy różne zestawy wag.

## Many-to-one, many-to-many i kierunkowość

- **Many-to-one**: cała sekwencja → jedna etykieta (np. klasyfikacja tekstu, lab
  tego modułu) — używamy $h_T$ (ostatni stan) jako reprezentacji całej sekwencji.
- **Many-to-many**: sekwencja → sekwencja tej samej długości (np. tagowanie
  każdego słowa) — używamy **wszystkich** $h_1,\ldots,h_T$.
- **Jednokierunkowy RNN**: $h_t$ zależy tylko od $x_1,\ldots,x_t$ — zgodne z
  przetwarzaniem przyczynowym/strumieniowym (w chwili $t$ nie znamy przyszłości).
- **Dwukierunkowy RNN** (bidirectional): dwa niezależne RNN, jeden czyta sekwencję
  od lewej, drugi od prawej, stany są łączone (np. konkatenowane) — $h_t$ zależy od
  **całej** sekwencji, włącznie z $x_{t+1},\ldots,x_T$.

❌ "Dwukierunkowy RNN jest po prostu lepszy, więc warto go używać zawsze."

✅ Dwukierunkowy RNN wymaga dostępu do **całej** sekwencji z góry — nie da się go
użyć do przetwarzania strumieniowego (np. live transkrypcja mowy w czasie
rzeczywistym, gdzie przyszłość jeszcze nie istnieje). Wybór jedno- vs dwukierunkowy
to decyzja wynikająca z **wymagań zadania**, nie z czystej jakości.

## Sekwencje o zmiennej długości: padding

Batch z sekwencjami o różnej długości wymaga wspólnego kształtu tensora — uzupełniamy
krótsze sekwencje specjalnym tokenem `PAD` do długości najdłuższej w batchu.

❌ "Padding to po prostu neutralny token, model nauczy się go ignorować."

✅ Padding **nie niesie treści** i, przetworzony bez specjalnej obsługi, zniekształca
stan ukryty: `tanh`/bramki LSTM-GRU **nie wiedzą same z siebie**, że dany krok to
wypełnienie, nie prawdziwy token — aktualizują $h_t$ tak samo, jak dla prawdziwego
wejścia. Konsekwencja: `out[:, -1, :]` (ostatnia pozycja w tensorze) odpowiada
**ostatniemu rzeczywistemu tokenowi tylko wtedy, gdy nie ma paddingu** — w
przeciwnym razie to stan **po** kilku krokach paddingu, nie po prawdziwej treści.
Poprawne rozwiązanie: `pack_padded_sequence` (informuje PyTorch o rzeczywistych
długościach, żeby RNN/LSTM/GRU nie przetwarzał paddingu w ogóle) albo ręczne
zebranie $h_{\ell_i}$ dla każdej sekwencji $i$ o rzeczywistej długości $\ell_i$.
Zweryfikowane w notebooku: ten sam tekst, zbatchowany z różnej długości
"sąsiadami", daje **różną** reprezentację przy naiwnym `out[:,-1,:]`, i
**identyczną** przy poprawnej obsłudze.

## Przykład obliczeniowy — eksperyment

Notebook (`examples.ipynb`):

- **A**: rozwinięcie w czasie — ręczna komórka RNN (podstawowe operacje PyTorch) vs
  `nn.RNN`, wpływ zmiany $x_1$ na późniejsze stany, współdzielenie wag.
- **B**: zanikający gradient — norma $\partial L/\partial x_1$ w funkcji długości
  sekwencji $T\in\{2,5,10,20,40,80\}$, RNN vs LSTM vs GRU, wiele ziaren; osobno norma
  $\partial L/\partial W_h$ (inna wielkość, inne zachowanie z $T$).
- **C**: zadanie zależne od kolejności — sekwencje = permutacje tego samego
  zbioru tokenów (identyczny bag-of-tokens), etykieta zależy wyłącznie od kolejności
  dwóch konkretnych tokenów; baseline mean-pooling vs RNN vs GRU.
- **D**: LSTM/GRU — bramki na krótkiej sekwencji, kształty tensorów, liczba
  parametrów i czas RNN/LSTM/GRU.

## Co obserwujemy

**A (rozwinięcie w czasie).** Ręczna komórka RNN zgadza się z `nn.RNN` co do czwartej
cyfry po przecinku dla wszystkich trzech kroków. Zmiana wyłącznie $x_1$ zmienia **każdy**
późniejszy stan ($h_1, h_2, h_3$) — bezpośrednia konsekwencja rekurencji.

**B (zanikający gradient, $d_h=8$, strata = suma kwadratów ostatniego stanu).**
Norma $\partial L/\partial x_1$ w funkcji $T$:

| $T$ | RNN | LSTM | GRU |
|---:|---:|---:|---:|
| 2 | $4{,}8\times10^{-1}$ | $6{,}1\times10^{-2}$ | $1{,}6\times10^{-1}$ |
| 10 | $9{,}8\times10^{-4}$ | $1{,}7\times10^{-3}$ | $2{,}0\times10^{-3}$ |
| 20 | $2{,}8\times10^{-7}$ | $2{,}5\times10^{-6}$ | $1{,}1\times10^{-5}$ |
| 40 | $2{,}7\times10^{-12}$ | $2{,}6\times10^{-10}$ | $3{,}0\times10^{-9}$ |
| 80 | $0$ | $1{,}5\times10^{-17}$ | $1{,}2\times10^{-17}$ |

Wszystkie trzy architektury zanikają do praktycznie zera przy $T=80$ (RNN dosłownie
underflow'uje do $0$) — LSTM/GRU **spowalniają** zanikanie, nie eliminują go. Co
więcej, przy **krótkich** sekwencjach ($T=2$) RNN ma **większy** gradient niż LSTM
($4{,}8\times10^{-1}$ vs $6{,}1\times10^{-2}$) — przewaga bramkowania ujawnia się
dopiero przy dłuższych sekwencjach, nie jest uniwersalna. Sprawdzone na 5 ziarnach
przy $T=40$: kolejność GRU > LSTM > RNN utrzymuje się w większości, ale nie
wszystkich, przebiegów (np. seed 2: LSTM $1{,}9\times10^{-9}$ > GRU
$2{,}8\times10^{-10}$) — nie jest to matematyczna pewność dla każdej inicjalizacji.

Gradient względem współdzielonej macierzy $W_h$ zachowuje się **jakościowo inaczej**:
w RNN pozostaje rzędu $1$-$3$ dla **każdego** sprawdzonego $T$ (od $2$ do $80$), bez
systematycznego spadku — bo sumuje wkłady ze wszystkich kroków, a bliskie kroki nie
muszą "podróżować" daleko wstecz. To inna wielkość niż gradient względem $x_1$, z
innym zachowaniem — nie wolno ich traktować zamiennie.

**C (zadanie zależne od kolejności).** Baseline mean-pooling: val/test accuracy
$0{,}45/0{,}51$ — dokładnie poziom przypadku (2 klasy). Vanilla RNN i GRU: **oba**
$1{,}00/1{,}00$ — perfekcyjne rozwiązanie, mimo że RNN ma $498$ parametrów wobec
$1330$ dla GRU ($2{,}7\times$ mniej). Różnica wynika z **dostępu do informacji**
(uśrednienie jest niezmiennicze na permutacje), nie z pojemności modelu.

**D (LSTM/GRU — bramki, parametry, czas).** Ręczne równania bramek LSTM (kolejność
wag $[i,f,g,o]$) zgadzają się z `nn.LSTM` bit-exact, zarówno $h_t$ jak i $c_t$.
Liczba parametrów: LSTM dokładnie $4\times$ RNN, GRU dokładnie $3\times$ (przy tym
samym $d_{\text{in}}, d_h$) — zweryfikowane, nie przybliżone. Czas forward passu
($d_{\text{in}}=32, d_h=64, T=50$, batch $64$): RNN $1{,}3$ms, GRU $2{,}0$ms
($1{,}6\times$), LSTM $3{,}4$ms ($2{,}6\times$) — rośnie z liczbą bramek, ale **nie
proporcjonalnie** do liczby parametrów ($3\times$/$4\times$) — ten sam wzorzec
"parametry $\neq$ koszt obliczeniowy", co w modułach 5-7.

## Częste błędne przekonania

❌ "RNN ma osobne, niezależne wagi dla każdego kroku czasowego."

✅ Te same $W_x, W_h, b$ są używane na każdym kroku — zweryfikowane wprost w
przykładzie rozwiązanym.

❌ "Stan ukryty to doskonała, bezstratna pamięć całej sekwencji."

✅ $h_t$ ma stały rozmiar niezależny od $T$ — musi kompresować, więc musi tracić
informację; im dłuższa sekwencja, tym więcej "walczy" o to samo miejsce.

❌ "LSTM i GRU rozwiązują problem zanikającego gradientu."

✅ Łagodzą go (ścieżka addytywna $c_{t-1}\to c_t$), nie eliminują — zmierzone w
eksperymencie B: przy dostatecznie długich sekwencjach gradient zanika również w
LSTM/GRU, tylko wolniej niż w vanilla RNN.

❌ "Dłuższa sekwencja to zawsze lepsza predykcja (więcej kontekstu)."

✅ Dłuższa sekwencja to też dłuższa ścieżka dla zanikającego gradientu i więcej
okazji do "zapomnienia" wczesnej informacji — więcej danych wejściowych nie jest
automatycznie równoznaczne z lepszym sygnałem treningowym.

❌ "Padding to nieszkodliwy, neutralny token."

✅ Padding aktywnie wpływa na stan ukryty, jeśli nie jest specjalnie obsłużony —
zobacz sekcję o paddingu wyżej.

❌ "Ostatni element tensora po paddingu zawsze odpowiada ostatniemu prawdziwemu
tokenowi."

✅ Tylko gdy sekwencja nie ma paddingu. W przeciwnym razie to krok **po**
paddingu — zweryfikowane: ta sama reprezentacja tego samego tekstu zmienia się w
zależności od tego, z jak długimi sekwencjami jest zbatchowany, jeśli nie użyto
`pack_padded_sequence` albo analogicznego mechanizmu.

❌ "`nn.Embedding` samo z siebie rozumie znaczenie tekstu."

✅ Jak w module 8: świeżo zainicjalizowany `nn.Embedding` to losowe wektory —
struktura (w tym ewentualne wykorzystanie kolejności przez kolejne warstwy RNN)
powstaje dopiero w trakcie treningu na konkretnym celu.

❌ "Wyższa accuracy modelu sekwencyjnego dowodzi, że wykorzystuje on kolejność
słów."

✅ Wyższa accuracy może wynikać z innych różnic (pojemność, liczba parametrów,
inny budżet treningu) — dowodem wykorzystania kolejności jest eksperyment
**skonstruowany tak**, by baseline bez kolejności nie miał dostępu do
rozstrzygającej informacji (eksperyment C), nie sama różnica w wyniku na dowolnym
zadaniu.

❌ "Model dwukierunkowy jest ulepszoną wersją modelu jednokierunkowego, zawsze
lepszą."

✅ Dwukierunkowy RNN wymaga całej sekwencji z góry — nieużywalny w ustawieniu
strumieniowym/przyczynowym, niezależnie od tego, czy dawałby wyższą accuracy w
ustawieniu offline.

❌ "Sieci rekurencyjne to najlepszy wybór dla każdego problemu sekwencyjnego."

✅ RNN/LSTM/GRU mają realne ograniczenia (sekwencyjność = brak równoległości w
czasie treningu, trudność z bardzo długim kontekstem) — motywacja do mechanizmu
uwagi (*attention*, moduł 10), który daje modelowi bardziej bezpośredni dostęp do
odległych pozycji, zamiast polegać wyłącznie na skompresowanym $h_t$.

## Implikacje inżynierskie

- **Koszt a długość sekwencji.** Czas obliczeń RNN/LSTM/GRU rośnie **liniowo** z
  $T$ — ale, w przeciwieństwie do CNN (moduł 6) czy warstwy liniowej, kroki **nie
  mogą być zrównoleglone** w czasie treningu w ramach jednej sekwencji: krok $t$
  wymaga $h_{t-1}$, więc musi czekać na krok $t-1$. To fundamentalne ograniczenie
  (nie szczegół implementacyjny) — jedna z głównych motywacji dla architektur, które
  mogą przetwarzać pozycje równolegle (moduł 10).
- **Pamięć podczas BPTT.** Backpropagation przez $T$ kroków wymaga zapamiętania
  **wszystkich** pośrednich $h_t$ (i $c_t$ dla LSTM) z forward passu — pamięć rośnie
  liniowo z $T$, analogicznie do głębokości sieci w module 3, tylko tu "głębokość"
  to długość sekwencji.
- **Trening vs inferencja sekwencyjna.** Podczas treningu zwykle przetwarzamy całe
  sekwencje naraz (znamy $x_1,\ldots,x_T$ z góry). Podczas inferencji strumieniowej
  (np. live) model dostaje $x_t$ pojedynczo i musi **utrzymywać stan** $h_{t-1}$
  między wywołaniami — architektonicznie inny tryb pracy.
- **Ryzyko przypadkowego przenoszenia stanu.** Jeśli stan $h_{t-1}$ z jednego
  przykładu przypadkowo "przecieka" do przetwarzania kolejnego, niezależnego
  przykładu (błąd w pętli treningowej — brak resetu stanu między przykładami/batchami),
  model uczy się na błędnym sygnale. Zawsze jawnie inicjalizuj $h_0$ dla każdego
  nowego, niezależnego przykładu.
- **Batching i padding dla wydajności.** Batchowanie przyspiesza trening
  (wykorzystanie równoległości sprzętu w wymiarze batcha, nie czasu), ale wymaga
  paddingu sekwencji o różnej długości do wspólnego kształtu — stąd konieczność
  poprawnej obsługi z sekcji wyżej, bo inaczej zysk wydajności kosztuje poprawność.
- **Koszt vs jakość RNN/LSTM/GRU.** Zmierzone w eksperymencie D: LSTM ma dokładnie
  $4\times$ parametrów vanilla RNN (4 zestawy bramek), GRU dokładnie $3\times$ — to
  bezpośrednio przekłada się na czas obliczeń na krok, mierzalny, nie tylko
  teoretyczny.
- **Motywacja do attention.** Zamiast polegać wyłącznie na skompresowanym $h_t$ jako
  jedynym "oknie" do przeszłości, mechanizm uwagi pozwala modelowi na **bezpośredni**
  dostęp do reprezentacji z dowolnej wcześniejszej pozycji, ważony tym, jak bardzo
  jest ona istotna dla bieżącego kroku — bez konieczności przepychania tej informacji
  przez każdy pośredni krok rekurencji. To temat modułu 10 — tu tylko intuicja, nie
  implementacja.

## Co warto zapamiętać

- RNN: $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$, te same $W_x,W_h,b$ na każdym kroku.
- BPTT = backpropagation zastosowany do rozwinięcia w czasie; mnożenie $k$
  Jacobianów z rzędu to źródło zanikającego/eksplodującego gradientu.
- LSTM dodaje $c_t$ i trzy bramki, GRU upraszcza do jednego stanu i dwóch bramek —
  oba łagodzą, nie eliminują, zanikanie gradientu.
- Padding wymaga jawnej obsługi (`pack_padded_sequence` albo ręczne zebranie
  ostatniego prawdziwego kroku) — inaczej reprezentacja sekwencji zależy od tego, z
  czym akurat została zbatchowana.
- RNN/LSTM/GRU mają historyczną i dydaktyczną wartość i wciąż bywają rozsądnym
  wyborem (małe zadania, ograniczenia pamięci/mocy obliczeniowej, przetwarzanie
  strumieniowe) — nie są architekturą "do wyrzucenia", ale mają realne ograniczenia,
  które motywują mechanizm uwagi.

## Pytania dla studentów

1. Masz RNN z $d_h=16$ wytrenowany na sekwencjach długości $10$. Czy ten sam model
   (bez ponownego treningu) może przetworzyć sekwencję długości $100$? Co się zmieni,
   a co nie?
2. Dlaczego gradient względem **współdzielonej macierzy** $W_h$ nie zanika tak samo
   jak gradient względem **pierwszego wejścia** $x_1$, mimo że oba "przechodzą" przez
   tę samą rozwiniętą w czasie sieć?
3. Masz batch dwóch sekwencji: długości 3 i długości 8, dopełnionych paddingiem do
   długości 8. Jakie dokładnie dwa błędy mogą wystąpić, jeśli weźmiesz po prostu
   `out[:, -1, :]` dla obu?
4. Zaprojektuj (nie implementuj) zadanie klasyfikacji sekwencji, w którym baseline
   oparty na uśrednieniu embeddingów **teoretycznie nie może** osiągnąć accuracy
   wyższej niż przypadkowa. Jaki warunek musi spełniać takie zadanie?

## Bibliografia

- Elman, J. L. (1990). *Finding Structure in Time*. Cognitive Science, 14(2),
  179-211.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural
  Computation, 9(8), 1735-1780.
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk,
  H., & Bengio, Y. (2014). *Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation*. EMNLP, 1724-1734.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training
  recurrent neural networks*. ICML, PMLR 28(3), 1310-1318.
- PyTorch documentation: *torch.nn.RNN*, *torch.nn.LSTM*, *torch.nn.GRU*,
  `https://pytorch.org/docs/stable/generated/torch.nn.RNN.html`.
- PyTorch documentation: *torch.nn.utils.rnn.pack_padded_sequence*,
  `https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html`.
