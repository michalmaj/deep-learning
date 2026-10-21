Język: [English](lecture.en.md) | **Polski**

# Wykład 04. Gradient descent, SGD, momentum i Adam

## Cele wykładu

1. wyprowadzić i zapisać poprawnie reguły aktualizacji dla zwykłego gradient descent, mini-batch SGD, momentum (w konwencji PyTorch) i Adama (z korekcją obciążenia), z każdym symbolem jednoznacznie zdefiniowanym,
2. rozróżnić gradient straty, oszacowanie gradientu na mini-batchu, stan optymalizatora i wektor aktualizacji parametrów,
3. odczytać wykres poziomicowy funkcji straty i trajektorię optymalizatora oraz wyjaśnić przyczyny oscylacji, rozbieżności lub wolnej zbieżności,
4. odróżnić asymptotyczną stabilność momentum od szybkości zbieżności i zachowania w krótkim horyzoncie,
5. wyjaśnić mechanizm Adama (adaptacyjne skalowanie per-parametr) i wskazać, że to **inny** mechanizm niż tłumienie oscylacji przez momentum, nie jego "lepsza wersja",
6. wyjaśnić, dlaczego porównywanie optymalizatorów przy jednym, wspólnym learning rate i ogłaszanie zwycięzcy bywa mylące.

## Dlaczego to jest ważne

Wykład 3 wyjaśnił obliczanie gradientu przez `backward()`. Gradient wskazuje jednak tylko **kierunek**. Osobną decyzją pozostaje długość kroku, wykorzystanie historii oraz ewentualne dopasowanie kroku do każdego parametru. Od tych wyborów zależą stabilność, czas treningu i możliwość znalezienia dobrego rozwiązania.

W tym module głównym tematem staje się sposób treningu, a nie architektura. Nie istnieje jeden bezwarunkowo najlepszy optymalizator. Jego skuteczność zależy od problemu i dobranych hiperparametrów.

## Intuicja

### Optymalizacja jako poszukiwanie w krajobrazie strat

Stratę $L(\theta)$ można przedstawić jako krajobraz, którego wysokość w punkcie $\theta$ odpowiada wartości funkcji. Trening polega na stopniowym schodzeniu przy znajomości jedynie lokalnego nachylenia, bez mapy całego terenu. Gradient $\nabla L(\theta)$ wskazuje kierunek **najszybszego wzrostu**, więc aktualizacja odbywa się w stronę przeciwną.

Algorytmy optymalizacji różnią się sposobem wykorzystania tego kierunku. Zmieniają długość kroku, uwzględniają poprzednie aktualizacje albo dobierają skalę osobno dla każdego parametru.

### Dlaczego krajobraz bywa trudny

W geometrii funkcji straty często pojawiają się trzy problemy.

- **Złe uwarunkowanie** (*ill-conditioning*). Krajobraz jest stromy w jednym kierunku, a płaski w innym, tworząc wąską dolinę. Krok odpowiedni dla jednego kierunku okazuje się niedopasowany do drugiego. Ten przypadek stanowi podstawę eksperymentu.
- **Oscylacje.** Zbyt duży krok w stromym kierunku powoduje wielokrotne przeskakiwanie przez dno doliny zamiast stopniowego zbliżania się do niego.
- **Obszary płaskie i punkty siodłowe.** W wysokowymiarowych, nieliniowych funkcjach straty występują obszary o prawie zerowym gradiencie, które nie są minimami. Osobny eksperyment wymagałby większej przestrzeni parametrów niż użyta tutaj wypukła funkcja dwuwymiarowa. Momentum i adaptacyjne skalowanie mogą pomagać również w takich obszarach.

### Dlaczego "jedna liczba" (learning rate) nie wystarcza

Zwykły spadek gradientu używa jednego stałego kroku $\eta$ dla wszystkich parametrów i kierunków. Momentum dodaje **pamięć**, przez co poprzednie kroki wpływają na bieżący. Adam wprowadza **adaptację dla każdego parametru** na podstawie historii wielkości jego gradientów. Są to dwa różne mechanizmy i żaden z nich nie jest uniwersalnie lepszy.

## Formalny model

Trening iteracyjnie modyfikuje parametry $\theta$, aby zmniejszyć $L(\theta) = \frac1N\sum_{i=1}^N \ell(f(x_i;\theta), y_i)$. Potrzebne jest przy tym rozróżnienie czterech obiektów.

| Obiekt | Oznaczenie | Co to jest |
|---|---|---|
| Gradient (pełny) | $\nabla L(\theta_t)$ | pochodna straty po **całym** zbiorze danych w punkcie $\theta_t$; kosztowna i rzadko obliczana bezpośrednio |
| Oszacowanie gradientu | $\hat g_t$ | gradient z **mini-batcha**, czyli losowe i tańsze przybliżenie $\nabla L(\theta_t)$ |
| Stan optymalizatora | np. $v_t$ (momentum), $(m_t, v_t)$ (Adam) | pamięć algorytmu o wcześniejszych $\hat g_t$; przechowywana oddzielnie od parametrów modelu w `optimizer.state` |
| Wektor aktualizacji | $\Delta\theta_t = \theta_{t+1}-\theta_t$ | wartość dodawana do parametrów, zbudowana z $\hat g_t$, stanu i skali $\eta$ |

`backward()` oblicza $\hat g_t$ i zapisuje wynik w `.grad`. Następnie `optimizer.step()` wykorzystuje gradient oraz stan do wyznaczenia $\Delta\theta_t$ i aktualizacji $\theta$. Są to dwa **oddzielne** kroki.

## Matematyka

### Gradient descent (pełny batch)

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

$\eta > 0$ oznacza **learning rate**, czyli jedyny hiperparametr tej metody.

### Mini-batch SGD

Obliczanie $\nabla L(\theta_t)$ na całym zbiorze $N$ przykładów jest kosztowne. Mini-batch SGD losuje więc podzbiór $B_t$, zwykle $|B_t| \ll N$, i wyznacza

$$\hat g_t = \frac{1}{|B_t|}\sum_{i\in B_t} \nabla \ell\big(f(x_i;\theta_t), y_i\big), \qquad \theta_{t+1} = \theta_t - \eta\, \hat g_t$$

$\hat g_t$ jest **nieobciążonym estymatorem** $\nabla L(\theta_t)$. Przy jednostajnym losowaniu zachodzi $\mathbb E[\hat g_t] = \nabla L(\theta_t)$. Pojedynczy krok zawiera jednak szum, którego wariancja **maleje** wraz z rozmiarem batcha, $\text{Var}[\hat g_t] \propto 1/|B_t|$. Jest to ten sam gradient w sensie wartości oczekiwanej, lecz jego pojedyncze oszacowanie jest zaszumione.

**Full-batch** ($|B_t|=N$) usuwa szum, ale wymaga przejścia przez cały zbiór przed każdym krokiem.
**Stochastic** ($|B_t|=1$) daje najtańszy pojedynczy krok i największy szum.
**Mini-batch** stanowi praktyczny kompromis między stabilnością kierunku, kosztem pamięciowym i liczbą kroków w epoce.

### Momentum

**Konwencja.** Część podręczników używa średniej kroczącej $v_t = \beta v_{t-1} + (1-\beta)\hat g_t$, podobnej do pierwszego momentu Adama. **PyTorch (`torch.optim.SGD(momentum=...)`) stosuje inną definicję.** W kursie obowiązuje konwencja PyTorch, zgodna z używanym kodem i sprawdzona w przykładzie przez porównanie z `torch.optim.SGD`.

$$v_t = \beta v_{t-1} + \hat g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t \qquad (v_0 = 0,\ \text{więc } v_1 = \hat g_1)$$

$v_t$ jest **stanem optymalizatora**, interpretowanym jako prędkość. Parametr $\beta\in[0,1)$, zwykle $0{,}9$, określa długość pamięci poprzednich kroków. Analogia fizyczna przedstawia kulkę z bezwładnością, której ruch jest stopniowo korygowany przez bieżący gradient. Gradienty o stałym znaku zwiększają $v_t$ i przyspieszają ruch w płaskim kierunku. Naprzemienne znaki częściowo się znoszą, co może tłumić oscylacje. Zbyt duże $\beta$ przy learning rate dobranym dla zwykłego SGD może jednak **wzmocnić** oscylacje.

### Adam

Adam (*Adaptive Moment Estimation*, Kingma i Ba, 2015) utrzymuje dla każdego parametru **dwa** stany. $m_t$ jest wykładniczą średnią gradientu, nazywaną pierwszym momentem. $v_t$ stanowi wykładniczą średnią **kwadratu** gradientu i opisuje jego typową skalę. Symbol $v_t$ oznacza tutaj inny obiekt niż w sekcji o momentum, zgodnie z notacją PyTorch i oryginalnej pracy.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\hat g_t^2 \qquad (m_0=v_0=0)$$

$\beta_1$, domyślnie $0{,}9$, kontroluje pamięć **kierunku** w konwencji średniej ważonej z czynnikiem $1-\beta_1$. Nie jest to konwencja PyTorch-SGD z poprzedniej sekcji. $\beta_2$, domyślnie $0{,}999$, odpowiada za dłuższą pamięć **skali** gradientu. $v_t$ rośnie dla parametrów z dużymi gradientami i pozostaje mniejsze dla parametrów z małymi gradientami.

**Korekcja obciążenia.** Początkowe wartości $m_0=v_0=0$ zaniżają pierwsze estymaty. Dla $t=1$ zachodzi $m_1=(1-\beta_1)\hat g_1$, czyli przy $\beta_1=0{,}9$ pozostaje tylko $10\%$ wartości $\hat g_1$. Dzielenie przez $(1-\beta^t)$ koryguje ten efekt. Korekta jest silna dla małego $t$ i zanika, gdy mianownik zbliża się do 1.

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

**Aktualizacja:**

$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

$\varepsilon$, domyślnie $10^{-8}$, zapobiega dzieleniu przez zero przy małym $\hat v_t$. Dzielenie przez $\sqrt{\hat v_t}$ zmniejsza efektywny krok parametrów, które historycznie miały **duże** gradienty, i zwiększa go dla parametrów z **małymi** gradientami. Tak działa adaptacyjne skalowanie dla każdego parametru, którego nie mają SGD ani momentum.

## Przykład rozwiązany

Przykład obejmuje trzy kroki aktualizacji dla $f(\theta)=\frac12\theta^2$, gdzie $g(\theta)=\theta$ i $\theta_0=5$. Wszystkie algorytmy otrzymują celowo ten sam $\eta=0{,}5$. Momentum korzysta z $\beta=0{,}5$, a Adam z $\beta_1=0{,}9$, $\beta_2=0{,}999$ oraz $\varepsilon=10^{-8}$. Wspólny learning rate uwidacznia różnice między metodami, choć w praktyce zwykle wymaga osobnego dostrojenia.

**SGD:** $\theta_1 = 5 - 0{,}5\cdot5 = 2{,}5$; $\theta_2=2{,}5-0{,}5\cdot2{,}5=1{,}25$; $\theta_3=1{,}25-0{,}5\cdot1{,}25=0{,}625$.

**Momentum ($\beta=0{,}5$):**

| $t$ | $g_t=\theta_{t-1}$ | $v_t=\beta v_{t-1}+g_t$ | $\theta_t=\theta_{t-1}-\eta v_t$ |
|---|---|---|---|
| 1 | $5$ | $5$ | $5-0{,}5\cdot5=2{,}5$ |
| 2 | $2{,}5$ | $0{,}5\cdot5+2{,}5=5{,}0$ | $2{,}5-0{,}5\cdot5{,}0=0{,}0$ |
| 3 | $0$ | $0{,}5\cdot5{,}0+0=2{,}5$ | $0-0{,}5\cdot2{,}5=-1{,}25$ |

W kroku 3 gradient w $\theta_2=0$ wynosi $0$, mimo to parametr przechodzi do $-1{,}25$. Stan $v_3$ nadal zawiera poprzedni ruch. Jest to bezpośrednia ilustracja bezwładności i możliwości **przestrzelenia** minimum przez momentum.

**Adam ($\beta_1=0{,}9$, $\beta_2=0{,}999$, $\varepsilon=10^{-8}$):**

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | $\theta_t$ |
|---|---|---|---|---|---|---|
| 1 | $5{,}000$ | $0{,}500$ | $0{,}025$ | $5{,}000$ | $25{,}000$ | $4{,}500$ |
| 2 | $4{,}500$ | $0{,}900$ | $0{,}045$ | $4{,}737$ | $22{,}624$ | $4{,}002$ |
| 3 | $4{,}002$ | $1{,}210$ | $0{,}061$ | $4{,}466$ | $20{,}419$ | $3{,}508$ |

Pełne wyprowadzenie tabeli znajduje się w notebooku i zgadza się z `torch.optim.Adam` do szóstej cyfry. Przy tym samym $\eta=0{,}5$ Adam przechodzi przez wartości $5\to4{,}5\to4{,}0\to3{,}5$, wolniej niż SGD. Po ustabilizowaniu estymat iloraz $\hat m_t/\sqrt{\hat v_t}$ jest bliski $\pm1$, więc efektywny krok ma skalę zbliżoną do $\eta$, niemal niezależnie od wielkości gradientu. W SGD krok skaluje się bezpośrednio z gradientem. Z tego powodu dobre wartości $\eta$ dla obu algorytmów zwykle są różne.

## Eksperyment

Notebook [`examples.ipynb`](examples.ipynb) zawiera dwa eksperymenty.

**1. Geometria optymalizacji.** Funkcja $f(x,y)=\frac12(25x^2+y^2)$ jest stroma w kierunku $x$ i płaska w kierunku $y$. Trajektorie zaczynają się w $(1,1)$, a minimum znajduje się w $(0,0)$.

- SGD dla kilku wartości $\eta$ pokazuje teoretyczną granicę stabilności $\eta<2/25=0{,}08$. Dla $\eta=0{,}02$ jest stabilny, lecz wolny. Przy $\eta=0{,}06$ oscyluje i zbiega, a przy $\eta=0{,}1$ staje się rozbieżny.
- Momentum otrzymuje **ten sam** $\eta$ co oscylujący SGD. Wartość $\beta=0{,}3$ tłumi oscylacje i przyspiesza ruch w płaskim kierunku. Dla $\beta=0{,}9$ algorytm pozostaje asymptotycznie stabilny, ale ma długi, silnie oscylacyjny stan przejściowy. Mniejsze $\eta$ daje lepszy wynik przy krótkim budżecie kroków.
- Adam jest uruchamiany z kilkoma wartościami $\eta$. Trajektorie $x$ i $y$ są niemal identyczne pomimo 25-krotnej różnicy krzywizny, co pokazuje efekt adaptacyjnego skalowania.

**2. Trening małego MLP.** Syntetyczne dane zawierają dwa przeplatające się półksiężyce i $N=2\,000$ przykładów. Powstają lokalnie, są dzielone w proporcji 80/20 na trening i walidację oraz trafiają do mini-batchy po 64 przykłady.

- SGD, momentum i Adam są porównywane przy **wspólnym** $\eta=0{,}01$. Jest to eksperyment kontrolowany, a nie ogólny ranking metod.
- Drugie porównanie używa rozsądnego $\eta$ dobranego **osobno** dla każdego algorytmu.
- Przegląd $\eta$ dla SGD pokazuje przejście od stabilnego treningu przez niestabilność do rozbieżności.

## Obserwacje

- Granica stabilności SGD jest zgodna z teorią, $\eta=2/a=0{,}08$ dla $f=\frac12 a x^2$ i $a=25$. Przy $\eta=0{,}06$ trajektoria $x$ oscyluje, ale strata spada do $0{,}0035$ po 40 krokach. Dla $\eta=0{,}1$ strata rośnie wykładniczo i po 40 krokach osiąga rząd $10^{15}$.
- Momentum z $\beta=0{,}3$ oraz $\eta=0{,}06$ tłumi oscylację w $x$ i przyspiesza zbieżność w $y$. W tym ustawieniu poprawia oba aspekty względem SGD.
- Momentum z $\beta=0{,}9$ i tym samym $\eta=0{,}06$ wypada słabo w pierwszych 20 krokach. Strata skacze między $\approx 0{,}1$ a $\approx 15$ i kończy ten odcinek na $\approx1{,}92$. Dłuższy przebieg przynosi jednak stratę $\approx0{,}0066$ po 40 krokach i $\approx0{,}000025$ po 100. Promień spektralny macierzy aktualizacji wynosi $\sqrt{0{,}9}\approx0{,}9487<1$ dla obu krzywizn, dlatego konfiguracja jest asymptotycznie stabilna. Mniejsze $\eta=0{,}02$ działa szybciej w budżecie 40 kroków i osiąga stratę $0{,}003$, lecz nie jest warunkiem zbieżności. **Krótki, oscylacyjny przebieg nie wystarcza do oceny stabilności.**
- Adam daje trajektorie $x$ i $y$ zgodne do czwartego miejsca po przecinku mimo 25-krotnej różnicy krzywizny. Przy zbyt dużym $\eta$, od $0{,}3$ do $0{,}5$, oscyluje w obu kierunkach jednocześnie. Jest to inny rodzaj niestabilności niż w SGD i momentum.
- Przy wspólnym $\eta=0{,}01$ Adam osiąga po 30 epokach stratę treningową i walidacyjną około $0{,}13/0{,}15$, momentum $0{,}25/0{,}30$, a SGD $0{,}33/0{,}37$. Po osobnym dostrojeniu wartości do $0{,}5$ dla SGD, $0{,}1$ dla momentum i $0{,}01$ dla Adama różnica **w dużej mierze znika**. Wyniki wynoszą odpowiednio około $0{,}14/0{,}17$, $0{,}13/0{,}15$ oraz $0{,}13/0{,}15$. Początkowa przewaga Adama wynikała głównie z odporności na wybór $\eta$.
- Przegląd $\eta$ dla SGD na MLP pokazuje gładką zbieżność przy $\eta\in\{1,3\}$, rosnącą niestabilność przy $\eta=8$–$10$ oraz brak zbieżności dla $\eta\ge15$.

## Częste błędne przekonania

❌ Gradient wskazuje najlepszy, skończony krok do zrobienia.

✅ Gradient wskazuje tylko **kierunek** najszybszego **lokalnego** wzrostu, co omówiono w wykładzie 1. Nie określa długości kroku. Sposób wykorzystania tego kierunku zależy od algorytmu. SGD stosuje krok stały, momentum uwzględnia pamięć poprzednich kroków, a Adam dobiera skalę osobno dla każdego parametru.

❌ Większy learning rate zawsze oznacza szybsze skuteczne uczenie.

✅ Po przekroczeniu granicy zależnej od krzywizny straty i algorytmu większe $\eta$ prowadzi do oscylacji albo rozbieżności. W rozważanym przykładzie SGD przy $\eta=0{,}1$ **rozbiega się wykładniczo**, choć przy $\eta=0{,}06$ nadal zbiega, mimo oscylacji.

❌ Adam gwarantuje lepszą generalizację (albo w ogóle lepszy wynik) niż SGD.

✅ W eksperymencie geometrycznym, przy osobno dobranych $\eta$, dobrze dostrojone SGD i momentum dorównują Adamowi albo nieznacznie go wyprzedzają. Przewaga Adama przy wspólnym $\eta$ na MLP wynikała głównie z odporności na nietrafiony wybór $\eta$, a nie z gwarancji lepszego minimum lub generalizacji. Wynik dotyczy jednego problemu i nie stanowi ogólnej reguły.

❌ `optimizer.step()` oblicza gradienty.

✅ `backward()` oblicza gradienty i zapisuje je w `.grad`, zgodnie z opisem z wykładu 3. `optimizer.step()` **czyta** `.grad`, aktualizuje stan optymalizatora, na przykład $v_t$ i $m_t$, a następnie modyfikuje parametry. Są to dwa oddzielne kroki o różnych zadaniach.

❌ Użycie mini-batchy oznacza, że każdy krok korzysta z tego samego gradientu, co pełny zbiór danych.

✅ Gradient mini-batcha $\hat g_t$ jest **losowym, nieobciążonym estymatorem** pełnego gradientu $\nabla L(\theta_t)$. Średnio wskazuje właściwy kierunek, ale pojedynczy krok ma wariancję zależną od rozmiaru batcha. Większy batch oznacza mniej szumu, a nie inny kierunek.

❌ Spadek straty treningowej gwarantuje spadek straty walidacyjnej.

✅ Są to dwie osobne wielkości mierzone na różnych danych. W eksperymencie na MLP zmieniają się podobnie, ponieważ w tym krótkim treningu nie występuje przeuczenie. Nie jest to jednak ogólna gwarancja. Mechanizmy prowadzące do rozbieżności obu krzywych zostaną omówione w wykładzie 5.

## Implikacje inżynierskie

- **Koszt pamięciowy stanu optymalizatora.** SGD bez momentum nie przechowuje dodatkowego stanu. SGD z momentum przechowuje jeden tensor na parametr, $v_t$, co podwaja pamięć zajmowaną przez same parametry modelu. Adam przechowuje dwa takie tensory, $m_t$ i $v_t$, więc pamięć ta rośnie trzykrotnie. Dla dużych modeli jest to istotny koszt inżynierski.
- **Koszt obliczeniowy kroku.** Adam wykonuje więcej operacji na krok niż SGD, ponieważ oblicza dwie średnie kroczące, dzielenie i pierwiastek. Przy milionach parametrów i kroków różnica się kumuluje. Zwykle nadal dominuje koszt przejścia w przód i wstecz, a nie samego `optimizer.step()`.
- **Stabilność jako koszt projektowy.** Wąski bezpieczny zakres $\eta$ dla SGD na źle uwarunkowanym problemie oznacza więcej czasu potrzebnego na strojenie przed właściwym treningiem. Ten efekt pojawia się także w laboratorium. Adaptacyjność Adama bywa wybierana właśnie z tego praktycznego powodu, a nie z powodu lepszej teorii zbieżności.
- **AdamW i harmonogramy learning rate.** W praktyce często stosuje się wariant AdamW, który inaczej realizuje weight decay. Zagadnienie to wróci przy regularyzacji w wykładzie 5. Rzadko utrzymuje się też stałe $\eta$ przez cały trening. Typowym rozwiązaniem jest jego stopniowe zmniejszanie za pomocą harmonogramu. Oba tematy wykraczają poza zakres tego wykładu.

## Co warto zapamiętać

- Gradient określa kierunek, a algorytm optymalizacji sposób jego wykorzystania. SGD stosuje krok stały, momentum krok z pamięcią, a Adam krok adaptacyjny dla każdego parametru.
- Gradient mini-batcha jest nieobciążonym, ale zaszumionym estymatorem pełnego gradientu. Nie jest innym gradientem.
- Konwencja momentum w PyTorch, $v_t=\beta v_{t-1}+\hat g_t$, różni się od średniej kroczącej spotykanej w części podręczników. Tych definicji nie należy łączyć.
- Adam normalizuje krok osobno dla każdego parametru na podstawie historii wielkości gradientów. Mechanizm ten różni się od tłumienia oscylacji przez momentum.
- Dodanie momentum zmienia zarówno obszar stabilności, jak i szybkość tłumienia oscylacji. Learning rate należy oceniać dla konkretnego algorytmu i na dostatecznie długim horyzoncie. Słaby wynik po kilku krokach nie musi oznaczać niestabilności.
- Porównanie optymalizatorów przy jednym, wspólnym $\eta$ mówi głównie o odporności na wybór $\eta$, nie o tym, który algorytm jest "ogólnie lepszy".

## Pytania dla studentów

1. Dlaczego w kroku 3 przykładu rozwiązanego dla momentum parametr nadal się porusza, mimo że gradient w tym punkcie wynosi zero?
2. Dlaczego korekcja obciążenia w Adamie ma największy wpływ na pierwszych kilku krokach, a potem praktycznie zanika?
3. Dlaczego Adam radzi sobie ze źle uwarunkowanym problemem, czyli bardzo różną krzywizną w poszczególnych kierunkach, a zwykły SGD wymaga ostrożnego doboru $\eta$?
4. Którą hipotezę warto sprawdzić najpierw, gdy strata rośnie wykładniczo od początku treningu? Dlaczego właśnie tę?

## Bibliografia

- Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015 (arXiv:1412.6980).
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). *On the importance of initialization and momentum in deep learning*. ICML 2013.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/, rozdział 8 (Optimization for Training Deep Models).
- PyTorch Contributors. Dokumentacja *torch.optim*. https://pytorch.org/docs/stable/optim.html

Pełne wpisy bibliograficzne znajdują się w pliku [`references/references.bib`](../../references/references.bib).
