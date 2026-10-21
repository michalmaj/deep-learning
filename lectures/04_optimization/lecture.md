Język: [English](lecture.en.md) | **Polski**

# Wykład 04: Optymalizacja — gradient descent, SGD, momentum i Adam

## Cele wykładu

Po tym wykładzie powinieneś/powinnaś umieć:

1. wyprowadzić i zapisać poprawnie reguły aktualizacji dla zwykłego gradient descent, mini-batch SGD, momentum (w konwencji PyTorch) i Adama (z korekcją obciążenia), z każdym symbolem jednoznacznie zdefiniowanym,
2. rozróżnić cztery różne obiekty, które łatwo ze sobą pomylić: gradient straty, oszacowanie gradientu na mini-batchu, stan optymalizatora i wektor aktualizacji parametrów,
3. odczytać wykres poziomicowy funkcji straty i trajektorię optymalizatora, i wyjaśnić na jej podstawie, dlaczego dany algorytm oscyluje, rozbiega się albo wolno pełznie,
4. odróżnić asymptotyczną stabilność momentum od szybkości zbieżności i zachowania w krótkim horyzoncie,
5. wyjaśnić mechanizm Adama (adaptacyjne skalowanie per-parametr) i wskazać, że to **inny** mechanizm niż tłumienie oscylacji przez momentum, nie jego "lepsza wersja",
6. wyjaśnić, dlaczego porównywanie optymalizatorów przy jednym, wspólnym learning rate i ogłaszanie zwycięzcy bywa mylące.

## Dlaczego to jest ważne

W wykładzie 3 nauczyłeś/aś się liczyć gradient — `backward()` już nie jest czarną skrzynką. Ale gradient to tylko **kierunek**. To, *jak* z niego skorzystać — jak długi krok zrobić, czy pamiętać poprzednie kroki, czy dostosowywać krok osobno dla każdego parametru — to osobna, bogata decyzja projektowa, od której zależy, czy sieć w ogóle się nauczy, czy trening zajmie minuty czy dni, i czy w ogóle będzie stabilny. To jest też pierwszy wykład w tym kursie, w którym *sposób* trenowania (a nie architektura) staje się głównym bohaterem — i w którym zobaczysz, że "lepszy" optymalizator nie jest pojęciem absolutnym, tylko zależnym od konkretnego problemu.

## Intuicja

### Optymalizacja jako poszukiwanie w krajobrazie strat

Wyobraź sobie stratę $L(\theta)$ jako krajobraz: wysokość w punkcie $\theta$ to wartość straty. Trening to schodzenie w dół tego krajobrazu, krok po kroku, mając dostęp tylko do lokalnego nachylenia (gradientu) w aktualnym punkcie — nie do mapy całego terenu. Gradient $\nabla L(\theta)$ wskazuje kierunek **najszybszego wzrostu** w danym punkcie (wykład 1); idziemy w stronę przeciwną.

To, co różni algorytmy optymalizacji, to **co robimy z tym kierunkiem** poza samym "idź w stronę przeciwną": czy robimy długi czy krótki krok, czy pamiętamy, w którą stronę szliśmy wcześniej, czy krok jest taki sam dla wszystkich parametrów, czy dostosowany osobno dla każdego.

### Dlaczego krajobraz bywa trudny

Trzy klasyczne problemy geometrii straty:

- **Złe uwarunkowanie** (*ill-conditioning*): krajobraz jest stromy w jednym kierunku, a płaski w innym (wąska "dolina") — krok dobry dla stromego kierunku jest za duży dla płaskiego, i odwrotnie. To główny bohater eksperymentu tego wykładu.
- **Oscylacje**: przy zbyt dużym kroku w stromym kierunku algorytm "przeskakuje" przez dno doliny tam i z powrotem, zamiast w nie zjeżdżać.
- **Obszary płaskie i punkty siodłowe**: w wysokowymiarowych, nieliniowych stratach sieci neuronowych (inaczej niż w naszej dwuwymiarowej, wypukłej misce z eksperymentu) zdarzają się obszary o niemal zerowym gradiencie, które nie są minimum — gradient tam "milczy", mimo że dalej jest, dokąd schodzić. Nie demonstrujemy tego osobnym eksperymentem (wymagałoby to znacznie większej, nieliniowej przestrzeni parametrów niż nasza 2D miska) — ale mechanizmy, które dziś poznasz (momentum, adaptacyjne skalowanie), pomagają też w takich obszarach, nie tylko przy oscylacjach.

### Dlaczego "jedna liczba" (learning rate) nie wystarcza

Zwykły spadek gradientu ma jeden globalny, stały krok $\eta$ dla wszystkich parametrów i wszystkich kierunków naraz. Momentum dodaje **pamięć** (poprzednie kroki wpływają na bieżący). Adam dodaje **adaptację per-parametr** (krok jest skalowany osobno dla każdego parametru, na podstawie historii wielkości jego własnych gradientów). To dwa **różne** pomysły na ten sam problem — i, jak zobaczysz w eksperymencie, żaden nie jest uniwersalnie lepszy.

## Formalny model

Przypomnienie z wykładu 1: trening to iteracyjne poprawianie parametrów $\theta$, żeby zmniejszyć $L(\theta) = \frac1N\sum_{i=1}^N \ell(f(x_i;\theta), y_i)$. W tym wykładzie rozróżniamy jawnie cztery obiekty, które do tej pory traktowaliśmy łącznie:

| Obiekt | Oznaczenie | Co to jest |
|---|---|---|
| Gradient (pełny) | $\nabla L(\theta_t)$ | pochodna straty po **całym** zbiorze danych w punkcie $\theta_t$ — rzadko liczony wprost (za drogi) |
| Oszacowanie gradientu | $\hat g_t$ | gradient policzony na **mini-batchu** — losowa, tania przybliżona wersja $\nabla L(\theta_t)$ |
| Stan optymalizatora | np. $v_t$ (momentum), $(m_t, v_t)$ (Adam) | pamięć algorytmu o przeszłych $\hat g_t$ — **nie** jest parametrem modelu, jest przechowywana osobno (w PyTorch: `optimizer.state`) |
| Wektor aktualizacji | $\Delta\theta_t = \theta_{t+1}-\theta_t$ | to, co faktycznie dodajemy do parametrów — zbudowane z $\hat g_t$ i stanu, przeskalowane przez $\eta$ |

`backward()` liczy $\hat g_t$ (zapisuje w `.grad`, wykład 3). `optimizer.step()` korzysta z $\hat g_t$ i stanu, żeby policzyć $\Delta\theta_t$ i zastosować go do $\theta$. To są dwa **oddzielne** kroki — tak jak w wykładzie 3 podkreślaliśmy, że `backward()` nie aktualizuje parametrów.

## Matematyka

### Gradient descent (pełny batch)

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

$\eta > 0$ to **learning rate** — jedyny hiperparametr. Poznany w wykładzie 1; tu go uogólniamy.

### Mini-batch SGD

Liczenie $\nabla L(\theta_t)$ na całym zbiorze $N$ przykładów jest kosztowne. Zamiast tego losujemy mini-batch $B_t$ (zwykle $|B_t| \ll N$) i liczymy:

$$\hat g_t = \frac{1}{|B_t|}\sum_{i\in B_t} \nabla \ell\big(f(x_i;\theta_t), y_i\big), \qquad \theta_{t+1} = \theta_t - \eta\, \hat g_t$$

$\hat g_t$ jest **nieobciążonym estymatorem** $\nabla L(\theta_t)$: przy losowaniu $B_t$ jednostajnie ze zbioru danych, $\mathbb E[\hat g_t] = \nabla L(\theta_t)$ — średnio trafia we właściwy kierunek, ale dla pojedynczego kroku ma wariancję, która **maleje** wraz z rozmiarem batcha ($\text{Var}[\hat g_t] \propto 1/|B_t|$). To jest formalny sens zdania "mini-batch to nie to samo, co cały zbiór" z sekcji *Częste błędne przekonania*: nie jest to inny, gorszy gradient — to ten sam gradient w oczekiwaniu, tylko zaszumiony.

**Full-batch** ($|B_t|=N$): zero szumu, ale jeden krok na cały przebieg przez dane (drogi).
**Stochastic** ($|B_t|=1$): maksymalny szum, najtańszy pojedynczy krok.
**Mini-batch**: kompromis używany w praktyce niemal zawsze — wystarczająco dużo przykładów, żeby szum nie zdominował kierunku, wystarczająco mało, żeby zmieścić się w pamięci i robić wiele kroków na epokę.

### Momentum

**Uwaga o konwencji.** Różne podręczniki definiują zmienną momentum inaczej — część używa średniej kroczącej $v_t = \beta v_{t-1} + (1-\beta)\hat g_t$ (podobnej do pierwszego momentu Adama niżej). **PyTorch (`torch.optim.SGD(momentum=...)`) tego nie robi.** W tym kursie konsekwentnie używamy konwencji PyTorch, bo to z niej faktycznie korzysta nasz kod (zweryfikowane poniżej, w *Przykładzie rozwiązanym*, przez bezpośrednie porównanie z `torch.optim.SGD`):

$$v_t = \beta v_{t-1} + \hat g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t \qquad (v_0 = 0,\ \text{więc } v_1 = \hat g_1)$$

$v_t$ to **stan optymalizatora** (tu: "prędkość") — $\beta\in[0,1)$ (zwykle $0{,}9$) kontroluje, jak długo pamiętamy poprzednie kroki. Intuicja fizyczna: kulka tocząca się po krajobrazie straty ma **bezwładność** — nie zatrzymuje się natychmiast, gdy nachylenie się zmienia, tylko kontynuuje ruch, stopniowo korygowany przez bieżący gradient. Gdy kolejne $\hat g_t$ wskazują konsekwentnie w tę samą stronę (płaski, ale jednokierunkowy kierunek), $v_t$ narasta i przyspiesza ruch. Gdy kolejne $\hat g_t$ oscylują między przeciwnymi znakami (stromy kierunek "w tę i z powrotem"), ich wkłady do $v_t$ częściowo się znoszą — stąd tłumienie oscylacji. **Ale**, jak zobaczysz w eksperymencie, zbyt duże $\beta$ przy tym samym $\eta$ co dla zwykłego SGD może oscylację **wzmocnić**, nie stłumić — to nie jest darmowy mechanizm.

### Adam

Adam (*Adaptive Moment Estimation*, Kingma i Ba, 2015) utrzymuje **dwa** stany optymalizatora per parametr: wykładniczą średnią kroczącą gradientu ($m_t$, "pierwszy moment", podobny duchem do momentum) i wykładniczą średnią kroczącą **kwadratu** gradientu ($v_t$, "drugi moment", miara typowej wielkości/skali gradientu — inne $v_t$ niż w sekcji momentum powyżej, tu to inny obiekt o tej samej literze, zgodnie z notacją PyTorch i oryginalnej pracy; kontekst jednoznacznie je rozróżnia):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\hat g_t^2 \qquad (m_0=v_0=0)$$

$\beta_1$ (domyślnie $0{,}9$) kontroluje pamięć **kierunku** — podobnie jak momentum, ale tu w konwencji średniej ważonej (z czynnikiem $1-\beta_1$), nie w konwencji PyTorch-SGD z sekcji wyżej. $\beta_2$ (domyślnie $0{,}999$, czyli dłuższa pamięć niż $\beta_1$) kontroluje pamięć **skali** gradientu per parametr — $v_t$ rośnie dla parametrów, które konsekwentnie mają duże gradienty, i maleje dla tych z małymi.

**Korekcja obciążenia.** Ponieważ $m_0=v_0=0$, pierwsze kilka estymat jest systematycznie zaniżonych (ściągniętych w stronę zera) — dla $t=1$: $m_1=(1-\beta_1)\hat g_1$, czyli tylko $10\%$ prawdziwej wartości $\hat g_1$ przy $\beta_1=0{,}9$. Korygujemy to dzieląc przez $(1-\beta^t)$, co dla małych $t$ jest małą liczbą (silna korekta) i dąży do $1$ dla dużych $t$ (korekta zanika, gdy "rozgrzewka" się skończy):

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

**Aktualizacja:**

$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

$\varepsilon$ (domyślnie $10^{-8}$) zapobiega dzieleniu przez zero, gdy $\hat v_t$ jest bliskie zeru. Efekt dzielenia przez $\sqrt{\hat v_t}$: parametr, który historycznie miał **duże** gradienty, dostaje **mniejszy** efektywny krok (duże $\sqrt{\hat v_t}$ w mianowniku); parametr z **małymi** gradientami dostaje **większy** efektywny krok. To jest dokładnie adaptacyjne skalowanie per-parametr, o którym mowa w *Intuicji* — mechanizm, którego SGD i momentum nie mają w ogóle.

## Przykład rozwiązany

Policzmy ręcznie trzy kroki aktualizacji dla $f(\theta)=\frac12\theta^2$ (więc $g(\theta)=\theta$), $\theta_0=5$, $\eta=0{,}5$ dla wszystkich trzech algorytmów (ten sam $\eta$ celowo — zobacz, jak różnie reagują; w praktyce rzadko jest to dobry wybór dla wszystkich naraz, co pokażemy w eksperymencie), momentum $\beta=0{,}5$, Adam $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\varepsilon=10^{-8}$.

**SGD:** $\theta_1 = 5 - 0{,}5\cdot5 = 2{,}5$; $\theta_2=2{,}5-0{,}5\cdot2{,}5=1{,}25$; $\theta_3=1{,}25-0{,}5\cdot1{,}25=0{,}625$.

**Momentum ($\beta=0{,}5$):**

| $t$ | $g_t=\theta_{t-1}$ | $v_t=\beta v_{t-1}+g_t$ | $\theta_t=\theta_{t-1}-\eta v_t$ |
|---|---|---|---|
| 1 | $5$ | $5$ | $5-0{,}5\cdot5=2{,}5$ |
| 2 | $2{,}5$ | $0{,}5\cdot5+2{,}5=5{,}0$ | $2{,}5-0{,}5\cdot5{,}0=0{,}0$ |
| 3 | $0$ | $0{,}5\cdot5{,}0+0=2{,}5$ | $0-0{,}5\cdot2{,}5=-1{,}25$ |

Zwróć uwagę na krok 3: gradient w $\theta_2=0$ wynosi $0$ (jesteśmy dokładnie w minimum!), a mimo to $\theta$ **rusza dalej**, na $-1{,}25$ — bo $v_3$ wciąż "pamięta" poprzedni ruch. To jest dosłowna ilustracja bezwładności: momentum potrafi **przestrzelić** minimum, nawet gdy bieżący gradient milczy.

**Adam ($\beta_1=0{,}9$, $\beta_2=0{,}999$, $\varepsilon=10^{-8}$):**

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | $\theta_t$ |
|---|---|---|---|---|---|---|
| 1 | $5{,}000$ | $0{,}500$ | $0{,}025$ | $5{,}000$ | $25{,}000$ | $4{,}500$ |
| 2 | $4{,}500$ | $0{,}900$ | $0{,}045$ | $4{,}737$ | $22{,}624$ | $4{,}002$ |
| 3 | $4{,}002$ | $1{,}210$ | $0{,}061$ | $4{,}466$ | $20{,}419$ | $3{,}508$ |

(pełne wyprowadzenie liczb w tej tabeli — w notebooku, zweryfikowane co do szóstej cyfry względem `torch.optim.Adam`). Zauważ, jak wolno Adam schodzi przy tym samym $\eta=0{,}5$ co SGD ($\theta$: $5\to4{,}5\to4{,}0\to3{,}5$) — efektywny krok Adama jest z grubsza rzędu $\eta$ (bo $\hat m_t/\sqrt{\hat v_t}\approx \pm1$ po ustabilizowaniu się estymat), **niezależnie od rzeczywistej skali gradientu** — zupełnie inaczej niż SGD, gdzie krok skaluje się wprost z wielkością gradientu. To nie jest wada ani zaleta sama w sobie — to po prostu **inny** mechanizm, co ma bezpośrednią konsekwencję: dobry $\eta$ dla Adama zwykle **nie jest** dobrym $\eta$ dla SGD, i odwrotnie (zobacz eksperyment).

## Eksperyment

Notebook [`examples.ipynb`](examples.ipynb), dwa eksperymenty:

**1. Geometria optymalizacji** — funkcja $f(x,y)=\frac12(25x^2+y^2)$ (wyraźnie różna krzywizna: stromo w $x$, płasko w $y$), start $(1,1)$, minimum w $(0,0)$:
- SGD przy kilku $\eta$: pokazuje teoretyczną granicę stabilności ($\eta<2/25=0{,}08$ dla tego problemu) — stabilny-wolny poniżej ($\eta=0{,}02$), oscylujący-ale-zbieżny blisko granicy ($\eta=0{,}06$), wyraźnie rozbieżny powyżej ($\eta=0{,}1$),
- Momentum przy **tym samym** $\eta$, co oscylujący SGD: $\beta=0{,}3$ tłumi oscylację i przyspiesza w płaskim kierunku; $\beta=0{,}9$ (typowy domyślny wybór) pozostaje asymptotycznie stabilny, ale ma długi, silnie oscylacyjny stan przejściowy — mniejsze $\eta$ daje lepszy wynik w krótkim budżecie kroków,
- Adam przy kilku $\eta$: trajektorie w $x$ i $y$ są (dla tego problemu) niemal identyczne, mimo 25-krotnej różnicy krzywizny — adaptacyjne skalowanie w praktyce.

**2. Trening małego MLP** na syntetycznych danych (dwa przeplatające się półksiężyce, $N=2\,000$, generowane lokalnie, podział 80/20 trening/walidacja, mini-batche po 64 przykłady, jawna funkcja losująca indeksy z ustalonym ziarnem):
- porównanie SGD/momentum/Adam przy **wspólnym** $\eta=0{,}01$ (eksperyment kontrolowany — ale **nie** traktujemy wyniku jako ogólnego wyroku),
- to samo porównanie przy **osobno dobranym** rozsądnym $\eta$ dla każdego algorytmu,
- przegląd $\eta$ dla SGD pokazujący przejście od stabilnego, przez niestabilny, do rozbieżnego treningu.

## Co obserwujemy

- Granica stabilności zwykłego SGD na naszej misce jest dokładnie tam, gdzie przewiduje teoria ($\eta=2/a=0{,}08$ dla $f=\frac12 a x^2$, $a=25$): przy $\eta=0{,}06$ (poniżej granicy) trajektoria w $x$ oscyluje, ale strata maleje do $0{,}0035$ po 40 krokach; przy $\eta=0{,}1$ (powyżej granicy) strata rośnie wykładniczo — po 40 krokach rzędu $10^{15}$.
- Momentum $\beta=0{,}3$ przy $\eta=0{,}06$ (ten sam, oscylujący dla SGD) tłumi oscylację w $x$ **i** przyspiesza zbieganie w $y$ — wygrywa z SGD na obu frontach jednocześnie.
- Momentum $\beta=0{,}9$ przy **tym samym** $\eta=0{,}06$ przez pierwszych 20 kroków wygląda słabo: strata skacze między $\approx 0{,}1$ a $\approx 15$ i kończy ten odcinek na $\approx1{,}92$. Dłuższy przebieg pokazuje jednak stratę $\approx0{,}0066$ po 40 krokach i $\approx0{,}000025$ po 100. Promień spektralny macierzy aktualizacji wynosi $\sqrt{0{,}9}\approx0{,}9487<1$ dla obu krzywizn, więc konfiguracja jest asymptotycznie stabilna. Mniejsze $\eta=0{,}02$ jest szybsze w budżecie 40 kroków (strata $0{,}003$), ale nie jest warunkiem zbieżności. **Krótki, oscylacyjny przebieg nie wystarcza do rozstrzygnięcia stabilności.**
- Adam daje w tym eksperymencie identyczne, co do czwartej cyfry po przecinku, trajektorie $x$ i $y$ na każdym kroku — mimo 25-krotnej różnicy krzywizny. Przy zbyt dużym $\eta$ ($0{,}3$–$0{,}5$) oscyluje, ale **w obu kierunkach naraz** (izotropowo), nie tylko w stromym — inny charakter niestabilności niż SGD/momentum.
- Na MLP, przy wspólnym $\eta=0{,}01$: Adam wyraźnie wygrywa (strata trening/walidacja $\approx0{,}13/0{,}15$ po 30 epokach) z momentum ($\approx0{,}25/0{,}30$) i SGD ($\approx0{,}33/0{,}37$). Po osobnym dostrojeniu $\eta$ dla każdego (SGD $0{,}5$, momentum $0{,}1$, Adam $0{,}01$) różnica **w dużej mierze znika** (odpowiednio $\approx0{,}14/0{,}17$, $\approx0{,}13/0{,}15$, $\approx0{,}13/0{,}15$ — momentum i Adam praktycznie remisują, SGD blisko za nimi). Przewaga Adama przy wspólnym $\eta$ była w dużej mierze **odpornością na wybór $\eta$**, nie bezwarunkową przewagą jakościową.
- Przegląd $\eta$ dla SGD na MLP: gładka zbieżność przy $\eta\in\{1,3\}$, rosnąca niestabilność przy $\eta=8$–$10$ (skoki straty), wyraźny brak zbieżności przy $\eta\ge15$.

## Częste błędne przekonania

❌ Gradient wskazuje najlepszy, skończony krok do zrobienia.

✅ Gradient wskazuje tylko **kierunek** najszybszego **lokalnego** wzrostu (wykład 1) — nic nie mówi o tym, jak *daleko* w tym kierunku iść. To, ile i jak zrobić z tym kierunkiem, to cała treść tego wykładu: krok stały (SGD), krok z pamięcią (momentum), krok adaptacyjny per-parametr (Adam).

❌ Większy learning rate zawsze oznacza szybsze skuteczne uczenie.

✅ Powyżej pewnej granicy (zależnej od krzywizny straty i algorytmu) większy $\eta$ oznacza oscylacje albo rozbieżność, nie szybsze uczenie — zademonstrowane wprost: SGD przy $\eta=0{,}1$ na naszej misce **rozbiega się wykładniczo**, mimo że przy $\eta=0{,}06$ wciąż zbiegał (choć oscylując).

❌ Adam gwarantuje lepszą generalizację (albo w ogóle lepszy wynik) niż SGD.

✅ W naszym eksperymencie geometrycznym, przy osobno dobranych $\eta$, dobrze dostrojony SGD/momentum dorównują Adamowi albo go nieznacznie biją. Przewaga Adama, którą zobaczyliśmy przy **wspólnym** $\eta$ na MLP, była w dużej mierze odpornością na nietrafiony wybór $\eta$, a nie gwarancją lepszego minimum ani lepszej generalizacji — i dotyczy jednego, konkretnego problemu, nie jest prawem ogólnym.

❌ `optimizer.step()` oblicza gradienty.

✅ `backward()` (wykład 3) oblicza gradienty i zapisuje je w `.grad`. `optimizer.step()` **czyta** `.grad`, aktualizuje stan optymalizatora (np. $v_t$, $m_t$) i stosuje wynikającą z niego aktualizację do parametrów — to są dwa oddzielne kroki o różnej odpowiedzialności.

❌ Użycie mini-batchy oznacza, że każdy krok korzysta z tego samego gradientu, co pełny zbiór danych.

✅ Mini-batch gradient $\hat g_t$ to **losowy, nieobciążony estymator** pełnego gradientu $\nabla L(\theta_t)$ — trafia we właściwy kierunek *średnio*, ale pojedynczy krok ma wariancję zależną od rozmiaru batcha. Większy batch to mniejszy szum, nie inny kierunek.

❌ Spadek straty treningowej gwarantuje spadek straty walidacyjnej.

✅ To dwie osobne wielkości, mierzone na różnych danych. W naszym eksperymencie na MLP akurat poruszają się razem (bez przeuczenia w tym krótkim treningu) — ale to nie jest gwarancja ogólna; mechanizmy, przez które się rozjeżdżają (przeuczenie), to temat wykładu 5.

## Implikacje inżynierskie

- **Koszt pamięciowy stanu optymalizatora**: SGD bez momentum nie przechowuje żadnego stanu; SGD z momentum przechowuje jeden tensor na parametr ($v_t$) — podwaja pamięć zajmowaną przez same parametry modelu; Adam przechowuje **dwa** tensory na parametr ($m_t$, $v_t$) — potraja ją. Dla dużych modeli to realny koszt inżynierski, nie szczegół teoretyczny.
- **Koszt obliczeniowy kroku**: Adam robi więcej arytmetyki na krok (dwie średnie kroczące, dzielenie, pierwiastek) niż SGD — przy milionach parametrów i milionach kroków ta różnica się kumuluje, choć zwykle wciąż dominuje koszt samego forward/backward, nie `optimizer.step()`.
- **Stabilność jako koszt projektowy**: wąski, "bezpieczny" zakres $\eta$ dla SGD na źle uwarunkowanym problemie (zobaczysz to też w laboratorium) oznacza więcej czasu spędzonego na strojeniu hiperparametru przed właściwym treningiem — adaptacyjność Adama bywa wybierana właśnie z tego praktycznego powodu, nie z powodu lepszej teorii zbieżności.
- **AdamW i harmonogramy learning rate — zapowiedź, nie temat**: w praktyce Adam często używany jest w wariancie AdamW (inaczej realizuje weight decay — pełne omówienie przy regularyzacji w wykładzie 5) i rzadko ze stałym $\eta$ przez cały trening — typowe jest zmniejszanie $\eta$ w trakcie treningu (*learning rate scheduler*). To są kierunki dalszej nauki, nie część tego wykładu.

## Co warto zapamiętać

- Gradient to kierunek; algorytm optymalizacji decyduje, co z nim zrobić — stały krok (SGD), krok z pamięcią (momentum), krok adaptacyjny per-parametr (Adam).
- Gradient na mini-batchu jest nieobciążonym, ale zaszumionym estymatorem gradientu pełnego — nie innym gradientem.
- Konwencja momentum w PyTorch ($v_t=\beta v_{t-1}+\hat g_t$) różni się od średniej kroczącej używanej czasem w podręcznikach — nie mieszaj ich.
- Adam normalizuje krok osobno dla każdego parametru na podstawie historii wielkości jego gradientów — to inny mechanizm niż tłumienie oscylacji przez momentum.
- Dodanie momentum zmienia zarówno obszar stabilności, jak i szybkość tłumienia oscylacji — learning rate trzeba oceniać dla konkretnego algorytmu i dostatecznie długiego horyzontu, zamiast utożsamiać słaby wynik po kilku krokach z niestabilnością.
- Porównanie optymalizatorów przy jednym, wspólnym $\eta$ mówi głównie o odporności na wybór $\eta$, nie o tym, który algorytm jest "ogólnie lepszy".

## Pytania dla studentów

1. Dlaczego w kroku 3 przykładu rozwiązanego dla momentum parametr nadal się porusza, mimo że gradient w tym punkcie wynosi zero?
2. Wyjaśnij, dlaczego korekcja obciążenia w Adamie ma największy wpływ na pierwszych kilku krokach, a potem praktycznie zanika.
3. Mając źle uwarunkowany problem (bardzo różna krzywizna w różnych kierunkach), dlaczego Adam radzi sobie z nim "automatycznie", a zwykły SGD wymaga bardzo ostrożnego doboru $\eta$?
4. Dostałeś/aś trening, w którym strata rośnie wykładniczo od pierwszych kroków. Jaka jest najbardziej prawdopodobna pierwsza hipoteza do sprawdzenia, i dlaczego akurat ta?

## Bibliografia

- Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015 (arXiv:1412.6980).
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). *On the importance of initialization and momentum in deep learning*. ICML 2013.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/ — rozdział 8 (Optimization for Training Deep Models).
- PyTorch Contributors. *torch.optim* — dokumentacja. https://pytorch.org/docs/stable/optim.html

Pełne wpisy bibliograficzne: [`references/references.bib`](../../references/references.bib).
