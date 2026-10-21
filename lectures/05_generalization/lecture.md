Język: [English](lecture.en.md) | **Polski**

# Wykład 05. Generalizacja, overfitting i regularyzacja

## Cele wykładu

1. rozróżnić ryzyko empiryczne (to, co faktycznie minimalizujemy) od ryzyka oczekiwanego (to, na czym naprawdę nam zależy), z jednoznacznie zdefiniowanymi symbolami,
2. rozpoznać na krzywych straty treningowej i walidacyjnej underfitting, dobre dopasowanie i overfitting oraz uzasadnić diagnozę konkretnymi liczbami,
3. wyprowadzić regułę aktualizacji dla regresji L2 w zwykłym SGD i wskazać dokładny warunek, w którym jest ona tożsama z "weight decay",
4. wyjaśnić, dlaczego ta tożsamość **nie** musi zachodzić dla Adama, i czym różni się od niej AdamW,
5. zapisać matematycznie inverted dropout (maska Bernoulliego, skalowanie) i wyjaśnić różnicę między `model.train()` a `model.eval()`,
6. zaimplementować early stopping, który **przywraca** najlepszy zapamiętany stan modelu.

## Dlaczego to jest ważne

Od wykładu 1 celem treningu było zmniejszanie straty na zbiorze treningowym. Nie oznacza to jednak, że model najlepiej dopasowany do tych danych okaże się najlepszy na nowych przykładach. W tym miejscu optymalizacja i generalizacja stają się osobnymi zagadnieniami. Pierwsze dotyczy szybkiego i stabilnego zmniejszania straty treningowej, a drugie skuteczności poza zbiorem użytym do uczenia. Ocena wymaga więc poprawnej metodologii, a nie jednego korzystnego przebiegu.

## Intuicja

### Rzeczywisty cel treningu

Celem jest model skuteczny na nowych danych pochodzących z tego samego rozkładu co dane treningowe. Strata treningowa stanowi jedynie **przybliżenie** tego celu, obliczone na skończonej próbce. Im lepiej próbka reprezentuje cały rozkład, tym lepsza może być generalizacja.

### Underfitting i overfitting

- **Underfitting** (niedouczenie). Model jest zbyt prosty albo trenowany zbyt krótko, aby uchwycić strukturę obecną w danych treningowych. Strata treningowa i walidacyjna pozostają wysokie i zbliżone do siebie.
- **Overfitting** (przeuczenie). Model dopasowuje się nie tylko do struktury danych, ale także do szumu konkretnej próbki. Strata treningowa nadal spada, podczas gdy walidacyjna przestaje maleć i zaczyna rosnąć.

**Zależność między pojemnością modelu a jakością generalizacji nie jest monotoniczna** i często wymaga sprawdzenia eksperymentalnego. Większy model może przeuczyć się mocniej, ale nie jest to regułą. Małe modele także ulegają przeuczeniu przy niewielkiej liczbie danych, a duże mogą dobrze generalizować przy odpowiednich danych i regularyzacji. Rozmiar modelu nie wystarcza więc do przewidzenia wyniku.

### Dlaczego regularyzacja

Regularyzacja obejmuje techniki ograniczające zachowania typowe dla przeuczenia. L2 i weight decay zniechęcają do dużych wag, dropout ogranicza zależność od pojedynczych neuronów, a early stopping zapobiega zbyt długiemu treningowi. Każda z tych metod wprowadza **kompromis**, który trzeba ocenić eksperymentalnie dla konkretnego problemu.

## Formalny model

Niech $P$ będzie (nieznanym) rozkładem prawdopodobieństwa, z którego pochodzą pary $(x,y)$. Definiujemy:

$$R(\theta) = \mathbb{E}_{(x,y)\sim P}\big[\ell(f(x;\theta), y)\big] \qquad \text{(ryzyko oczekiwane / } \textit{expected risk}\text{)}$$

Jest to średni błąd modelu w *całym* rozkładzie danych. Nie można go policzyć bezpośrednio, ponieważ rozkład $P$ pozostaje nieznany i dostępna jest tylko skończona próbka.

$$\hat R(\theta) = \frac{1}{N}\sum_{i=1}^N \ell\big(f(x_i;\theta), y_i\big) \qquad \text{(ryzyko empiryczne /}\textit{ empirical risk}\text{)}$$

Jest to $L(\theta)$ z wykładu 1, czyli średnia strata na konkretnej skończonej próbce $\{(x_i,y_i)\}_{i=1}^N$. Trening minimalizuje $\hat R(\theta)$ jako **przybliżenie** $R(\theta)$. Jakość przybliżenia rośnie wraz z liczbą reprezentatywnych danych i spada, gdy model dopasowuje się do cech właściwych wyłącznie próbce.

**Luka generalizacyjna** (*generalization gap*) ma postać $R(\theta) - \hat R(\theta)$. W praktyce przybliża ją różnica $L_{\text{val}}(\theta) - L_{\text{train}}(\theta)$. Rosnąca luka przy malejącej stracie treningowej stanowi używaną tutaj operacyjną definicję overfittingu.

**Podział danych** przypisuje każdej części inną rolę. Zbiór treningowy służy do dopasowania parametrów, walidacyjny do porównywania modeli i hiperparametrów oraz wyboru momentu zatrzymania, a testowy do **jednokrotnej** oceny wybranego modelu. Każde użycie testu podczas podejmowania decyzji włącza go do procesu doboru i pozbawia roli niezależnej miary generalizacji.

**Data leakage** (przeciek danych) występuje, gdy informacja ze zbioru walidacyjnego, testowego lub z przyszłości wpływa na trening albo dobór modelu. Przykładem jest obliczenie parametrów normalizacji na całym zbiorze przed podziałem zamiast wyłącznie na danych treningowych. Prowadzi to do zawyżonej oceny, ponieważ model pośrednio korzysta z informacji o danych ewaluacyjnych. Efekt zostanie zmierzony w laboratorium.

## Matematyka

### Regularyzacja L2 i jej wpływ na gradient (SGD)

Rozszerzamy funkcję celu o karę za duże wagi:

$$L_{\text{reg}}(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|_2^2 = L(\theta) + \frac{\lambda}{2}\sum_j \theta_j^2$$

gdzie $\lambda \geq 0$ to siła regularyzacji (hiperparametr). Gradient:

$$\nabla L_{\text{reg}}(\theta) = \nabla L(\theta) + \lambda\theta$$

Krok zwykłego SGD z tym gradientem:

$$\theta \leftarrow \theta - \eta\big(\nabla L(\theta) + \lambda\theta\big) = \underbrace{(1-\eta\lambda)\theta}_{\text{"skurczenie" wag}} - \eta\nabla L(\theta)$$

Jest to definicja **weight decay**. Każdy krok mnoży wagi przez $(1-\eta\lambda) < 1$, przesuwając je w stronę zera, a następnie wykonuje zwykłą aktualizację gradientową. **Dla zwykłego SGD dodanie $\frac{\lambda}{2}\|\theta\|^2$ do straty i pomnożenie wag przez $(1-\eta\lambda)$ w ramach tego samego kroku są dokładnie równoważne.** Notebook potwierdza zgodność z `torch.optim.SGD(weight_decay=λ)` przy `atol=1e-3`.

Równoważność wymaga jednego połączonego kroku. Wykonanie najpierw pełnej aktualizacji $\theta'=\theta-\eta\nabla L(\theta)$, a później osobnego skurczenia $(1-\eta\lambda)\theta'$ daje
$$(1-\eta\lambda)\theta'= (1-\eta\lambda)\theta - \eta(1-\eta\lambda)\nabla L(\theta),$$
czyli dodatkowy czynnik $(1-\eta\lambda)$ przy składniku gradientowym. Różnica wynosi dokładnie $\eta^2\lambda\nabla L(\theta)$, dlatego dla $\lambda\neq0$ jest to ogólnie **inna** aktualizacja. Jej kierunek zależy od znaku $\nabla L(\theta)$.

### L2 a weight decay dla optymalizatorów adaptacyjnych

W Adamie **z parametrem `weight_decay`**, zgodnie z domyślną implementacją `torch.optim.Adam`, kara L2 trafia do gradientu **przed** adaptacyjną normalizacją.

$$\hat g_t = \nabla \ell(\theta_t) + \lambda\theta_t, \qquad m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1}+(1-\beta_2)\hat g_t^2, \qquad \dots$$

Dalsza część algorytmu pozostaje taka jak w wykładzie 4. $\hat g_t$ jest dzielony przez $\sqrt{\hat v_t}$, więc mechanizm normalizujący gradient zadania normalizuje również wkład kary L2. Jeśli gradient zadania jest w przybliżeniu proporcjonalny do $\theta$, dodanie $\lambda\theta$ może nie zmienić znormalizowanego kierunku ani kroku Adama. W tej sytuacji kara L2 może być praktycznie **bezskuteczna** pomimo niezerowego $\lambda$. Notebook pokazuje taki przypadek.

**AdamW** (Loshchilov i Hutter, 2019) **odsprzęga** (*decoupling*) karę od adaptacyjnej normalizacji. Momenty $m_t$ i $v_t$ są obliczane z czystego gradientu zadania, a skurczenie wag stosuje się osobno po normalizacji.

$$m_t, v_t \ \text{liczone z } \nabla\ell(\theta_t) \text{ (bez } \lambda\theta_t\text{)}, \qquad \theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\theta_t\right)$$

Wkład $\lambda\theta_t$ nie zależy wtedy od historii ani skali gradientu zadania, podobnie jak w SGD. Z tego powodu do regularyzacji wag z Adamem zwykle stosuje się `AdamW`, także w eksperymencie tego wykładu.

### Dropout, maska Bernoulliego i *inverted dropout*

Dla warstwy o aktywacjach $a\in\mathbb R^m$ i prawdopodobieństwie odrzucenia $p\in(0,1)$, w trybie **treningowym**:

$$r_i \sim \text{Bernoulli}(1-p) \ \text{niezależnie dla każdego } i, \qquad \tilde a_i = \frac{r_i}{1-p}\, a_i$$

$r_i=0$ z prawdopodobieństwem $p$, co zeruje neuron w danym kroku, ale nie usuwa go z architektury. Nowa maska jest losowana przy każdym przejściu w przód. Dla $r_i=1$ prawdopodobieństwo wynosi $1-p$. Skalowanie przez $\frac{1}{1-p}$, nazywane *inverted dropout* i stosowane w `nn.Dropout`, zapewnia $\mathbb E[\tilde a_i] = a_i$. Oczekiwana wartość wyjścia pozostaje więc niezmieniona mimo losowego zerowania. Notebook potwierdza to na 10 000 próbkach.

W trybie **ewaluacji**, `model.eval()`, zachodzi $\tilde a_i = a_i$. Nie występuje losowanie ani skalowanie. Ponieważ skalowanie odbywa się podczas treningu, inferencja pozostaje zwykłym deterministycznym przejściem w przód.

`model.train()` i `model.eval()` przełączają **zachowanie konkretnych warstw**, w tym dropout. Nie sterują obliczaniem gradientów, za które odpowiadają `torch.no_grad()` i `requires_grad`. `model.eval()` nie wyłącza gradientów automatycznie, co notebook potwierdza przez poprawne obliczenie `.grad` w trybie `eval()`.

### Early stopping jako procedura

Po każdej epoce $e$ ze stratą walidacyjną $V_e$:

1. Jeśli $V_e < V_{\text{best}} - \delta$, gdzie $\delta$ oznacza minimalną istotną poprawę (*min_delta*), należy zapisać $V_{\text{best}}\leftarrow V_e$, zachować **pełny stan parametrów** modelu i wyzerować licznik epok bez poprawy.
2. W przeciwnym razie licznik rośnie. Po osiągnięciu `patience` trening zostaje zatrzymany.
3. Po zatrzymaniu lub wyczerpaniu budżetu epok należy **przywrócić** najlepszy zapisany stan. Model używany dalej pochodzi z epoki $e_{\text{best}}$, a nie z ostatniej epoki treningu.

Pominięcie trzeciego kroku pozostawia model w zwykle gorszym stanie z ostatniej epoki. Samo wcześniejsze zakończenie pętli nie wystarcza do poprawnego działania early stopping.

## Przykład rozwiązany

Przyjmijmy $\theta_0=2$, $\eta=0{,}1$ i celowo duże $\lambda=0{,}5$. Dla zadania $\ell(\theta)=\frac12\theta^2$ zachodzi $\nabla\ell(\theta)=\theta$, tak jak w modułach 1 i 4.

**SGD, bez regularyzacji:** $\theta_1 = 2 - 0{,}1\cdot 2 = 1{,}8$.

**SGD z L2/weight decay:** $\nabla L_{\text{reg}} = 2 + 0{,}5\cdot2 = 3{,}0$; $\theta_1 = 2 - 0{,}1\cdot3{,}0 = 1{,}7$. Równoważny zapis $(1-\eta\lambda)\theta_0 - \eta\nabla\ell(\theta_0) = 0{,}95\cdot2 - 0{,}2 = 1{,}7$ prowadzi do tej samej wartości.

**Adam, `weight_decay=0,5` (konwencja L2-w-gradiencie):**

$$\hat g_1 = \theta_0 + \lambda\theta_0 = 2+1=3{,}0 \quad\Rightarrow\quad m_1=0{,}1\cdot3{,}0=0{,}3,\ v_1=0{,}001\cdot9{,}0=0{,}009$$
$$\hat m_1 = 0{,}3/0{,}1=3{,}0,\quad \hat v_1=0{,}009/0{,}001=9{,}0 \quad\Rightarrow\quad \theta_1 = 2 - 0{,}1\cdot\frac{3{,}0}{\sqrt{9{,}0}} = 2-0{,}1 = 1{,}9$$

**Dla tego samego $\theta_0$ Adam bez regularyzacji, czyli przy $\lambda=0$, także daje $\theta_1=1{,}9$.** Wtedy $\hat g_1=\theta_0$, $\hat m_1=2{,}0$, $\hat v_1=4{,}0$, a $\theta_1=2-0{,}1\cdot\frac{2{,}0}{\sqrt{4{,}0}}=1{,}9$. Gradient zadania jest proporcjonalny do $\theta$, więc dodanie $\lambda\theta$ jedynie go skaluje. Znormalizowany kierunek Adama $\hat m/\sqrt{\hat v}$ pozostaje niewrażliwy na takie jednolite skalowanie. **Kara L2 nie zmieniła tego kroku.**

**AdamW, `weight_decay=0,5` (konwencja odsprzężona):** moment jest liczony z czystego $\nabla\ell(\theta_0)=2{,}0$, bez $\lambda$. Otrzymuje się $\hat m_1=2{,}0$, $\hat v_1=4{,}0$ i znormalizowany krok $\hat m_1/\sqrt{\hat v_1}=1{,}0$.

$$\theta_1 = \theta_0 - \eta\left(1{,}0 + \lambda\theta_0\right) = 2 - 0{,}1\cdot(1{,}0+0{,}5\cdot2) = 2-0{,}1\cdot2{,}0 = 1{,}8$$

AdamW uwzględnia regularyzację, ponieważ bez niej dałby $\theta_1=1{,}9$, a z nią otrzymuje $1{,}8$. Adam z `weight_decay` nie zmienia w tym kroku wyniku. Wszystkie cztery wartości, $1{,}8$, $1{,}7$, $1{,}9$ i $1{,}8$, zostały porównane w notebooku z odpowiednimi klasami `torch.optim`.

## Eksperyment

Notebook [`examples.ipynb`](examples.ipynb) zawiera trzy części.

1. **Trzy pojemności dla tego samego problemu.** Regresja $y=\sin(3x)+0{,}3x+\varepsilon$ korzysta z $N=30$ zaszumionych punktów treningowych i gęstego zbioru walidacyjnego z $N=200$. Porównywane są mały MLP z underfittingiem, średni MLP z dobrym dopasowaniem i duży MLP `[128,128,128]` z overfittingiem. Dane i procedura treningu są identyczne.
2. **Regularyzacja modelu podatnego na przeuczenie.** Porównanie obejmuje brak regularyzacji, `AdamW(weight_decay=...)`, `nn.Dropout`, early stopping oraz zbyt silny dropout prowadzący do underfittingu.
3. **Opcjonalny wpływ liczby przykładów treningowych.** Ten sam model jest trenowany dla $N\in\{15,30,60,120,300\}$.

Wyniki dwóch pierwszych części zostały **sprawdzone dla kilku ziaren losowości**, a nie tylko dla jednego wybranego przebiegu.

## Obserwacje

- **Underfitting.** Mały model ma wysoką, zbliżoną stratę treningową i walidacyjną, na przykład około $0{,}5$ i $0{,}6$. Nie potrafi więc dopasować nawet danych treningowych.
- **Dobre dopasowanie.** Średni model osiąga niską stratę treningową i umiarkowaną walidacyjną. Pod koniec treningu strata walidacyjna nadal pozostaje blisko swojego minimum.
- **Zmierzony overfitting.** Duży model z $33\,409$ parametrami osiąga najniższą stratę treningową spośród trzech modeli, około $0{,}01$–$0{,}05$. Strata walidacyjna najpierw spada, osiąga minimum wyraźnie przed końcem treningu, a następnie rośnie mimo dalszego spadku straty treningowej. Skala odbicia zależy od ziarna. W testach stosunek straty końcowej do minimalnej wynosił od około $1{,}1\times$ do ponad $8\times$. Kierunek efektu pozostawał stabilny, ale jego wielkość nie.
- **Weight decay z AdamW pomaga umiarkowanie.** Przy dobrze dobranym $\lambda$ rzędu $10^{-2}$ końcowa strata walidacyjna wyraźnie spadała względem modelu bez regularyzacji. Wielkość poprawy zależała od ziarna.
- **Dropout działał mocniej i bardziej konsekwentnie niż weight decay.** Przy $p\approx0{,}1$ końcowa strata walidacyjna była zbliżona do najlepszego wyniku modelu bez regularyzacji, a czasem od niego niższa. Jest to obserwacja właściwa temu problemowi i architekturze, a nie ogólna przewaga dropout nad weight decay.
- **Zbyt silny dropout szkodzi.** Przy $p=0{,}5$ strata treningowa wyraźnie rosła, ponieważ model nie potrafił dopasować nawet danych treningowych. Regularyzacja zamieniła overfitting w underfitting.
- **Early stopping wymaga przywrócenia stanu.** Strata w momencie zatrzymania bywała wyraźnie gorsza niż w najlepszej zapamiętanej epoce. Korzyść pojawiała się dopiero po przywróceniu zapisanych wag.
- **Dropout wpływa na ustawienia early stopping.** Połączenie dropout z `patience` dobranym dla modelu bez dropout prowadziło do przedwczesnego zatrzymania. Dodatkowy szum krzywej walidacyjnej sprawia, że ten sam próg cierpliwości może być zbyt niski. Jest to rzeczywista interakcja technik, a nie błąd implementacji.

## Częste błędne przekonania

❌ Niska strata treningowa oznacza dobrą generalizację.

✅ Niska strata treningowa oznacza jedynie dobre dopasowanie do **konkretnej próbki**, łącznie z jej szumem. W eksperymencie model o najniższej stracie treningowej miał jedną z wyższych strat walidacyjnych.

❌ Wysoka strata walidacyjna zawsze oznacza overfitting.

✅ Wysoka strata walidacyjna może także wynikać z underfittingu, któremu towarzyszy wysoka strata treningowa. O diagnozie decyduje **relacja** między obiema wartościami. Dwie wysokie i zbliżone straty wskazują underfitting, a rosnąca różnica przy niskiej stracie treningowej wskazuje overfitting.

❌ Dropout trwale usuwa neurony z architektury.

✅ Dropout losuje **nową** maskę zer przy każdym przejściu w przód w trybie treningowym. Żaden neuron nie jest usuwany na stałe. W kolejnym kroku może znów być aktywny, a w trybie `eval()` wszystkie neurony pozostają włączone.

❌ `model.eval()` wyłącza automatycznie liczenie gradientów.

✅ `model.eval()` zmienia zachowanie wybranych warstw, w tym dropout. Gradientami steruje osobny mechanizm oparty na `requires_grad` i `torch.no_grad()`. Notebook potwierdza, że w trybie `eval()` nadal można obliczyć `.grad`.

❌ Early stopping to po prostu wybranie ostatniej epoki treningu.

✅ Early stopping zatrzymuje trening po braku poprawy przez liczbę epok określoną przez `patience` i **przywraca wagi z najlepszej wcześniejszej epoki**. Stan z ostatniej epoki jest gorszy albo co najwyżej równy najlepszemu zapisanemu stanowi.

❌ Zbiór testowy można wykorzystać do wyboru hiperparametrów albo liczby epok, o ile finalny wynik i tak się raportuje.

✅ Użycie testu do wyboru modelu, $\lambda$ lub liczby epok włącza go do procesu doboru. Przestaje wtedy być niezależną miarą generalizacji.

❌ Silniejsza regularyzacja zawsze daje lepszy wynik.

✅ Przy $p=0{,}5$ dropout pogorszył wynik względem $p=0{,}1$, a strata treningowa przestała spadać. Siła regularyzacji ma optimum, więc większa wartość nie musi oznaczać lepszego wyniku.

## Implikacje inżynierskie

- **Koszt dodatkowych eksperymentów.** Porównanie kilku wartości $\lambda$, $p$ lub `patience` zwielokrotnia koszt pojedynczego treningu. W praktyce strojenie bywa bardziej czasochłonne niż końcowy trening.
- **Przechowywanie checkpointów.** Poprawny early stopping wymaga zachowania pełnego stanu najlepszego modelu w pamięci lub na dysku. Dla dużych modeli jest to istotny koszt. Trzeba także zdecydować o częstotliwości nadpisywania i liczbie przechowywanych checkpointów.
- **Trening i inferencja.** Dropout oraz batch normalization zachowują się inaczej w `train()` i `eval()`. Brak zmiany trybu przed ewaluacją lub wdrożeniem może pogorszyć wynik bez jawnego błędu programu.
- **Ryzyko przecieku danych w pipeline.** Normalizacja, selekcja cech lub augmentacja dopasowana do statystyk spoza zbioru treningowego prowadzi do nieuczciwie optymistycznych wyników. Taki błąd jest trudny do wykrycia, ponieważ pipeline nadal działa.
- **Niestabilność oceny przy małych zbiorach.** Różnica między $1{,}1\times$ a $8\times$ przy tej samej konfiguracji i różnych próbkach nie jest artefaktem eksperymentu. Małe zbiory powodują rzeczywistą zmienność, którą trzeba uwzględniać przy interpretacji pojedynczego wyniku.
- **Przesunięcie rozkładu danych** (*distribution shift*). Przedstawiona matematyka zakłada, że dane produkcyjne pochodzą z tego samego rozkładu $P$ co dane treningowe i walidacyjne. Zmiany w czasie, nowi użytkownicy lub inny kontekst mogą naruszyć to założenie. Nawet dobrze regularyzowany model nie musi wtedy zachować jakości. Jest to osobny problem od overfittingu.

## Co warto zapamiętać

- Ryzyko empiryczne stanowi przybliżenie ryzyka oczekiwanego. Rosnąca luka między nimi wskazuje overfitting.
- Underfitting daje wysokie i zbliżone straty treningową oraz walidacyjną. Przy overfittingu niska strata treningowa oddala się od rosnącej straty walidacyjnej.
- L2 i weight decay są równoważne dla SGD, ale **nie** dla Adama. AdamW odsprzęga karę od adaptacyjnej normalizacji.
- Inverted dropout skaluje aktywacje przez $1/(1-p)$ podczas treningu, dzięki czemu `eval()` używa zwykłego przejścia w przód.
- `train()`/`eval()` i liczenie gradientów to dwa niezależne mechanizmy.
- Early stopping bez przywrócenia najlepszego stanu nie daje żadnej korzyści.
- Zbiór testowy służy do jednokrotnej oceny po podjęciu wszystkich decyzji.
- Żadna metoda regularyzacji nie gwarantuje poprawy. Każdą należy sprawdzić dla konkretnego problemu i więcej niż jednego ziarna losowości.

## Pytania dla studentów

1. Czy na podstawie niskiej straty treningowej i wysokiej walidacyjnej można odróżnić overfitting od błędu w obliczaniu walidacji, na przykład danych pochodzących z innego rozkładu? Jakich dodatkowych informacji potrzeba?
2. Dlaczego kara L2 w Adamie (`weight_decay`) może nie zadziałać mimo obecności w gradiencie? Jak wiąże się to z adaptacyjną normalizacją z wykładu 4?
3. Jaki błąd jest najbardziej prawdopodobny, gdy implementacja early stopping nie zmienia zachowania modelu po treningu?
4. Dlaczego nie można wybrać najlepszego $\lambda$ (weight decay) na podstawie wyniku na zbiorze testowym, nawet jeśli finalnie i tak podajemy wynik testowy w raporcie?

## Bibliografia

- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019 (arXiv:1711.05101).
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(56), 1929–1958.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/, rozdział 5.2 (Capacity, Overfitting and Underfitting) i rozdział 7 (Regularization for Deep Learning).
- PyTorch Contributors. Dokumentacja *torch.nn.Dropout*. https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- PyTorch Contributors. Dokumentacja *torch.optim.AdamW*. https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

Pełne wpisy bibliograficzne znajdują się w [`references/references.bib`](../../references/references.bib).
