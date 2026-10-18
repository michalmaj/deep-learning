# Wykład 05: Generalizacja, overfitting i regularyzacja

## Cele wykładu

Po tym wykładzie powinieneś/powinnaś umieć:

1. rozróżnić ryzyko empiryczne (to, co faktycznie minimalizujemy) od ryzyka oczekiwanego (to, na czym naprawdę nam zależy), z jednoznacznie zdefiniowanymi symbolami,
2. rozpoznać na krzywych straty treningowej i walidacyjnej underfitting, dobre dopasowanie i overfitting — i uzasadnić tę diagnozę konkretnymi liczbami, nie wrażeniem,
3. wyprowadzić regułę aktualizacji dla regresji L2 w zwykłym SGD i wskazać dokładny warunek, w którym jest ona tożsama z "weight decay",
4. wyjaśnić, dlaczego ta tożsamość **nie** musi zachodzić dla Adama, i czym różni się od niej AdamW,
5. zapisać matematycznie inverted dropout (maska Bernoulliego, skalowanie) i wyjaśnić różnicę między `model.train()` a `model.eval()`,
6. zaimplementować poprawny early stopping — taki, który **przywraca** najlepszy zapamiętany stan modelu, nie tylko o nim pamięta.

## Dlaczego to jest ważne

Od wykładu 1 minimalizowaliśmy stratę na zbiorze treningowym, milcząco zakładając, że to dobry cel. Ten wykład podważa to założenie wprost: model, który idealnie dopasowuje się do danych treningowych, może być **gorszym** modelem niż taki, który dopasowuje się gorzej. To jest moment w kursie, w którym "optymalizacja" (wykład 4: jak szybko i stabilnie zejść nisko ze stratą treningową) i "uczenie maszynowe" (czy to, czego się nauczyliśmy, działa na nowych danych) formalnie się rozchodzą — i w którym uczysz się na to patrzeć uczciwie, metodologicznie, a nie na podstawie jednego korzystnego przebiegu.

## Intuicja

### Czego naprawdę chcemy

Nie chcemy modelu, który dobrze działa na danych, które już widział — chcemy modelu, który dobrze zadziała na danych, których **jeszcze nie widział**, a które pochodzą z tego samego "świata" (tego samego rozkładu prawdopodobieństwa). Strata na zbiorze treningowym to tylko **przybliżenie** tego, na czym nam zależy, policzone na skończonej próbce. Im lepiej to przybliżenie odpowiada rzeczywistości, tym lepiej **generalizuje** model.

### Underfitting i overfitting

- **Underfitting** (niedouczenie): model jest zbyt prosty (albo zbyt krótko trenowany), żeby uchwycić nawet strukturę obecną w danych treningowych. Objaw: **wysoka strata zarówno na treningu, jak i na walidacji**, zbliżone do siebie.
- **Overfitting** (przeuczenie): model ma wystarczającą (albo nadmiarową) pojemność, żeby dopasować się nie tylko do prawdziwej struktury danych, ale też do szumu konkretnej próbki treningowej. Objaw: **strata treningowa dalej spada, a strata walidacyjna przestaje spadać i zaczyna rosnąć** — model "uczy się na pamięć" przykładów treningowych, łącznie z ich przypadkowym szumem, kosztem ogólnej struktury.

Kluczowe zastrzeżenie, do którego wrócimy empirycznie: **zależność między pojemnością modelu a jakością generalizacji nie jest monotoniczna** i nie zawsze da się przewidzieć bez eksperymentu. Większy model **może** bardziej się przeuczyć — ale nie musi; małe modele też się przeuczają przy wystarczająco małej liczbie danych, a duże modele przy wystarczającej ilości danych i odpowiedniej regularyzacji potrafią generalizować bardzo dobrze. Nie traktuj "mały = bezpieczny, duży = ryzykowny" jako prawa natury.

### Dlaczego regularyzacja

Regularyzacja to rodzina technik, które **nie** zmieniają tego, czego model się uczy się bezpośrednio z sygnału danych, tylko zniechęcają go do zachowań typowych dla przeuczenia: zbyt dużych wag (L2/weight decay), nadmiernego polegania na konkretnych, pojedynczych neuronach (dropout), albo zbyt długiego treningu (early stopping). Żadna z nich nie jest gwarancją — każda to **kompromis**, który trzeba zweryfikować eksperymentalnie na konkretnym problemie, nie założyć z góry.

## Formalny model

Niech $P$ będzie (nieznanym) rozkładem prawdopodobieństwa, z którego pochodzą pary $(x,y)$. Definiujemy:

$$R(\theta) = \mathbb{E}_{(x,y)\sim P}\big[\ell(f(x;\theta), y)\big] \qquad \text{(ryzyko oczekiwane / } \textit{expected risk}\text{)}$$

To jest **to, na czym naprawdę nam zależy** — średni błąd modelu na *całym* rozkładzie danych, nie tylko na tych, które mamy. Nie możemy go policzyć wprost (nie znamy $P$, mamy tylko skończoną próbkę).

$$\hat R(\theta) = \frac{1}{N}\sum_{i=1}^N \ell\big(f(x_i;\theta), y_i\big) \qquad \text{(ryzyko empiryczne /}\textit{ empirical risk}\text{)}$$

To jest dokładnie $L(\theta)$ z wykładu 1 — średnia strata na **naszej konkretnej, skończonej próbce** $\{(x_i,y_i)\}_{i=1}^N$. Trening minimalizuje $\hat R(\theta)$ jako **przybliżenie** $R(\theta)$ — to przybliżenie jest tym lepsze, im więcej reprezentatywnych danych mamy, i tym gorsze, im bardziej model "wykorzystuje" to, że $\hat R$ i $R$ to nie ta sama funkcja.

**Luka generalizacyjna** (*generalization gap*): $R(\theta) - \hat R(\theta)$ — w praktyce przybliżana przez $L_{\text{val}}(\theta) - L_{\text{train}}(\theta)$ (strata na zbiorze walidacyjnym, który model nie widział podczas treningu, minus strata treningowa). Rosnąca luka przy malejącej stracie treningowej to operacyjna definicja overfittingu, której używamy w tym wykładzie.

**Podział danych i jego role** (przypomnienie i uściślenie): **trening** — dopasowanie parametrów; **walidacja** — porównywanie modeli/hiperparametrów i (w tym wykładzie) decyzja, kiedy zatrzymać trening; **test** — **jednokrotna**, końcowa ocena już wybranego modelu. Użycie testu do podejmowania jakiejkolwiek decyzji (który model, który hiperparametr, która epoka) czyni go częścią procesu doboru — i unieważnia go jako miarę generalizacji, z tego samego powodu, dla którego strata treningowa nie jest miarą generalizacji.

**Data leakage** (przeciek danych): każda sytuacja, w której informacja ze zbioru walidacyjnego/testowego (albo z przyszłości, w danych czasowych) wpływa na trening lub dobór modelu — np. policzenie średniej/odchylenia do normalizacji na **całym** zbiorze przed podziałem, zamiast tylko na treningowym. Efekt: zawyżona, nieuczciwa ocena generalizacji — model "widział" (pośrednio, przez statystyki) dane, na których go oceniamy. W laboratorium zmierzysz to, nie tylko przeczytasz o tym.

## Matematyka

### Regularyzacja L2 i jej wpływ na gradient (SGD)

Rozszerzamy funkcję celu o karę za duże wagi:

$$L_{\text{reg}}(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|_2^2 = L(\theta) + \frac{\lambda}{2}\sum_j \theta_j^2$$

gdzie $\lambda \geq 0$ to siła regularyzacji (hiperparametr). Gradient:

$$\nabla L_{\text{reg}}(\theta) = \nabla L(\theta) + \lambda\theta$$

Krok zwykłego SGD z tym gradientem:

$$\theta \leftarrow \theta - \eta\big(\nabla L(\theta) + \lambda\theta\big) = \underbrace{(1-\eta\lambda)\theta}_{\text{"skurczenie" wag}} - \eta\nabla L(\theta)$$

To jest dokładnie to, co nazywamy **weight decay**: każdy krok najpierw mnoży wagi przez $(1-\eta\lambda) < 1$ ("kurczy" je w stronę zera), a potem wykonuje zwykły krok gradientowy. **Dla zwykłego SGD te dwa sformułowania — "dodaj $\frac{\lambda}{2}\|\theta\|^2$ do straty" i "pomnóż wagi przez $(1-\eta\lambda)$ w każdym kroku" — są dokładnie tym samym wzorem**, nie dwoma podobnymi technikami (zweryfikowane w granicach tolerancji numerycznej w notebooku, `atol=1e-3`, przeciw `torch.optim.SGD(weight_decay=λ)`). Zwróć uwagę, że to musi być **ten sam, pojedynczy krok**: osobne "najpierw pełny krok gradientowy $\theta'=\theta-\eta\nabla L(\theta)$, a potem oddzielne skurczenie $(1-\eta\lambda)\theta'$" daje
$$(1-\eta\lambda)\theta'= (1-\eta\lambda)\theta - \eta(1-\eta\lambda)\nabla L(\theta),$$
czyli różni się od wzoru wyżej dodatkowym czynnikiem $(1-\eta\lambda)$ przy składniku gradientowym — różnica to dokładnie $\eta^2\lambda\nabla L(\theta)$, więc dla $\lambda\neq0$ to ogólnie **inny** wynik niż jeden, połączony krok, a to, czy większy czy mniejszy, zależy od znaku $\nabla L(\theta)$ (nie jest to uniwersalnie "mniejszy wynik" w żadną stronę).

### L2 a weight decay dla optymalizatorów adaptacyjnych — UWAGA, to nie to samo

Dla Adama **z parametrem `weight_decay`** (tak jak go domyślnie implementuje `torch.optim.Adam`) kara L2 wchodzi do gradientu **przed** adaptacyjną normalizacją:

$$\hat g_t = \nabla \ell(\theta_t) + \lambda\theta_t, \qquad m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1}+(1-\beta_2)\hat g_t^2, \qquad \dots$$

(reszta dokładnie jak w wykładzie 4). Problem: $\hat g_t$ przechodzi dalej przez dzielenie przez $\sqrt{\hat v_t}$ — **ten sam mechanizm, który normalizuje skalę gradientu zadania, normalizuje też wkład kary L2**. Konsekwencja, zweryfikowana w notebooku: jeśli gradient zadania jest już w przybliżeniu proporcjonalny do $\theta$ (co zdarza się czasem, i co dokładnie pokazuje przykład rozwiązany niżej), dodanie $\lambda\theta$ do niego **nie zmienia w ogóle** znormalizowanego kierunku/kroku Adama — kara L2 w tej konwencji może być w praktyce **bezskuteczna**, mimo niezerowego $\lambda$.

**AdamW** (Loshchilov i Hutter, 2019) rozwiązuje to, **odsprzęgając** (*decoupling*) karę od adaptacyjnej normalizacji — moment $m_t, v_t$ liczymy z **czystego** gradientu zadania, a skurczenie wag stosujemy osobno, już po normalizacji:

$$m_t, v_t \ \text{liczone z } \nabla\ell(\theta_t) \text{ (bez } \lambda\theta_t\text{)}, \qquad \theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\theta_t\right)$$

Teraz $\lambda\theta_t$ działa dokładnie tak samo, niezależnie od historii/skali gradientu zadania — przewidywalnie, tak jak w SGD. To jest powód, dla którego w praktyce (i w eksperymencie tego wykładu) do regularyzacji wag przy Adamie używa się `AdamW`, nie `Adam(weight_decay=...)`.

### Dropout: maska Bernoulliego i *inverted dropout*

Dla warstwy o aktywacjach $a\in\mathbb R^m$ i prawdopodobieństwie odrzucenia $p\in(0,1)$, w trybie **treningowym**:

$$r_i \sim \text{Bernoulli}(1-p) \ \text{niezależnie dla każdego } i, \qquad \tilde a_i = \frac{r_i}{1-p}\, a_i$$

$r_i=0$ z prawdopodobieństwem $p$ (neuron "wyłączony" w tym kroku — wyzerowany, **nie** usunięty z architektury: w następnym kroku może znów być aktywny, maska losowana jest na nowo za każdym forward pass), $r_i=1$ z prawdopodobieństwem $1-p$. Skalowanie przez $\frac{1}{1-p}$ (*inverted dropout* — ta wersja, którą implementuje `nn.Dropout`) zapewnia $\mathbb E[\tilde a_i] = a_i$: oczekiwana wartość wyjścia jest zachowana, mimo losowego zerowania (zweryfikowane w notebooku na 10 000 próbkach).

W trybie **ewaluacji** (`model.eval()`): $\tilde a_i = a_i$ — żadnego losowania, żadnego skalowania. Dzięki temu, że skalowanie wykonano już podczas treningu (stąd *inverted*), inferencja jest zwykłym, deterministycznym przejściem w przód — nic dodatkowego nie trzeba robić.

`model.train()` / `model.eval()` to przełącznik **zachowania konkretnych warstw** (tu: dropout), **nie** przełącznik liczenia gradientów — to jest osobny mechanizm (`torch.no_grad()` / `requires_grad`, wykład 3). `model.eval()` nie wyłącza automatycznie gradientów (zweryfikowane w notebooku: `.grad` jest liczone normalnie w trybie `eval()`).

### Early stopping jako procedura

Po każdej epoce $e$ ze stratą walidacyjną $V_e$:

1. jeśli $V_e < V_{\text{best}} - \delta$ (minimalna istotna poprawa $\delta$, *min\_delta*): zapamiętaj $V_{\text{best}}\leftarrow V_e$, zapamiętaj **pełny stan parametrów** modelu (nie tylko numer epoki!), wyzeruj licznik epok bez poprawy;
2. w przeciwnym razie: zwiększ licznik epok bez poprawy; jeśli licznik osiągnie `patience`: **zatrzymaj trening**;
3. po zatrzymaniu (albo po wyczerpaniu budżetu epok): **przywróć** zapamiętany najlepszy stan parametrów — model, którego dalej używasz, pochodzi z epoki $e_{\text{best}}$, nie z ostatniej wytrenowanej epoki.

Krok 3 jest tym, który najczęściej się pomija (stąd błędne przekonanie w dalszej sekcji) — bez niego "early stopping" to tylko wcześniejsze zatrzymanie pętli, bez żadnej korzyści, bo model i tak zostaje w (gorszym) stanie z ostatniej epoki.

## Przykład rozwiązany

$\theta_0=2$, $\eta=0{,}1$, $\lambda=0{,}5$ (duże, dla czytelności), zadanie $\ell(\theta)=\frac12\theta^2$ (więc $\nabla\ell(\theta)=\theta$) — ten sam wzór na gradient zadania, co w module 1/4.

**SGD, bez regularyzacji:** $\theta_1 = 2 - 0{,}1\cdot 2 = 1{,}8$.

**SGD z L2/weight decay:** $\nabla L_{\text{reg}} = 2 + 0{,}5\cdot2 = 3{,}0$; $\theta_1 = 2 - 0{,}1\cdot3{,}0 = 1{,}7$. (Równoważnie: $(1-\eta\lambda)\theta_0 - \eta\nabla\ell(\theta_0) = 0{,}95\cdot2 - 0{,}2 = 1{,}7$ — ta sama liczba, drugim wzorem).

**Adam, `weight_decay=0,5` (konwencja L2-w-gradiencie):**

$$\hat g_1 = \theta_0 + \lambda\theta_0 = 2+1=3{,}0 \quad\Rightarrow\quad m_1=0{,}1\cdot3{,}0=0{,}3,\ v_1=0{,}001\cdot9{,}0=0{,}009$$
$$\hat m_1 = 0{,}3/0{,}1=3{,}0,\quad \hat v_1=0{,}009/0{,}001=9{,}0 \quad\Rightarrow\quad \theta_1 = 2 - 0{,}1\cdot\frac{3{,}0}{\sqrt{9{,}0}} = 2-0{,}1 = 1{,}9$$

**Zaskakujące, ale zweryfikowane: dla tego samego $\theta_0$ Adam *bez* regularyzacji ($\lambda=0$) daje dokładnie to samo $\theta_1=1{,}9$** — bo $\hat g_1=\theta_0$ (bez $\lambda$), $\hat m_1=2{,}0$, $\hat v_1=4{,}0$, $\theta_1=2-0{,}1\cdot\frac{2{,}0}{\sqrt{4{,}0}}=2-0{,}1=1{,}9$. Gdy gradient zadania jest (jak tu) proporcjonalny do $\theta$, dodanie $\lambda\theta$ tylko przeskalowuje go — a znormalizowany kierunek Adama ($\hat m/\sqrt{\hat v}$) jest niewrażliwy na jednolite przeskalowanie gradientu. **Kara L2 w tej konwencji, dla tego kroku, nie zrobiła dosłownie nic.**

**AdamW, `weight_decay=0,5` (konwencja odsprzężona):** moment liczony z czystego $\nabla\ell(\theta_0)=2{,}0$ (bez $\lambda$): $\hat m_1=2{,}0$, $\hat v_1=4{,}0$, znormalizowany krok $=\hat m_1/\sqrt{\hat v_1}=1{,}0$.

$$\theta_1 = \theta_0 - \eta\left(1{,}0 + \lambda\theta_0\right) = 2 - 0{,}1\cdot(1{,}0+0{,}5\cdot2) = 2-0{,}1\cdot2{,}0 = 1{,}8$$

AdamW **widzi** regularyzację ($1{,}8 \neq$ AdamW bez regularyzacji, które dałoby $\theta_1=1{,}9$ — ten sam rachunek co Adam bez $\lambda$, bo bez kary oba wzory się pokrywają) — Adam z `weight_decay` w tym kroku jej nie widzi wcale. Wszystkie cztery liczby ($1{,}8$; $1{,}7$; $1{,}9$; $1{,}8$) zweryfikowane w notebooku przeciw odpowiednim klasom `torch.optim`.

## Eksperyment

Notebook [`examples.ipynb`](examples.ipynb):

1. **Trzy pojemności, ten sam problem** — regresja $y=\sin(3x)+0{,}3x+\varepsilon$, $N=30$ zaszumionych punktów treningowych, gęsty zbiór walidacyjny ($N=200$) z tego samego rozkładu. Mały MLP (underfitting), średni MLP (dobre dopasowanie), duży MLP `[128,128,128]` (overfitting) — identyczne dane, identyczna procedura treningu.
2. **Regularyzacja na modelu z (1), który się przeuczał** — bez regularyzacji, z `AdamW(weight_decay=...)`, z `nn.Dropout`, z early stopping — oraz przykład zbyt silnego dropout prowadzącego do underfittingu.
3. **(Opcjonalnie, jeśli starczy czasu) Wpływ liczby przykładów treningowych** — ten sam model, $N\in\{15,30,60,120,300\}$.

Każda z sekcji 1–2 jest **zweryfikowana przy kilku ziarnach losowości** przed napisaniem wniosków niżej — nie tylko na jednym, wybranym przebiegu.

## Co obserwujemy

- **Underfitting**: mały model ma wysoką stratę treningową **i** walidacyjną, blisko siebie (np. $\approx 0{,}5$ i $\approx 0{,}6$) — model nie ma czym dopasować się nawet do danych treningowych.
- **Dobre dopasowanie**: średni model ma niską stratę treningową, umiarkowaną walidacyjną, i strata walidacyjna jest wciąż blisko swojego minimum pod koniec treningu — nie ma wyraźnego odbicia w górę.
- **Overfitting, zmierzony, nie założony**: duży model ($33\,409$ parametrów) osiąga najniższą stratę treningową ($\approx 0{,}01$–$0{,}05$) ze wszystkich trzech — ale strata walidacyjna **najpierw spada, osiąga minimum wyraźnie wcześniej niż koniec treningu, a potem rośnie**, mimo że strata treningowa dalej maleje. Dokładna skala tego odbicia zależy od ziarna losowości (sprawdzone: w naszych testach mnożnik strata-końcowa/strata-minimalna wahał się między $\approx 1{,}1\times$ a ponad $8\times$ w zależności od konkretnej próbki danych) — **kierunek efektu jest stabilny, jego dokładna wielkość nie jest**, i tak to przedstawiamy.
- **Weight decay (AdamW) pomaga, umiarkowanie**: w naszych testach, przy dobrze dobranym $\lambda$ (rząd $10^{-2}$), strata walidacyjna końcowa spadała wyraźnie względem modelu bez regularyzacji, choć wielkość poprawy zależała od ziarna.
- **Dropout pomógł mocniej i bardziej konsekwentnie niż weight decay w naszych testach** (przy $p\approx0{,}1$, strata walidacyjna końcowa spadała do poziomu zbliżonego do, a czasem poniżej, najlepszego momentu modelu bez regularyzacji) — to konkretna obserwacja *dla tego problemu i tej architektury*, nie ogólne prawo "dropout > weight decay".
- **Zbyt silny dropout szkodzi**: przy $p=0{,}5$ strata treningowa rosła wyraźnie (model przestał być w stanie dopasować się nawet do danych treningowych) — regularyzacja zamieniła overfitting w underfitting.
- **Early stopping działa, o ile faktycznie przywraca stan** — zatrzymanie samo w sobie nic nie daje (strata w momencie zatrzymania bywała wyraźnie gorsza niż strata w najlepszej zapamiętanej epoce); dopiero przywrócenie zapamiętanych wag daje korzyść.
- **Interakcja, której nie zakładaliśmy z góry**: połączenie dropout + early stopping z tym samym `patience`, co dla modelu bez dropout, zatrzymało trening **przedwcześnie** — dropout wprowadza dodatkowy szum do krzywej walidacyjnej z epoki na epokę, więc ten sam próg cierpliwości bywa zbyt niski. To nie jest błąd implementacji — to rzeczywista interakcja dwóch technik, którą trzeba uwzględnić w praktyce.

## Częste błędne przekonania

❌ Niska strata treningowa oznacza dobrą generalizację.

✅ Niska strata treningowa oznacza tylko, że model dobrze dopasował się do **tej konkretnej próbki** — łącznie z jej szumem. W naszym eksperymencie model o najniższej stracie treningowej miał jedną z gorszych strat walidacyjnych.

❌ Wysoka strata walidacyjna zawsze oznacza overfitting.

✅ Wysoka strata walidacyjna może też oznaczać underfitting (model zbyt prosty, strata treningowa też wysoka) — rozróżnia je właśnie **relacja** między stratą treningową a walidacyjną (bliskie siebie i obie wysokie = underfitting; rozjeżdżające się, treningowa niska = overfitting), nie sama wartość straty walidacyjnej w oderwaniu od treningowej.

❌ Dropout trwale usuwa neurony z architektury.

✅ Dropout losuje **nową** maskę zer przy każdym przejściu w przód w trybie treningowym — żaden neuron nie jest usuwany na stałe; w kolejnym kroku (i zawsze w trybie `eval()`) jest z powrotem w pełni aktywny.

❌ `model.eval()` wyłącza automatycznie liczenie gradientów.

✅ `model.eval()` zmienia tylko zachowanie konkretnych warstw (dropout, i inne niepoznane jeszcze w tym kursie) — liczenie gradientów kontroluje osobny mechanizm (`requires_grad`, `torch.no_grad()`, wykład 3), niezależny od trybu `train`/`eval` (zweryfikowane: w `eval()` `.grad` dalej się liczy).

❌ Early stopping to po prostu wybranie ostatniej epoki treningu.

✅ Early stopping to zatrzymanie treningu **po** wykryciu braku poprawy przez `patience` epok, **i przywrócenie wag z najlepszej, wcześniejszej epoki** — ostatnia wytrenowana epoka jest z definicji gorsza (albo co najwyżej taka sama) niż ta zapamiętana jako najlepsza.

❌ Zbiór testowy można wykorzystać do wyboru hiperparametrów albo liczby epok, o ile finalny wynik i tak się raportuje.

✅ Jakiekolwiek użycie testu do podjęcia decyzji (który model, który $\lambda$, która epoka) czyni go częścią procesu doboru modelu — przestaje być niezależną miarą generalizacji z dokładnie tego samego powodu, dla którego strata treningowa nią nie jest.

❌ Silniejsza regularyzacja zawsze daje lepszy wynik.

✅ Zmierzone wprost: przy $p=0{,}5$ dropout pogorszył wynik względem $p=0{,}1$ (i względem samej straty treningowej, która przestała spadać) — regularyzacja ma optimum, nie jest "im więcej, tym lepiej".

## Implikacje inżynierskie

- **Koszt dodatkowych eksperymentów**: porównanie kilku wartości $\lambda$/$p$/`patience` to wielokrotność kosztu pojedynczego treningu — w praktyce to często największy koszt czasowy projektu, większy niż sam finalny trening.
- **Przechowywanie checkpointów**: poprawny early stopping wymaga trzymania w pamięci (albo na dysku) pełnego stanu najlepszego modelu podczas treningu, nie tylko metryki — dla dużych modeli to realny koszt pamięciowy/dyskowy, o którym trzeba zdecydować świadomie (co ile epok nadpisywać checkpoint, czy trzymać więcej niż jeden).
- **Trening vs. inferencja**: dropout (i, poza zakresem tego wykładu, batch normalization) zachowują się **inaczej** w `train()` i `eval()` — zapomnienie przełączenia trybu przed ewaluacją/wdrożeniem to częsty, cichy błąd (model dalej "działa", tylko gorzej i w sposób trudny od razu zdiagnozować).
- **Ryzyko przecieku danych w pipeline'ie**: normalizacja, selekcja cech, augmentacja dopasowana do statystyk z danych spoza zbioru treningowego to jeden z najczęstszych, najtrudniej wykrywalnych błędów w praktycznych projektach ML — bo model "działa" i daje liczby, tylko te liczby są nieuczciwie optymistyczne.
- **Niestabilność oceny przy małych zbiorach**: zmierzona w tym wykładzie wrażliwość wyniku na ziarno losowości (różnica między $1{,}1\times$ a $8\times$ przy tej samej konfiguracji, różnej próbce danych) to nie artefakt naszego eksperymentu — to realna własność małych zbiorów danych, którą trzeba uwzględnić przy interpretacji pojedynczego wyniku w praktyce.
- **Przesunięcie rozkładu danych** (*distribution shift*): cała matematyka generalizacji w tym wykładzie zakłada, że dane produkcyjne pochodzą z tego samego rozkładu $P$, co dane treningowe/walidacyjne. Jeśli to założenie przestaje być prawdziwe (dane się zmieniają w czasie, nowi użytkownicy, inny kontekst), żadna z tych gwarancji — nawet przy idealnej regularyzacji — nie musi się przenosić. To osobny problem od overfittingu, często z nim mylony.

## Co warto zapamiętać

- Minimalizujemy ryzyko empiryczne jako przybliżenie ryzyka oczekiwanego — luka między nimi to overfitting.
- Underfitting: wysoka strata training i walidacyjna, blisko siebie. Overfitting: rozjeżdżające się, treningowa niska.
- L2 i weight decay to ten sam wzór dla SGD — ale **nie** dla Adama; AdamW naprawia to przez odsprzężenie kary od adaptacyjnej normalizacji.
- Inverted dropout skaluje aktywacje przez $1/(1-p)$ podczas treningu, żeby `eval()` mógł być zwykłym przejściem bez zmian.
- `train()`/`eval()` i liczenie gradientów to dwa niezależne mechanizmy.
- Early stopping bez przywrócenia najlepszego stanu nie daje żadnej korzyści.
- Zbiór testowy dotykamy raz, po wszystkich decyzjach — inaczej przestaje mierzyć to, co ma mierzyć.
- Żadna metoda regularyzacji nie jest gwarancją — każdą trzeba zweryfikować na konkretnym problemie, przy więcej niż jednym ziarnie losowości.

## Pytania dla studentów

1. Masz model z niską stratą treningową i wysoką walidacyjną. Jak odróżnisz, na podstawie samych tych dwóch liczb, czy to overfitting, czy np. błąd w kodzie liczącym stratę walidacyjną (np. dane walidacyjne są z innego rozkładu)?
2. Dlaczego kara L2 w Adamie (`weight_decay`) może nie zadziałać, mimo że matematycznie "jest obecna" w gradiencie? Odpowiedz, odwołując się do mechanizmu adaptacyjnej normalizacji z wykładu 4.
3. Zaimplementowałeś/aś early stopping, ale model po treningu zachowuje się dokładnie tak samo, jakby go nie było. Jaki jest najbardziej prawdopodobny błąd?
4. Dlaczego nie można wybrać najlepszego $\lambda$ (weight decay) na podstawie wyniku na zbiorze testowym, nawet jeśli finalnie i tak podajemy wynik testowy w raporcie?

## Bibliografia

- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019 (arXiv:1711.05101).
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(56), 1929–1958.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/ — rozdział 5.2 (Capacity, Overfitting and Underfitting) i rozdział 7 (Regularization for Deep Learning).
- PyTorch Contributors. *torch.nn.Dropout* — dokumentacja. https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- PyTorch Contributors. *torch.optim.AdamW* — dokumentacja. https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

Pełne wpisy bibliograficzne: [`references/references.bib`](../../references/references.bib).
