# Wykład 07 — Transfer learning: wykorzystanie wcześniej wyuczonych reprezentacji

## Cele wykładu

1. Zrozumieć, dlaczego reprezentacje wyuczone przez CNN na dużym zbiorze danych
   (ImageNet) bywają przydatne w zupełnie innym zadaniu klasyfikacji obrazów.
2. Rozróżnić trzy strategie adaptacji modelu pretrained: zamrożony backbone (*linear
   probing*), częściowy fine-tuning, pełny fine-tuning — i wiedzieć, kiedy która ma sens.
3. Poprawnie manipulować `requires_grad`, grupami parametrów optymalizatora i trybem
   `train()`/`eval()` — i rozumieć, że to są **trzy różne, niezależne** mechanizmy.
4. Rozumieć szczególne ryzyko BatchNorm podczas zamrażania i fine-tuningu.
5. Nie traktować transfer learningu jako uniwersalnego sposobu na wyższą accuracy —
   umieć ocenić, kiedy faktycznie pomaga, a kiedy nie.

## Dlaczego to jest ważne

Trenowanie CNN od zera (wykład 6) wymaga dużo danych i czasu — ImageNet ma 1,2 mln
obrazów i 1000 klas; model taki jak ResNet18 trenowany od zera dochodzi do dobrej
jakości po dniach treningu na GPU. W praktyce inżynierskiej rzadko mamy tyle danych do
własnego, wąskiego zadania (np. 600 zdjęć 10 kategorii obiektów). Transfer learning to
odpowiedź na to ograniczenie: zamiast trenować od zera, **zaczynamy od wag
wytrenowanych na dużym, ogólnym zbiorze** i dostosowujemy je (albo tylko ich fragment)
do nowego zadania. To jeden z najczęściej stosowanych wzorców w praktycznym computer
vision — i jeden z łatwiejszych do zepsucia przez subtelne błędy implementacyjne, co
zobaczymy w sekcji o BatchNorm.

## Intuicja

CNN trenowana na ImageNet (wykład 6: warstwy konwolucyjne, pooling, rosnące pole
recepcyjne) uczy się hierarchii cech: wczesne warstwy wykrywają krawędzie, kontury i
proste wzorce tekstur — cechy **ogólne**, przydatne praktycznie dla każdego zadania
związanego z obrazami. Głębsze warstwy wykrywają coraz bardziej złożone, coraz bardziej
**specyficzne dla zbioru treningowego** wzorce (kształty charakterystyczne dla
konkretnych kategorii ImageNet). Intuicja transfer learningu: wczesne/ogólne cechy
można **ponownie wykorzystać** bez zmian; to, co ewentualnie trzeba dostosować, to
głębsze warstwy i (zawsze) finalna głowica klasyfikacyjna, bo liczba i tożsamość klas w
nowym zadaniu różni się od ImageNet.

❌ "Pretrained model zawsze poprawi wynik, bo 'wie więcej' niż model trenowany od zera."

✅ Pretrained model pomaga, gdy cechy wyuczone na zbiorze źródłowym (ImageNet: zdjęcia
obiektów z poziomu ziemi, codzienne sceny) są **rzeczywiście użyteczne** dla nowej
domeny. Im nowa domena wizualnie bardziej odległa od ImageNet (zobaczymy to dosłownie w
laboratorium — zdjęcia satelitarne), tym mniejsza gwarancja korzyści. Sprawdzamy to
eksperymentalnie, nie zakładamy z góry.

## Formalny model: backbone i głowica

Traktujemy pretrained model jako złożenie dwóch funkcji:

$$f(x;\theta) = f_{\text{head}}\Big(f_{\text{backbone}}(x;\theta_{\text{backbone}});\ \theta_{\text{head}}\Big), \qquad \theta = (\theta_{\text{backbone}}, \theta_{\text{head}})$$

- $f_{\text{backbone}}$ (**backbone**, *feature extractor*) — część konwolucyjna sieci
  (dla ResNet18: `conv1, bn1, layer1..layer4, avgpool`), mapuje obraz
  $x\in\mathbb R^{3\times224\times224}$ na wektor cech $z = f_{\text{backbone}}(x) \in
  \mathbb R^{512}$.
- $f_{\text{head}}$ (**głowica klasyfikacyjna**, *classification head*) — ostatnia
  warstwa (lub kilka warstw), mapuje wektor cech na $C$ logitów nowego zadania:
  $f_{\text{head}}: \mathbb R^{512}\to\mathbb R^{C}$. W ResNet18 to domyślnie pojedyncza
  `nn.Linear(512, 1000)` (dla oryginalnego ImageNet) — **pierwszy krok transfer
  learningu to zawsze podmiana tej warstwy** na `nn.Linear(512, C)` z losową
  inicjalizacją, bo $C$ (liczba klas Twojego zadania) różni się od 1000.

**Uwaga o notacji.** Można by się spodziewać, że parametry głowicy oznaczymy $\varphi$
(spotykane w literaturze). W tym kursie $\varphi$ jest już zarezerwowane dla **funkcji
aktywacji** (wykłady 2–4) — żeby uniknąć kolizji, zamiast nowego greckiego symbolu
rozszerzamy istniejącą konwencję $\theta$ = parametry modelu (wykład 1) o jawne
indeksy: $\theta_{\text{backbone}}$, $\theta_{\text{head}}$.

## Matematyka: trzy strategie adaptacji

Tak jak zawsze, trenowanie minimalizuje stratę $L(\theta) =
\frac1N\sum_{i=1}^N\ell(f(x_i;\theta), y_i)$ (wykład 1). Różnica między strategiami to
**względem jakiego podzbioru $\theta$ liczymy i stosujemy gradient**.

**1. Zamrożony backbone (linear probing).** $\theta_{\text{backbone}}$ ustawione na
wartość pretrained i **wyłączone** z optymalizacji:

$$\min_{\theta_{\text{head}}} L(\theta_{\text{backbone}}^{\text{pretrained}}, \theta_{\text{head}}), \qquad \theta_{\text{backbone}} \text{ ustalone, nie wchodzi do } \nabla$$

W PyTorch: `requires_grad=False` dla wszystkich parametrów backbone, optymalizator
dostaje tylko `model.fc.parameters()`.

**2. Częściowy fine-tuning.** Dzielimy backbone na część zamrożoną i odmrożoną,
$\theta_{\text{backbone}} = (\theta_{\text{backbone}}^{\text{frozen}},
\theta_{\text{backbone}}^{\text{unfrozen}})$ (np. `layer1..layer3` zamrożone,
`layer4` odmrożone):

$$\min_{\theta_{\text{backbone}}^{\text{unfrozen}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{\text{frozen}} = \theta_{\text{backbone}}^{\text{frozen,pretrained}} \text{ ustalone}$$

**3. Pełny fine-tuning.** Wszystko trenowalne, ale **inicjalizacja** wciąż pretrained
(nie losowa!) dla backbone:

$$\min_{\theta_{\text{backbone}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{(0)} = \theta_{\text{backbone}}^{\text{pretrained}} \text{ (punkt startowy, nie stała)}$$

**Które parametry dostają gradient?** `loss.backward()` (wykład 3) liczy
$\partial L/\partial\theta_j$ dla **każdego** $\theta_j$ z `requires_grad=True` —
niezależnie od strategii, forward pass zawsze przechodzi przez **cały** model (inaczej
nie policzylibyśmy straty). To, co się różni, to które z policzonych pochodnych cząstkowych
trafiają do `.grad` (PyTorch pomija akumulację dla `requires_grad=False`) i które
parametry otrzymuje `optimizer.step()`. Zamrożenie **nie przyspiesza automatycznie
forward passu** — zobaczymy to liczbowo w eksperymencie.

**Różne learning rate dla różnych grup parametrów** — `torch.optim` wspiera to
natywnie przez *parameter groups*:

```python
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": eta_backbone},
    {"params": model.fc.parameters(), "lr": eta_head},
])
```

gdzie zwykle $\eta_{\text{backbone}} \ll \eta_{\text{head}}$ (np. $10$–$100\times$
mniejszy).

**Dlaczego małe kroki na backbone, a nie żadne albo duże?** Z modułu 4: strata to
krajobraz, gradient wskazuje lokalny kierunek spadku. Pretrained $\theta_{\text{backbone}}$
leży w jakimś dobrym rejonie *dla swojego oryginalnego zadania* — dla nowego zadania ten
punkt zwykle nie jest optymalny, ale jest **znacznie lepszym punktem startowym** niż
losowa inicjalizacja (bo wiele cech niskiego/średniego poziomu — krawędzie, tekstury,
proste kształty — jest użytecznych niezależnie od konkretnego zadania). Duży krok, zanim
losowo zainicjalizowana głowica nauczy się cokolwiek sensownego, oznacza propagowanie
dużych, chaotycznych gradientów w głąb backbone'u — ryzyko odejścia od tego dobrego
punktu zanim zdąży się do niego cokolwiek "dostroić" (potocznie: zniszczenie wcześniej
wyuczonych cech, zanim nowa głowica zdąży z nich skorzystać). Stąd typowa kolejność w
praktyce (i w naszym eksperymencie): najpierw zamrożony backbone + trening samej
głowicy, dopiero potem, ostrożnie, częściowe odmrożenie z małym $\eta_{\text{backbone}}$.
Nie powtarzamy tu pełnego wyprowadzenia backpropagation (wykład 3) — korzystamy z niego
jako gotowego mechanizmu.

❌ "Fine-tuning całego modelu zawsze da lepszy wynik niż trening samej głowicy, bo
model ma więcej do powiedzenia."

✅ Pełny fine-tuning ma **więcej stopni swobody do dopasowania się do małego zbioru
danych** — co oznacza też większe ryzyko przeuczenia (moduł 5) przy małej liczbie
przykładów. Sprawdzamy to eksperymentalnie, nie zakładamy.

## Przykład obliczeniowy: ile parametrów faktycznie się trenuje?

ResNet18 z podmienioną głowicą na $C=10$ klas:

| Strategia | Parametry trenowalne | % całości |
|---|---:|---:|
| Cały model (referencja) | 11 181 642 | 100,00% |
| Zamrożony backbone (tylko `fc`) | 5 130 | 0,05% |
| Częściowy fine-tuning (`layer4` + `fc`) | 8 398 858 | 75,11% |
| Pełny fine-tuning | 11 181 642 | 100,00% |

(Liczba 5 130 = $512\times10 + 10$ — wagi i biasy `nn.Linear(512, 10)`, dokładnie wzór z
wykładu o warstwach gęstych z modułu 2. Liczba 8 398 858 to `layer4` ResNet18
[8 393 728 — dwa bloki `BasicBlock`] plus głowica.)

**To nie wystarczy policzyć przez samo odczytanie `requires_grad=True` — trzeba
zweryfikować, że zamrożone parametry faktycznie się nie zmieniają, a trenowalne
faktycznie się zmieniają, po realnym kroku optymalizatora:**

```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
# ... jeden krok: zero_grad -> forward -> backward -> step ...
for n, p in model.named_parameters():
    changed = not torch.equal(before[n], p.detach())
    assert changed == p.requires_grad, f"{n}: niespodziewany stan"
```

Zweryfikowane programowo (notebook powtarza to na realnym treningu): **0 zamrożonych
tensorów zmienia się** przy każdej z trzech strategii, **wszystkie trenowalne tensory
faktycznie się zmieniają** po kilku krokach `optimizer.step()`.

**Zaskakujące odkrycie przy tej weryfikacji — BatchNorm.** ResNet18 ma **20** warstw
`BatchNorm2d`. Gdy model jest w trybie `model.train()`, **wszystkie 20** z nich
aktualizują swoje bufory `running_mean`/`running_var` **niezależnie od
`requires_grad`** — łącznie z warstwami głęboko w "zamrożonym" `layer1`. Dopiero
`model.eval()` to zatrzymuje. Rozwijamy to w sekcji *Błędne przekonania* niżej — to
jeden z najczęstszych, cichych błędów w transfer learningu.

## Eksperyment

Notebook (`examples.ipynb`) porównuje na tym samym zadaniu (Caltech101, 10
zbalansowanych klas z dobrym pokryciem w ImageNet: `airplanes, Motorbikes, elephant,
kangaroo, grand_piano, car_side, dalmatian, butterfly, starfish, dolphin`, 600 obrazów,
podział 360/120/120):

1. **From scratch** — ResNet18 z losową inicjalizacją, pełny trening.
2. **Frozen backbone** (linear probing) — dwa warianty: naiwny (pełny forward przez
   backbone co epokę) i z cache'owaniem wyekstrahowanych cech (ekstrakcja raz,
   trening głowicy na zapisanych wektorach $512$-wymiarowych).
3. **Partial fine-tuning** (`layer4` + `fc`, osobny, niższy $\eta_{\text{backbone}}$).

Dla każdej strategii: parametry całkowite/trenowalne, strata treningowa/walidacyjna,
accuracy walidacyjna, zmierzony czas treningu i inferencji. Porównanie **przy tej samej
liczbie epok** jest jawnie odróżnione od porównania **przy tym samym czasie obliczeń** —
to nie są równoważne warunki.

## Co obserwujemy

Rzeczywiste wyniki z notebooka (10 klas Caltech101, 360 obrazów treningowych):

| Strategia | Param. trenowalne | Accuracy walidacyjna | Czas treningu | Inferencja/obraz |
|---|---:|---:|---:|---:|
| Od zera (8 epok) | 11 181 642 (100%) | 0,65 | 86,4s | 12,5ms |
| Zamrożony backbone, naiwnie (5 epok) | 5 130 (0,05%) | 1,00 | 27,1s | — |
| Zamrożony backbone, z cache'em (5 epok, ten sam wynik) | 5 130 | 1,00 | **7,5s** | 11,7ms |
| Zamrożony backbone, finalnie (15 epok na cache'u) | 5 130 | 1,00 | 7,5s | — |
| Częściowy fine-tuning, `layer4`+`fc` (8 epok) | 8 398 858 (75%) | 1,00 | 51,3s | 11,7ms |

**Transfer learning dał tu ogromną, jednoznaczną przewagę** — zamrożony backbone
osiąga accuracy walidacyjną $1{,}00$ przy $0{,}05\%$ trenowalnych parametrów, podczas
gdy pełny trening od zera na tych samych $360$ obrazach osiąga tylko $0{,}65$. To nie
przypadek jednego przebiegu: powtórzone na $2$ ziarnach, zamrożony backbone (cache)
daje $1{,}00/1{,}00$ accuracy walidacyjnej i $0{,}983/1{,}00$ testowej, częściowy
fine-tuning $1{,}00/1{,}00$ walidacyjnej i $0{,}958/0{,}992$ testowej — stabilnie
blisko sufitu w obu przypadkach.

**Czy częściowy fine-tuning dał coś ponad zamrożony backbone?** W tym konkretnym
eksperymencie — **nie, w granicach błędu pomiaru**: obie strategie osiągają walidacyjną
accuracy $1{,}00$ i bardzo zbliżoną testową ($0{,}98$–$1{,}00$ dla cache, $0{,}96$–$0{,}99$
dla partial). Częściowy fine-tuning trwał $51$s (8 epok) wobec $7{,}5$s dla wersji
finalnej cache'u (15 epok) — **to porównanie przy różnych budżetach epok i różnej
liczbie aktualizacji, nie przy tym samym koszcie obliczeniowym**, więc sama różnica
czasu nie jest miarodajnym wnioskiem o względnej efektywności obliczeniowej tych dwóch
strategii per se (uczciwe porównanie "ten sam budżet epok" jest niżej: naiwne liczenie
vs cache, oba przy 5 epokach). To, co *jest* tu wiarygodne — bo nie zależy od budżetu
epok — to brak mierzalnej korzyści jakościowej: obie strategie osiągają ten sam sufit
accuracy. To **nie** jest dowód, że częściowy fine-tuning jest bezużyteczny w ogóle —
to konsekwencja tego, że wybrane klasy mają bardzo silne pokrycie z ImageNet, więc
zamrożone cechy już niemal idealnie separują problem (efekt sufitu, *ceiling effect*).
W laboratorium (zdjęcia satelitarne EuroSAT, domena wizualnie dalsza od ImageNet)
zmierzono inny wynik — częściowy fine-tuning dał tam mierzalną poprawę. Te dwa
eksperymenty (Caltech101 i EuroSAT) **nie wystarczają, by ustalić ogólną zależność**
"im większe przesunięcie dziedziny, tym większa korzyść z fine-tuningu" — pokazują
tylko, że w tych dwóch konkretnych przypadkach wynik był różny, co samo w sobie obala
twierdzenie "częściowy fine-tuning nigdy/zawsze nie pomaga".

**Cache vs naiwne liczenie, przy identycznej liczbie epok (5) i identycznym wyniku
($1{,}00$ accuracy)**: $27{,}1$s naiwnie vs $7{,}5$s z cache'em — **$3{,}6\times$
szybciej**, bez żadnej różnicy w jakości (bo backbone naprawdę się nie zmienia).

**Wpływ rozmiaru danych** (eksperyment dodatkowy, $6/18/36$ obrazów na klasę):
zamrożony backbone (cache) osiąga $0{,}992 \to 1{,}00 \to 1{,}00$ — niemal od razu
blisko sufitu, nawet przy zaledwie $6$ obrazach na klasę. Trening od zera: $0{,}250 \to
0{,}517 \to 0{,}575$ — rośnie z ilością danych, ale zostaje daleko w tyle nawet przy
pełnych $36$/klasę. Przewaga transfer learningu jest **największa właśnie przy małej
ilości danych** — to zgadza się z intuicją z sekcji *Dlaczego to jest ważne*.

## Częste błędne przekonania

❌ "Zamrożenie parametrów (`requires_grad=False`) to to samo, co `model.eval()`."

✅ To **dwa niezależne mechanizmy**. `requires_grad=False` mówi autogradowi: "nie licz
i nie akumuluj gradientu dla tego tensora" — wpływa na to, co dzieje się **wstecz**
(backward). `model.eval()`/`model.train()` zmienia **zachowanie konkretnych warstw w
przód** (BatchNorm: czy aktualizować bieżące statystyki czy używać zapisanych; Dropout:
czy losowo zerować aktywacje). Można zamrozić wagi i zostawić model w `train()` —
wagi się nie zmienią, ale **statystyki BatchNorm nadal będą dryfować** (zweryfikowane
wyżej: 20/20 warstw). Można też zostawić `requires_grad=True`, a wywołać model pod
`torch.no_grad()` — wtedy żadne gradienty w ogóle nie powstaną, niezależnie od tego, co
mówi `requires_grad`.

❌ "`torch.no_grad()` i `torch.inference_mode()` to to samo co zamrożenie parametrów."

✅ To kontrolują **czy w ogóle budowany jest graf obliczeniowy do propagacji wstecznej**
w danym fragmencie kodu — niezależnie od `requires_grad` poszczególnych parametrów.
Używamy ich przy inferencji (nie potrzebujemy gradientów, oszczędzamy pamięć i czas) i
przy jednorazowej ekstrakcji cech do cache'u (patrz niżej) — nie przy samym treningu
głowicy, bo tam akurat **potrzebujemy** gradientu względem $\theta_{\text{head}}$.
`torch.inference_mode()` to ściślejsza, nowsza wersja `no_grad()` — tensory z niej
wychodzące nie nadają się do dalszego różniczkowania, więc używamy jej tylko tam, gdzie
na pewno nic więcej nie będziemy przez to różniczkować (ekstrakcja cech do cache'u).

❌ "Zamrożony backbone oznacza, że cały model przestał się uczyć."

✅ Backbone przestał się uczyć — głowica **nadal się uczy**, i to ona niesie cały
ciężar dopasowania do nowego zadania. "Zamrożony model" to w praktyce zawsze "zamrożona
**część** modelu" — zawsze jest coś trenowalnego (przynajmniej nowa głowica, inaczej
nie byłoby czego optymalizować).

❌ "Fine-tuning wszystkich warstw musi dać lepszy wynik niż trenowanie samej głowicy."

✅ Sprawdzone eksperymentalnie w notebooku — przy małym zbiorze danych więcej stopni
swobody oznacza też większe ryzyko przeuczenia (moduł 5). Wygrywa ta strategia, która
faktycznie ma najlepszą stratę/accuracy **walidacyjną** w danym przebiegu, nie ta z
największą liczbą trenowalnych parametrów.

❌ "Większy pretrained model to zawsze lepszy wybór."

✅ Większy model (więcej warstw/kanałów) zwykle oznacza wyższą accuracy na ImageNet,
ale też większy koszt pamięci, inferencji i treningu — w ograniczonym środowisku (CPU,
limit czasu) to realny kompromis, nie oczywisty wybór. Zobacz też: mniejszy model nie
musi być szybszy (sekcja *Implikacje inżynierskie*).

❌ "Niższa strata treningowa po fine-tuningu oznacza lepszą generalizację."

✅ Tak samo jak w module 5: strata/accuracy treningowa mierzy dopasowanie do danych,
które model widział. Jedyny wiarygodny sygnał generalizacji to walidacja/test.

❌ "Skoro używam wag pretrained, mogę pominąć porządną walidację i analizę błędów."

✅ Pretrained wagi zmieniają punkt startowy optymalizacji, nie zwalniają z niczego, co
dotyczy oceny wyniku — confusion matrix, analiza błędów i uczciwy podział
train/val/test (moduł 5) obowiązują dokładnie tak samo.

❌ "Reprezentacje wyuczone na ImageNet przenoszą się dobrze na każdą domenę obrazów."

✅ ImageNet to głównie zdjęcia obiektów i scen "z poziomu ziemi". Domeny wizualnie
odległe (zdjęcia satelitarne, obrazy medyczne, mikroskopowe) mogą skorzystać dużo
mniej — to się nazywa **przesunięcie dziedziny** (*domain shift*), a w skrajnym
przypadku transfer może nawet **zaszkodzić** względem dobrze dobranego modelu
trenowanego od zera (*negative transfer*). Zbadamy to wprost w laboratorium (zdjęcia
satelitarne EuroSAT).

## Implikacje inżynierskie

- **Pamięć.** Nawet przy zamrożonym backbone trzeba trzymać w pamięci jego aktywacje
  podczas forward passu (potrzebne do backward przez głowicę) — zamrożenie oszczędza
  pamięć na **gradienty i stan optymalizatora** backbone'u (Adam: dodatkowo $2\times$
  tyle pamięci co same wagi, moduł 4), nie na same aktywacje.
- **Koszt obliczeniowy a liczba parametrów — znów nie to samo.** Zmierzone w tym
  module: MobileNetV3-Small ma $4,6\times$ mniej parametrów i $32\times$ mniej FLOP-ów
  niż ResNet18, a mimo to jest na tym CPU **$4\times$ wolniejszy** w praktyce
  (konwolucje depthwise-separable słabo wykorzystują standardowe jądra CPU bez
  specjalnej optymalizacji). Ten sam wzorzec, co w modułach 5 i 6: nie ufaj liczbie
  parametrów jako proxy dla czasu.
- **Zamrożony backbone nie przyspiesza treningu tak, jak można by się spodziewać** —
  naiwnie wykonany, nadal wymaga pełnego forward passu przez cały model co epokę.
  Przyspieszenie przychodzi z **cache'owania cech** (ekstrakcja raz, trening głowicy na
  gotowych wektorach) — ale to działa **tylko wtedy, gdy backbone się nie zmienia i
  preprocessing jest deterministyczny** (ta sama transformacja za każdym razem, bez
  losowej augmentacji). Przy partial/full fine-tuningu cache jest nieważny już po
  pierwszym kroku optymalizatora, bo backbone (a więc i cechy, które produkuje) się
  zmienia — cache **nie zastępuje** fine-tuningu, tylko przyspiesza bardzo konkretny,
  wąski przypadek (backbone naprawdę zamrożony).
- **Rozmiar pobieranych wag i reprodukowalność.** Wagi pretrained to zewnętrzny plik
  binarny (ResNet18: ~45MB) pobierany z konkretnego źródła z konkretną wersją — warto
  przypiąć ją explicite (`ResNet18_Weights.IMAGENET1K_V1`, nie `DEFAULT`), bo alias
  `DEFAULT` może w przyszłości wskazywać na inne wagi.
- **Kompromis jakość/rozmiar/koszt** jest trójwymiarowy, nie jednowymiarowy — wybór
  strategii i architektury zależy od tego, co faktycznie ogranicza: czas treningu,
  pamięć, czy opóźnienie inferencji w produkcji.

## Co warto zapamiętać

- Transfer learning = backbone + głowica, $\theta = (\theta_{\text{backbone}},
  \theta_{\text{head}})$; trzy strategie różnią się tym, który podzbiór $\theta$ dostaje
  gradient i jest aktualizowany.
- `requires_grad`, `model.train()`/`eval()`, `torch.no_grad()`/`inference_mode()` to
  **trzy niezależne mechanizmy** — każdy odpowiada za co innego.
- Zamrożenie backbone'u przez `requires_grad=False` **nie** zatrzymuje automatycznie
  aktualizacji statystyk BatchNorm w trybie `train()` — potrzebny jest `model.eval()`.
- Transfer learning nie jest gwarancją lepszego wyniku — zależy od tego, jak bardzo
  domena nowego zadania przypomina domenę, na której model był trenowany.

## Pytania dla studentów

1. Masz ResNet18 z zamrożonym backbone (`requires_grad=False` na wszystkim poza `fc`).
   Model jest w trybie `model.train()`. Czy po 5 krokach `optimizer.step()` zmieni się
   `model.layer1[0].conv1.weight`? A `model.layer1[0].bn1.running_mean`? Uzasadnij.
2. Masz 50 obrazów nowej klasy i pretrained ResNet18. Czy wybierzesz zamrożony
   backbone, częściowy czy pełny fine-tuning? Jakie dwa argumenty (nie tylko
   "więcej parametrów = lepiej") uzasadniają Twój wybór?
3. Dlaczego cache wyekstrahowanych cech jest poprawny dla zamrożonego backbone'u, ale
   staje się nieaktualny po pierwszym kroku fine-tuningu?
4. Policz liczbę trenowalnych parametrów dla ResNet18 z głowicą `nn.Linear(512, 37)`
   (37 klas), gdy odmrożone jest `layer3` i `layer4` razem z głowicą. (Wskazówka:
   `layer3` ma 2 099 712 parametrów.)

## Bibliografia

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image
  Recognition*. CVPR. (architektura ResNet)
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A
  Large-Scale Hierarchical Image Database*. CVPR. (zbiór pretrainingu)
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are
  features in deep neural networks?* NeurIPS. (ogólność vs specyficzność cech w głębi
  sieci — źródło intuicji o wczesnych/głębokich warstwach)
- PyTorch documentation: *torchvision.models.resnet18*,
  `https://pytorch.org/vision/stable/models/resnet.html`.
- PyTorch documentation: *Autograd mechanics — Locally disabling gradient computation*,
  `https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation`.
