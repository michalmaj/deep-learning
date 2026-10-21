Język: [English](lecture.en.md) | **Polski**

# Wykład 07. Transfer learning i ponowne wykorzystanie reprezentacji

## Cele wykładu

1. wyjaśnić, dlaczego cechy wyuczone przez CNN na ImageNet mogą być przydatne w innym zadaniu klasyfikacji obrazów,
2. rozróżnić zamrożony backbone, częściowy fine-tuning i pełny fine-tuning,
3. poprawnie stosować `requires_grad`, grupy parametrów optymalizatora oraz tryby `train()` i `eval()`,
4. rozpoznać ryzyko związane z BatchNorm podczas zamrażania modelu,
5. ocenić eksperymentalnie, czy transfer learning pomaga w danym zadaniu.

## Dlaczego to jest ważne

Trenowanie CNN od zera wymaga wielu danych i długich obliczeń. ImageNet zawiera 1,2 mln obrazów należących do 1000 klas, a trening ResNet18 na GPU trwa wiele godzin lub dni. W praktycznych zadaniach często dostępnych jest zaledwie kilkaset przykładów.

Transfer learning pozwala rozpocząć od wag wyuczonych na dużym, ogólnym zbiorze i dostosować je do nowego problemu. Rozwiązanie to jest powszechne w computer vision, ale łatwo popełnić przy nim subtelne błędy. Szczególnej uwagi wymagają parametry optymalizatora, tryb pracy modelu i warstwy BatchNorm.

## Intuicja

CNN trenowana na ImageNet uczy się hierarchii cech. Wczesne warstwy reagują na krawędzie, kontury i proste tekstury, które przydają się w wielu zadaniach obrazowych. Głębsze warstwy opisują bardziej złożone wzorce związane ze zbiorem treningowym.

Ogólne cechy można często wykorzystać bez zmian. Dostosowania wymagają przede wszystkim głębsze warstwy oraz nowa głowica klasyfikacyjna, ponieważ liczba i znaczenie klas różnią się od ImageNet.

❌ Model pretrained zawsze poprawia wynik, ponieważ wie więcej niż model trenowany od zera.

✅ Korzyść zależy od podobieństwa domen. ImageNet zawiera głównie codzienne zdjęcia obiektów i scen. Cechy z takiego zbioru mogą gorzej pasować do zdjęć satelitarnych, medycznych lub mikroskopowych. Efekt należy zmierzyć, a nie zakładać.

## Backbone i głowica

Model pretrained można zapisać jako złożenie dwóch funkcji.

$$f(x;\theta) = f_{\text{head}}\Big(f_{\text{backbone}}(x;\theta_{\text{backbone}});\ \theta_{\text{head}}\Big), \qquad \theta = (\theta_{\text{backbone}}, \theta_{\text{head}})$$

- $f_{\text{backbone}}$ to część konwolucyjna sieci. W ResNet18 obejmuje `conv1`, `bn1`, `layer1..layer4` i `avgpool`. Obraz $x\in\mathbb R^{3\times224\times224}$ jest mapowany na wektor $z\in\mathbb R^{512}$.
- $f_{\text{head}}$ to głowica klasyfikacyjna mapująca wektor cech na $C$ logitów. Domyślne `nn.Linear(512, 1000)` należy zastąpić przez `nn.Linear(512, C)` z losową inicjalizacją.

Parametry obu części są oznaczane przez $\theta_{\text{backbone}}$ i $\theta_{\text{head}}$. Symbol $\varphi$ pozostaje zarezerwowany dla funkcji aktywacji. Symbol $C$ oznacza tutaj liczbę klas, choć w konwencji NCHW oznacza liczbę kanałów. Znaczenie wynika z kontekstu.

## Trzy strategie adaptacji

Trening minimalizuje stratę

$$L(\theta) = \frac1N\sum_{i=1}^N\ell(f(x_i;\theta), y_i).$$

Strategie różnią się zbiorem parametrów, dla których obliczany i stosowany jest gradient.

### 1. Zamrożony backbone

Parametry backbone'u zachowują wartości pretrained i nie są optymalizowane.

$$\min_{\theta_{\text{head}}} L(\theta_{\text{backbone}}^{\text{pretrained}}, \theta_{\text{head}}), \qquad \theta_{\text{backbone}} \text{ ustalone, nie wchodzi do } \nabla$$

W PyTorch parametry backbone'u otrzymują `requires_grad=False`, a optymalizator dostaje wyłącznie `model.fc.parameters()`. Strategia jest też nazywana *linear probing*.

### 2. Częściowy fine-tuning

Backbone zostaje podzielony na część zamrożoną i trenowalną. Przykładowo `layer1..layer3` pozostają zamrożone, a `layer4` jest dostrajane.

$$\min_{\theta_{\text{backbone}}^{\text{unfrozen}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{\text{frozen}} = \theta_{\text{backbone}}^{\text{frozen,pretrained}} \text{ ustalone}$$

### 3. Pełny fine-tuning

Wszystkie parametry są trenowalne, ale backbone nadal rozpoczyna od wag pretrained.

$$\min_{\theta_{\text{backbone}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{(0)} = \theta_{\text{backbone}}^{\text{pretrained}}$$

### Gradient i grupy parametrów

`loss.backward()` oblicza gradient dla każdego parametru z `requires_grad=True`. Forward pass zawsze przechodzi przez cały model, ponieważ wyjście backbone'u jest potrzebne głowicy. Zamrożenie nie usuwa więc kosztu forward passu.

Różne części modelu mogą używać osobnych learning rate dzięki grupom parametrów.

```python
optimizer = torch.optim.Adam(
    [
        {"params": model.layer4.parameters(), "lr": eta_backbone},
        {"params": model.fc.parameters(), "lr": eta_head},
    ]
)
```

Zwykle $\eta_{\text{backbone}}$ jest od $10$ do $100\times$ mniejsze niż $\eta_{\text{head}}$. Wagi pretrained stanowią dobry punkt startowy, lecz nie muszą być optymalne dla nowego zadania. Duży krok może je szybko zaburzyć, zwłaszcza zanim losowo zainicjalizowana głowica zacznie tworzyć użyteczny sygnał. Często najpierw trenuje się samą głowicę, a dopiero później ostrożnie odmraża fragment backbone'u.

❌ Pełny fine-tuning zawsze daje lepszy wynik niż trening samej głowicy.

✅ Więcej trenowalnych parametrów zwiększa swobodę dopasowania, ale także ryzyko przeuczenia na małym zbiorze. O wyborze powinna decydować walidacja.

## Ile parametrów jest trenowanych

ResNet18 z głowicą dla $C=10$ klas ma następującą liczbę trenowalnych parametrów.

| Strategia | Parametry trenowalne | % całości |
|---|---:|---:|
| Cały model (referencja) | 11 181 642 | 100,00% |
| Zamrożony backbone, tylko `fc` | 5 130 | 0,05% |
| Częściowy fine-tuning, `layer4` + `fc` | 8 398 858 | 75,11% |
| Pełny fine-tuning | 11 181 642 | 100,00% |

Głowica ma $512\times10 + 10 = 5\,130$ parametrów. `layer4` zawiera 8 393 728 parametrów, więc wraz z głowicą daje 8 398 858.

Samo sprawdzenie `requires_grad` nie wystarcza. Warto porównać parametry przed rzeczywistym krokiem optymalizatora i po nim.

```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
# ... jeden krok: zero_grad -> forward -> backward -> step ...
for n, p in model.named_parameters():
    changed = not torch.equal(before[n], p.detach())
    assert changed == p.requires_grad, f"{n}: niespodziewany stan"
```

Notebook potwierdza, że żaden zamrożony tensor nie zmienia się, a wszystkie trenowalne tensory są aktualizowane po kilku krokach.

### Szczególny przypadek BatchNorm

ResNet18 ma 20 warstw `BatchNorm2d`. W trybie `model.train()` wszystkie aktualizują bufory `running_mean` i `running_var`, niezależnie od `requires_grad`. Dotyczy to także BatchNorm wewnątrz zamrożonego `layer1`. Aktualizacje zatrzymuje dopiero `model.eval()`.

Oznacza to, że `requires_grad=False` i `model.eval()` rozwiązują różne problemy. Pomylenie tych mechanizmów jest częstym, trudnym do zauważenia błędem.

## Eksperyment

Notebook [`examples.ipynb`](examples.ipynb) porównuje trzy strategie na 600 obrazach z 10 zbalansowanych klas Caltech101. Podział wynosi 360/120/120, a wybrane klasy mają dobre pokrycie w ImageNet.

1. ResNet18 trenowany od zera z losową inicjalizacją.
2. Zamrożony backbone w wariancie naiwnym oraz z cache'em cech.
3. Częściowy fine-tuning `layer4` i `fc` z niższym $\eta_{\text{backbone}}$.

Raportowane są parametry całkowite i trenowalne, straty, accuracy oraz czas treningu i inferencji. Porównanie przy tej samej liczbie epok nie jest tym samym co porównanie przy tym samym koszcie obliczeniowym.

## Obserwacje

| Strategia | Param. trenowalne | Accuracy walidacyjna | Czas treningu | Inferencja/obraz |
|---|---:|---:|---:|---:|
| Od zera (8 epok) | 11 181 642 (100%) | 0,65 | 86,5 s | 13,37 ms |
| Zamrożony backbone, naiwnie (5 epok) | 5 130 (0,05%) | 1,00 | 26,6 s | brak pomiaru |
| Zamrożony backbone, cache (5 epok) | 5 130 | 1,00 | **7,75 s** | 11,63 ms |
| Zamrożony backbone, finalnie (15 epok na cache'u) | 5 130 | 1,00 | 7,8 s | brak pomiaru |
| Częściowy fine-tuning, `layer4` + `fc` (8 epok) | 8 398 858 (75%) | 1,00 | 51,0 s | 12,41 ms |

Zamrożony backbone osiąga accuracy walidacyjną $1{,}00$ przy $0{,}05\%$ trenowalnych parametrów. Model trenowany od zera na tych samych 360 obrazach osiąga $0{,}65$. Dla dwóch ziaren zamrożony backbone uzyskuje $1{,}00/1{,}00$ na walidacji i $0{,}983/1{,}00$ na teście. Częściowy fine-tuning uzyskuje odpowiednio $1{,}00/1{,}00$ oraz $0{,}958/0{,}992$.

Częściowy fine-tuning nie przyniósł tu mierzalnej poprawy jakości. Nie oznacza to jednak, że jest ogólnie bezużyteczny. Wybrane klasy są dobrze reprezentowane w ImageNet, więc zamrożone cechy niemal idealnie rozdzielają dane. W laboratorium na EuroSAT częściowy fine-tuning daje poprawę. Dwa eksperymenty nie wystarczają do ustalenia ogólnej zależności między przesunięciem domeny a korzyścią z dostrajania.

Wariant naiwny i cache cech osiągają ten sam wynik po 5 epokach. Czasy wynoszą odpowiednio $26{,}6$ s i $7{,}75$ s, więc cache daje przyspieszenie $3{,}4\times$. Jest poprawny tylko wtedy, gdy backbone i preprocessing pozostają niezmienne.

Dodatkowy eksperyment wykorzystuje $6/18/36$ obrazów na klasę. Zamrożony backbone osiąga $0{,}992 \to 1{,}00 \to 1{,}00$, a trening od zera $0{,}250 \to 0{,}517 \to 0{,}575$. Przewaga transfer learningu jest największa przy małej ilości danych. Ten pomiar wykonano dla jednego ziarna, więc nie sprawdza stabilności między przebiegami.

## Częste błędne przekonania

❌ `requires_grad=False` działa tak samo jak `model.eval()`.

✅ `requires_grad` steruje obliczaniem gradientu. `train()` i `eval()` zmieniają zachowanie warstw takich jak BatchNorm i Dropout podczas forward passu. Zamrożone wagi mogą pozostać stałe, gdy statystyki BatchNorm nadal się zmieniają.

❌ `torch.no_grad()` i `torch.inference_mode()` są równoważne zamrożeniu parametrów.

✅ Oba mechanizmy wyłączają budowanie grafu obliczeniowego w danym fragmencie kodu. Stosuje się je podczas inferencji i jednorazowej ekstrakcji cech, ale nie podczas treningu głowicy. `inference_mode()` jest bardziej restrykcyjny, a utworzone w nim tensory nie nadają się do późniejszego różniczkowania.

❌ Zamrożony backbone oznacza, że cały model przestał się uczyć.

✅ Zamrożony jest tylko backbone. Nowa głowica nadal się uczy i odpowiada za dopasowanie do klas nowego zadania.

❌ Większy model pretrained jest zawsze lepszy.

✅ Większy model może mieć lepszą accuracy na ImageNet, ale wymaga więcej pamięci i obliczeń. Liczba parametrów nie jest też pewnym wskaźnikiem czasu wykonania.

❌ Niższa strata treningowa po fine-tuningu oznacza lepszą generalizację.

✅ Strata treningowa mierzy dopasowanie do znanych danych. Generalizację ocenia się na zbiorze walidacyjnym i testowym.

❌ Wagi pretrained zwalniają z walidacji i analizy błędów.

✅ Zmienia się punkt startowy optymalizacji, ale nadal potrzebne są uczciwy podział danych, confusion matrix i analiza błędów.

❌ Reprezentacje z ImageNet dobrze przenoszą się na każdą domenę obrazów.

✅ Duże przesunięcie dziedziny może ograniczyć korzyść, a czasem prowadzić do *negative transfer*. Dlatego transfer należy porównać z sensownym modelem trenowanym od zera.

## Implikacje inżynierskie

- **Pamięć.** Zamrożenie oszczędza pamięć na gradienty i stan optymalizatora backbone'u, ale forward pass nadal tworzy aktywacje.
- **Parametry i czas to różne miary.** MobileNetV3-Small ma $4{,}6\times$ mniej parametrów i $32\times$ mniej FLOPs niż ResNet18, lecz na badanym CPU działa $4\times$ wolniej. Konwolucje depthwise-separable mogą słabo wykorzystywać standardowe jądra CPU.
- **Cache cech ma wąskie zastosowanie.** Przyspiesza trening głowicy tylko dla niezmiennego backbone'u i deterministycznego preprocessingu. Po kroku fine-tuningu zapisane cechy stają się nieaktualne.
- **Wersja wag wpływa na reprodukowalność.** Wagi ResNet18 zajmują około 45 MB. Warto wskazać `ResNet18_Weights.IMAGENET1K_V1` zamiast aliasu `DEFAULT`, który może zmienić znaczenie.
- **Jakość, rozmiar i koszt wymagają osobnej oceny.** Wybór zależy od ograniczeń pamięci, czasu treningu i latencji inferencji.

## Co warto zapamiętać

- Transfer learning łączy backbone z nową głowicą. Strategie różnią się zakresem aktualizowanych parametrów.
- `requires_grad`, `train()` i `eval()` oraz `no_grad()` i `inference_mode()` pełnią różne funkcje.
- `requires_grad=False` nie zatrzymuje aktualizacji statystyk BatchNorm w trybie `train()`.
- Skuteczność transferu zależy od podobieństwa domen i wymaga weryfikacji na danych.

## Pytania dla studentów

1. Jak po 5 krokach `optimizer.step()` zachowają się `model.layer1[0].conv1.weight` i `model.layer1[0].bn1.running_mean`, jeśli backbone ma `requires_grad=False`, a model pozostaje w trybie `train()`?
2. Która strategia nadaje się do zadania z 50 obrazami nowej klasy? Odpowiedź powinna uwzględniać co najmniej dwa argumenty poza liczbą parametrów.
3. Dlaczego cache cech jest poprawny dla zamrożonego backbone'u i staje się nieaktualny po pierwszym kroku fine-tuningu?
4. Ile trenowalnych parametrów ma ResNet18 z głowicą `nn.Linear(512, 37)`, gdy odmrożone są `layer3`, `layer4` i głowica? `layer3` zawiera 2 099 712 parametrów.

## Bibliografia

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS.
- PyTorch documentation. *torchvision.models.resnet18*. `https://pytorch.org/vision/stable/models/resnet.html`.
- PyTorch documentation. *Autograd mechanics, Locally disabling gradient computation*. `https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation`.
