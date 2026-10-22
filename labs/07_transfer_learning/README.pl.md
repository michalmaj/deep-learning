Język: [English](README.md) | **Polski**

# Laboratorium 07. Adaptacja modelu pretrained do nowego zadania

Laboratorium trwa około 2 godzin i rozwija materiał z
[`lectures/07_transfer_learning/lecture.md`](../../lectures/07_transfer_learning/lecture.md).
Obejmuje zamrożony backbone, częściowy fine-tuning oraz różnice między
`requires_grad`, trybami modelu i `no_grad()`.

Eksperyment wykorzystuje zdjęcia satelitarne EuroSAT, czyli domenę wizualnie odległą
od codziennych zdjęć z ImageNet i Caltech101. Celem jest sprawdzenie, czy transfer
learning nadal pomaga oraz czy częściowy fine-tuning poprawia wynik zamrożonego
backbone'u.

## 0. Konfiguracja (gotowe)

Import bibliotek, wybór urządzenia i ustawienie ziarna.

## 1. Dane EuroSAT (gotowe + krótkie zadanie)

EuroSAT zawiera 10 klas użycia terenu na zdjęciach Sentinel-2: `AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River, SeaLake`. Obrazy RGB mają rozmiar $64\times64$.

Zbiór jest pobierany automatycznie przez `torchvision.datasets.EuroSAT` i nie trafia
do repozytorium. Szczegóły znajdują się w
[`data/registry.yml`](../../data/registry.yml).

**Zadanie.** Po obejrzeniu jednego obrazu z każdej klasy należy w jednym zdaniu
zapisać, czy dane przypominają typowe obrazy z ImageNet. Ta początkowa intuicja
zostanie porównana z wynikiem w sekcji 10.

## 2. Preprocessing zgodny z wagami pretrained

ResNet18 wymaga takiego samego rozmiaru wejścia i normalizacji jak podczas treningu
na ImageNet. Funkcja `build_preprocess()` powinna zwracać transformację powiązaną z
`ResNet18_Weights.IMAGENET1K_V1` przez `weights.transforms()`. Ręczne przepisywanie
wartości normalizacji nie jest potrzebne.

Gotowa asercja sprawdza, czy jeden obraz EuroSAT po transformacji ma kształt
`(3, 224, 224)`.

## 3. Podział i podzbiór (gotowe)

Używanych jest 500 zbalansowanych obrazów, po 50 z każdej klasy. Podział na zbiory
treningowy, walidacyjny i testowy wynosi 30/10/10 obrazów na klasę.

## 4. Model i nowa głowica

Należy wczytać `resnet18` z jawnie wskazanymi wagami
`ResNet18_Weights.IMAGENET1K_V1`, a następnie zastąpić `model.fc` warstwą `nn.Linear`
z 10 wyjściami. Alias `DEFAULT` nie powinien być używany.

**Zadanie w kodzie, 2-3 linie.** Należy programowo potwierdzić poprawną liczbę wyjść
oraz to, że utworzona głowica jest tym samym obiektem, do którego odwołuje się
`model.fc`, a nie niepodłączoną warstwą.

## 5. Zamrożony backbone i trening głowicy

Wszystkie parametry backbone'u powinny otrzymać `requires_grad=False`, a głowica ma
pozostać trenowalna. Kolejność ma znaczenie. Najpierw wymieniana jest głowica, później
zamrażane są parametry i tworzony jest optymalizator. Sekcja 7 pokazuje konsekwencje
błędnej kolejności.

Należy napisać pętlę treningową dla głowicy. Cache cech jest opcjonalny. Tryby
`train()` i `eval()` trzeba ustawić tak, aby statystyki BatchNorm w backbone nie
zmieniały się podczas treningu głowicy.

Gotowa asercja porównuje parametry backbone'u przed treningiem i po nim. Do zapisania
pozostają accuracy walidacyjna oraz czas treningu.

## 6. Częściowy fine-tuning

Należy odmrozić `layer4` wraz z głowicą i utworzyć optymalizator z dwiema grupami
parametrów. `layer4` powinno korzystać z niższego learning rate niż głowica.

**Zadanie w Markdown, 2-3 zdania.** Trzeba podać wartości learning rate obu grup i
uzasadnić ich stosunek.

Weryfikacja powinna potwierdzić, że `layer1` do `layer3` pozostały bez zmian, a
`layer4` i głowica zostały zaktualizowane.

## 7. Zadanie diagnostyczne

Kod w notebooku wygląda poprawnie i nie zgłasza wyjątku, lecz strata nie maleje, a
accuracy pozostaje blisko poziomu losowego zgadywania.

**Zadanie w Markdown, 4-6 zdań.** Należy wskazać konkretną linię powodującą problem,
opisać obserwowaną stratę, wartości `requires_grad` oraz zawartość
`optimizer.param_groups`. Odpowiedź powinna wyjaśniać, który obiekt otrzymuje
aktualizacje, który powinien je otrzymywać i dlaczego samo sprawdzenie
`requires_grad` nie ujawnia błędu.

Pomocna jest kontrola `id()` obiektu w `optimizer.param_groups` i
`id(model.fc.weight)` po uruchomieniu wadliwej komórki.

## 8. Porównanie strategii

Wyniki zamrożonego backbone'u i częściowego fine-tuningu należy zestawić pod względem
accuracy walidacyjnej i testowej, liczby trenowalnych parametrów oraz czasu treningu.
Zbiór testowy może zostać użyty po raz pierwszy dopiero tutaj, po wyborze konfiguracji
na podstawie walidacji.

**Rozszerzenie opcjonalne.** Można zmienić stosunek
$\eta_{\text{backbone}}/\eta_{\text{head}}$ albo odmrozić `layer3` wraz z `layer4`.
Dodatkowy przebieg nie należy do obowiązkowego rdzenia.

## 9. Confusion matrix i analiza błędów

Dla strategii wybranej na podstawie walidacji należy narysować confusion matrix na
zbiorze testowym i pokazać 6-8 błędnie sklasyfikowanych obrazów.

**Zadanie w Markdown, 3-5 zdań.** Analiza powinna wskazać najczęściej mylone klasy i
ocenić, czy podobny wygląd terenu z lotu ptaka wyjaśnia pomyłki.

## 10. Wnioski

Odpowiedzi powinny korzystać z rzeczywistych wyników z sekcji 5-9.

1. Czy wynik potwierdził początkową intuicję dotyczącą transferu z ImageNet do zdjęć
   satelitarnych? Jak duża była różnica względem oczekiwań?
2. Czy częściowy fine-tuning dał mierzalną poprawę względem zamrożonego backbone'u,
   czy tylko zwiększył koszt? Jak wynik różni się od eksperymentu Caltech101 z
   wykładu?
3. Jakie ograniczenie, na przykład rozmiar podzbioru, liczba epok lub zakres
   odmrażania, utrudnia uogólnienie wniosku na wszystkie dane satelitarne?

## Orientacyjny podział czasu

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, eksploracja) | 10 min |
| Sekcja 2 (preprocessing + weryfikacja) | 10 min |
| Sekcja 3 (podział, gotowe) | 5 min |
| Sekcja 4 (wymiana głowicy + weryfikacja) | 10 min |
| Sekcja 5 (zamrożony backbone + trening głowicy) | 20 min |
| Sekcja 6 (częściowy fine-tuning, 2 grupy LR) | 20 min |
| Sekcja 7 (zadanie diagnostyczne) | 15 min |
| Sekcja 8 (porównanie strategii) | 10 min |
| Sekcja 9 (confusion matrix + błędy) | 10 min |
| Sekcja 10 (wnioski) | 10 min |

Sekcje 0-10 tworzą obowiązkowy rdzeń. Trening na 500 obrazach trwa na CPU od kilku
do kilkudziesięciu sekund, dlatego większość czasu można przeznaczyć na decyzje i
analizę.

**Rozszerzenia opcjonalne.** Można wykonać dodatkowy eksperyment z sekcji 8,
wytrenować model od zera albo przeprowadzić pełny fine-tuning i porównać jego koszt.

## Kryteria oceny

- Preprocessing korzysta z `weights.transforms()`, a asercja kształtu przechodzi.
- Zamrożony backbone i jego statystyki BatchNorm pozostają bez zmian.
- Częściowy fine-tuning używa dwóch grup parametrów z różnymi learning rate, a
  zamrożona część modelu się nie zmienia.
- Diagnoza z sekcji 7 wskazuje konkretną przyczynę w kodzie.
- Zbiór testowy pojawia się dopiero w sekcji 8 po wyborze na walidacji.
- Wnioski odwołują się do zmierzonych wartości i porównują wynik z wykładem.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego w
`decisions/solutions/07_transfer_learning/` i nie jest częścią repozytorium.
