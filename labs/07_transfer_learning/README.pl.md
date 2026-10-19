Język: [English](README.md) | **Polski**

# Laboratorium 07: dostosowanie modelu pretrained do nowego problemu klasyfikacji obrazów

Czas: ok. 2 godziny. Bazuje na [`lectures/07_transfer_learning/lecture.md`](../../lectures/07_transfer_learning/lecture.md)
(backbone/głowica, zamrożony backbone, częściowy fine-tuning, `requires_grad` vs
`train()`/`eval()` vs `no_grad()`), ale na **innym zbiorze danych i innym pytaniu** niż
wykład — EuroSAT (zdjęcia satelitarne, nie zdjęcia obiektów z poziomu ziemi jak
Caltech101 na wykładzie) i pytanie: **czy transfer learning pomaga, gdy domena nowego
zadania jest wizualnie odległa od ImageNet, i czy częściowy fine-tuning daje tu coś
więcej niż sam zamrożony backbone** (na wykładzie — przy silnym pokryciu z ImageNet —
nie dawał; zobaczysz, czy tutaj jest inaczej).

## 0. Konfiguracja (gotowe)

Import bibliotek, urządzenie, ziarno losowości.

## 1. Dane: EuroSAT — struktura i eksploracja (gotowe + krótkie zadanie)

EuroSAT: 10 klas użycia terenu z satelitarnych zdjęć Sentinel-2 (`AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River,
SeaLake`), obrazy $64\times64$ RGB. Pobiera się automatycznie przez
`torchvision.datasets.EuroSAT` (nigdy nie trafia do repozytorium — zobacz
[`data/registry.yml`](../../data/registry.yml)).

**Zadanie:** obejrzyj po jednym przykładowym obrazie z każdej klasy (gotowy kod do
wyświetlenia). Czy te zdjęcia przypominają cokolwiek z ImageNet (zwierzęta, przedmioty,
sceny "z poziomu ziemi")? Zapisz zdaniem swoją wstępną intuicję — do niej wrócisz w
sekcji 8.

## 2. Preprocessing zgodny z wagami pretrained — Twoje zadanie

ResNet18 wymaga konkretnego preprocessingu (rozmiar wejścia, normalizacja) — **tego
samego**, jakiego użyto podczas treningu na ImageNet, inaczej cechy wyuczone przez
backbone nie będą miały sensu dla nowych danych.

Zaimplementuj `build_preprocess()`, która zwraca transformację zgodną z wagami
`ResNet18_Weights.IMAGENET1K_V1` (podpowiedź: `weights.transforms()` — nie przepisuj
tego ręcznie, użyj gotowej transformacji z obiektu wag, żeby mieć pewność zgodności z
tym, na czym model był trenowany).

Zweryfikuj: przepuść jeden obraz EuroSAT przez swoją funkcję i sprawdź, że wynikowy
kształt to `(3, 224, 224)` (gotowa asercja).

## 3. Podział i ekstrakcja podzbioru (gotowe)

500 obrazów (50/klasę, zbalansowane), podział 30/10/10 na train/val/test.

## 4. Wczytanie modelu i wymiana głowicy — Twoje zadanie

Wczytaj `resnet18` z wagami `ResNet18_Weights.IMAGENET1K_V1` (**przypięte explicite**,
nie `DEFAULT`) i zastąp `model.fc` nową warstwą `nn.Linear` o odpowiedniej liczbie
wyjść (10 klas EuroSAT).

**Zadanie (kod, 2-3 linie):** zweryfikuj programowo (nie tylko czytając kod), że nowa
głowica ma poprawną liczbę parametrów wyjściowych i że to **ten sam obiekt**, do
którego odwołuje się `model.fc` po przypisaniu (nie osobna, niepodłączona warstwa).

## 5. Zamrożenie backbone'u i trening samej głowicy — Twoje zadanie

Zamroź wszystkie parametry backbone'u (`requires_grad=False`), zostaw trenowalną tylko
głowicę. **Pamiętaj o kolejności**: najpierw wymiana głowicy (sekcja 4), dopiero potem
zamrażanie i budowa optymalizatora — zobaczysz dlaczego w sekcji diagnostycznej (sekcja
7).

Napisz pętlę treningową dla samej głowicy (możesz, ale nie musisz, użyć cache'owania
cech z wykładu — to przyspiesza trening, ale nie jest wymagane). Ustaw model w
odpowiednim trybie (`train()`/`eval()`) tak, żeby **statystyki BatchNorm backbone'u się
nie zmieniały** — to jest dokładnie błąd omówiony na wykładzie (sekcja *Błędne
przekonania*), tutaj masz go samodzielnie uniknąć, nie tylko przeczytać o nim.

**Zweryfikuj programowo**: po treningu zamrożone parametry backbone'u są identyczne jak
przed (gotowa asercja porównująca stan przed/po).

Zanotuj: accuracy walidacyjna i czas treningu.

## 6. Częściowy fine-tuning — Twoje zadanie

Odmroź **`layer4`** (ostatni blok konwolucyjny ResNet18) razem z głowicą. Zbuduj
optymalizator z **dwiema grupami parametrów** o różnych learning rate — mniejszy dla
`layer4`, większy dla głowicy (uzasadnienie w `lecture.md`, sekcja *Matematyka*).

**Zadanie (Markdown, 2-3 zdania):** jakie wartości learning rate wybrałaś/eś dla obu
grup i dlaczego akurat taki stosunek między nimi?

Zweryfikuj programowo: zamrożona część backbone'u (`layer1`-`layer3`) się nie zmieniła,
`layer4` i głowica faktycznie się zmieniły.

## 7. Zadanie diagnostyczne — Twoje zadanie

Poniższy kod (w notebooku) **nie działa poprawnie** — strata w ogóle nie maleje mimo
wielu kroków treningu, accuracy zostaje na poziomie przypadkowego zgadywania. Kod nie
rzuca żadnego wyjątku — wygląda na poprawny.

**Zadanie (Markdown, 4-6 zdań):** znajdź przyczynę, odwołując się do **konkretnej
linii** kodu i do tego, co faktycznie obserwujesz (strata, `requires_grad`, zawartość
`optimizer.param_groups`). Nie wystarczy "coś jest nie tak z optymalizatorem" — wskaż
dokładnie, który obiekt dostaje aktualizacje, a który powinien, ale ich nie dostaje, i
dlaczego to się nie rzuca w oczy przy samym sprawdzeniu `requires_grad`.

(Podpowiedź do weryfikacji, nie do razu do rozwiązania: porównaj `id()` obiektu w
`optimizer.param_groups` z `id(model.fc.weight)` już **po** uruchomieniu komórki z
błędem.)

## 8. Porównanie strategii i wpływ ustawień — Twoje zadanie

Zestaw wyniki zamrożonego backbone'u (sekcja 5) i częściowego fine-tuningu (sekcja 6):
accuracy walidacyjna i testowa, liczba trenowalnych parametrów, czas treningu. Zbiór
testowy wykorzystujesz tutaj **po raz pierwszy**, po wyborze lepszej konfiguracji na
podstawie walidacji.

**Zadanie:** zbadaj wpływ **jednego** ustawienia na wynik częściowego fine-tuningu — do
wyboru: inny stosunek $\eta_{\text{backbone}}/\eta_{\text{head}}$ (np. $10\times$ vs
$100\times$ mniejszy) **albo** inny zakres odmrażania (`layer3`+`layer4` zamiast
samego `layer4`). Krótko uzasadnij wybór i zinterpretuj wynik.

## 9. Confusion matrix i analiza błędów — Twoje zadanie

Dla lepszej (według walidacji) strategii: confusion matrix na zbiorze testowym (gotowa
funkcja do rysowania czeka na dane) i analiza 6-8 błędnie sklasyfikowanych przykładów.

**Zadanie (Markdown, 3-5 zdań):** które klasy użycia terenu są najczęściej mylone?
Patrząc na same obrazy — czy pomyłki wydają się sensowne (podobny wygląd terenu z
lotu ptaka, np. różne typy upraw) czy przypadkowe?

## 10. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami z sekcji 5-9:

1. Wróć do swojej intuicji z sekcji 1 — czy transfer learning z ImageNet rzeczywiście
   pomógł na danych tak odległych wizualnie (zdjęcia satelitarne)? Jak duża jest
   różnica względem tego, czego się spodziewałaś/eś?
2. Czy częściowy fine-tuning dał **mierzalną** korzyść względem samego zamrożonego
   backbone'u, czy tylko zwiększył koszt? Porównaj z wynikiem z wykładu (Caltech101) —
   czy odpowiedź jest taka sama, czy różna? Co to mówi o tym, kiedy warto inwestować w
   fine-tuning, a kiedy nie?
3. Jakie jest co najmniej jedno ograniczenie tego eksperymentu (rozmiar podzbioru,
   liczba epok, konkretny zakres odmrażania), które nie pozwala uogólnić wniosków na
   "strategia X jest zawsze lepsza dla danych satelitarnych"?

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, eksploracja) | 10 min |
| Sekcja 2 (preprocessing + weryfikacja) | 10 min |
| Sekcja 3 (podział, gotowe) | 5 min |
| Sekcja 4 (wymiana głowicy + weryfikacja) | 10 min |
| Sekcja 5 (zamrożony backbone + trening głowicy) | 20 min |
| Sekcja 6 (częściowy fine-tuning, 2 grupy LR) | 20 min |
| Sekcja 7 (zadanie diagnostyczne) | 15 min |
| Sekcja 8 (porównanie + wpływ ustawień) | 15 min |
| Sekcja 9 (confusion matrix + błędy) | 10 min |
| Sekcja 10 (wnioski) | 10 min |

Sekcje 0-9 to rdzeń obowiązkowy. Trening na zredukowanym podzbiorze (500 obrazów)
trwa pojedyncze do kilkudziesięciu sekund na CPU — budżet czasowy ma iść w decyzje
projektowe i analizę, nie w czekanie na trening.

**Rozszerzenie (opcjonalne, jeśli zostanie czas):** trening od zera (bez wag
pretrained) jako dodatkowy punkt odniesienia — ile accuracy traci się, rezygnując z
transfer learningu całkowicie na tym zadaniu? Albo: pełny fine-tuning (wszystkie
warstwy odmrożone) — czy daje coś więcej niż częściowy, i jakim kosztem?

## Kryteria oceny

- Preprocessing zgodny z wagami pretrained (`weights.transforms()`, nie ręczna
  normalizacja "na oko"); asercja kształtu przechodzi.
- Zamrożenie backbone'u w sekcji 5 faktycznie nie zmienia jego parametrów (asercja
  przechodzi) i nie zmienia statystyk BatchNorm (poprawny tryb modelu).
- Częściowy fine-tuning w sekcji 6 używa dwóch grup parametrów z różnym LR; zamrożona
  część backbone'u faktycznie się nie zmienia.
- Zadanie diagnostyczne (sekcja 7) wskazuje konkretną przyczynę w kodzie, nie ogólnik.
- Zbiór testowy pojawia się dopiero w sekcji 8, po wyborze na podstawie walidacji.
- Wnioski w sekcji 10 odwołują się do konkretnych liczb z Twojego przebiegu i
  jawnie porównują z wynikiem z wykładu, nie do ogólników.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/07_transfer_learning/`) i nie jest częścią tego repozytorium.
