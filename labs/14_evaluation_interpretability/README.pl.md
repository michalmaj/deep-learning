Język: [English](README.md) | **Polski**

# Laboratorium 14: audyt błędów i interpretacja decyzji klasyfikatora obrazów

Czas: ok. 120 minut. Bazuje na
[`lectures/14_evaluation_interpretability/lecture.md`](../../lectures/14_evaluation_interpretability/lecture.md)
(metryki, kalibracja, Grad-CAM), ale na **innym problemie** niż wykład (tam:
Fashion-MNIST, mały CNN od zera). Tu: **EuroSAT** (zdjęcia satelitarne
Sentinel-2, moduły 07-08) + **ResNet18** `IMAGENET1K_V1` z zamrożonym
backbone'em i trenowaną głowicą (wzorem modułu 07).

**Cel**: nie jeden wykres Grad-CAM — **raport audytu klasyfikatora**,
odpowiadający na konkretne pytania o jakość, błędy, pewność i interpretację
decyzji modelu.

## 0. Konfiguracja i model (gotowe)

Import bibliotek, ziarno losowości, wczytanie EuroSAT
(`torchvision.datasets.EuroSAT`), wczytanie `resnet18(weights=
ResNet18_Weights.IMAGENET1K_V1)` z zamrożonym backbone'em i nową głowicą
`nn.Linear` (10 klas). Funkcja `extract_features()` do prekomputacji cech
zamrożonego backbone'u (drastycznie przyspiesza trening głowicy — nie
trzeba przeliczać konwolucji przy każdej epoce).

## 1. Dane i podział train/validation/test — Twoje zadanie

EuroSAT ma 27000 obrazów w 10 klasach (`AnnualCrop`, `Forest`,
`HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`,
`Residential`, `River`, `SeaLake`). Dla szybkości CPU, użyj **podzbioru**
(np. 1800 train / 400 val / 400 test, bez nakładania).

**Zadanie:**
1. Wylosuj indeksy i podziel na `train_idx`/`val_idx`/`test_idx` —
   **zweryfikuj programowo** brak nakładania.
2. Wypisz rozkład klas w każdym podzbiorze — czy są wyraźnie
   nierównomierne?
3. Krótki komentarz: co by się stało, gdybyś użył **tego samego** podziału
   do wyboru hiperparametrów **i** do finalnej oceny?

## 2. Trening głowicy (gotowe + krótkie zadanie)

Ekstrakcja cech (`extract_features`) i trening liniowej głowicy na
prekomputowanych cechach są **gotowe** — to nie jest centralny temat tego
labu (transfer learning było modułem 07). **Zadanie:** wywołaj trening na
**`train_idx`** (nie na `val_idx` czy `test_idx`!) i wypisz czas treningu.

## 3. Ewaluacja na niezależnym zbiorze — Twoje zadanie

Na **`test_idx`** (nigdy nie dotykanym do tego momentu): policz accuracy,
macro-F1, precision/recall/F1 per-class. Zbuduj i narysuj confusion matrix
($10\times10$).

**Zadanie (komentarz):** która klasa ma najniższy recall? Czy to zgadza się
z Twoją intuicją o tym, które kategorie krajobrazu są wizualnie podobne
(np. `PermanentCrop` vs `AnnualCrop`, `Highway` vs `Residential`)?

## 4. Najczęstsze i najkosztowniejsze błędy — Twoje zadanie

Z confusion matrix, znajdź **dwie** najczęściej mylone pary klas (patrz
wykład — nie wybieraj ich "na oko", odczytaj z macierzy). Dla **jednej**
z nich, zastanów się (komentarz): czy pomylenie tych dwóch konkretnych klas
ma inny "koszt" niż pomylenie innej pary (np. czy `River`→`SeaLake` jest
mniej kosztowne niż `Industrial`→`Residential` w jakimś realnym
zastosowaniu)?

## 5. Confidence poprawnych i błędnych predykcji — Twoje zadanie

Policz confidence (= $\max_c \hat y_c$) dla wszystkich predykcji testowych.
Porównaj rozkład dla poprawnych i błędnych (histogram). Znajdź **najbardziej
pewną błędną** predykcję — zapisz jej indeks, prawdziwą i przewidzianą
klasę, confidence.

## 6. Wybór próbek do szczegółowej analizy — Twoje zadanie

Zdefiniuj **jawną procedurę** (nie "na oko") wyboru 3 przykładów do
Grad-CAM/occlusion: (a) poprawna predykcja o wysokiej confidence; (b)
pomyłka o niskiej confidence (model "wie", że nie jest pewny); (c) pomyłka o
wysokiej confidence (myląca). Zapisz dokładnie, jak każdy został wybrany
(np. "pierwszy przykład w `test_idx`, dla którego warunek X jest prawdą").

## 7. Grad-CAM — Twoje zadanie

Zaimplementuj klasę `GradCAM` (forward hook na aktywacji wybranej warstwy,
`register_full_backward_hook` na gradiencie, $\alpha_k^c$ = global-average-
pool gradientu, ważona suma + ReLU, upsampling bilinearny, **obsługa
przypadku zerowej rozpiętości** mapy). Użyj `backbone.layer4` jako wybranej
warstwy (ostatni blok konwolucyjny ResNet18, przed `avgpool`).

**Uwaga, zweryfikowana w praktyce**: backbone jest **zamrożony**
(`requires_grad=False`). To oznacza, że samo `model.eval()` **nie
wystarczy** — potrzebujesz `x.requires_grad_(True)` na **wejściu**, inaczej
graf gradientów nigdy nie dotrze do `layer4` (zobacz wykład, sekcja
`eval()`/`no_grad()`/`inference_mode()`).

## 8. Porównanie heatmap i occlusion sensitivity — Twoje zadanie

Dla **trzech** przykładów z sekcji 6: policz i narysuj Grad-CAM (dla klasy
**przewidzianej**). Dla **jednego** z nich (poprawna predykcja o wysokiej
confidence), dodatkowo policz occlusion sensitivity (patch $32\times32$,
stride $32$, baseline = znormalizowany mid-gray, **zgodny** z
`ResNet18_Weights.IMAGENET1K_V1.transforms()` — **nie** surowe zero, bo to
nie jest zero w przestrzeni znormalizowanej).

**Zadanie (komentarz):** regiony Grad-CAM i occlusion **zgadzają się** czy
**różnią się**? Czy to ma sens, biorąc pod uwagę, że to są **różne pytania**
(gradient lokalny vs efekt fizycznego usunięcia)?

## 9. Analiza podgrup LUB odporność — Twoje zadanie (wybierz JEDNO)

**Opcja A — podgrupy (jasność):** podziel `test_idx` na "jasne"/"ciemne"
(mediana średniej wartości pikseli). Policz accuracy w każdej grupie.
**Zweryfikuj rozkład klas w każdej grupie** — czy różnica accuracy mogłaby
wynikać z różnic w rozkładzie klas, nie z samej jasności?

❌ "Różnica accuracy między grupami jasności dowodzi, że model jest
niesprawiedliwy/stronniczy względem jasnych lub ciemnych obrazów."

✅ **Nie wolno tego zakładać bez sprawdzenia.** Jasność **koreluje** z
klasą (np. las jest zwykle ciemniejszy niż tereny przemysłowe) — różnica
accuracy może być **w dużej mierze** konsekwencją tego, że jedna grupa ma
więcej przykładów z trudniejszych klas, nie efektem samej jasności.
**Zweryfikuj to wprost**, nie zakładaj.

**Opcja B — odporność (flip horyzontalny):** zdjęcia satelitarne nie mają
"kanonicznej" orientacji (w przeciwieństwie np. do zdjęć ludzi) — flip
horyzontalny **powinien** zachowywać etykietę. Policz accuracy/confidence
na `test_idx` i jego flipowanej wersji. Znajdź **konkretne** przykłady, dla
których predykcja **się zmieniła**.

❌ "Jeśli model jest odporny na flip, jest odporny na rzeczywisty
distribution shift (inny sensor, inna sezonowość, inny region)."

✅ **Nieprawda.** Jedna, kontrolowana transformacja testuje **tylko** tę
jedną transformację — nie uogólnia się automatycznie na inne, prawdziwe
źródła zmiany rozkładu danych.

## 10. Zadanie diagnostyczne

Poniższy, **oddzielnie dostarczony** kod próbuje policzyć Grad-CAM, ale ma
**dwa powiązane** problemy: (a) cała analiza jest zawarta w bloku
`torch.no_grad()` (ktoś "zoptymalizował" kod, zakładając, że inferencja nie
potrzebuje gradientów), **i** (b) nawet po usunięciu `no_grad()`, backbone
jest zamrożony i wejście nie ma `requires_grad=True`.

**Zadanie:**
1. Uruchom dostarczony kod — zaobserwuj **konkretny** błąd (nie "coś nie
   działa" — podaj typ wyjątku i miejsce).
2. Usuń `torch.no_grad()`. Uruchom ponownie — **nadal** błąd. Zdiagnozuj,
   **dlaczego** usunięcie `no_grad()` nie wystarczyło.
3. Napraw **w pełni** (dodaj `x.requires_grad_(True)` na wejściu).
4. Zweryfikuj: mapa Grad-CAM jest skończona (`torch.isfinite`) i ma
   niezerową rozpiętość dla **przynajmniej jednego** z Twoich przykładów z
   sekcji 6.

❌ "Ten błąd wymaga zgadywania, jak działają hooki PyTorch wewnętrznie."

✅ **Nie — błąd daje czysty, wyjaśnialny sygnał** (`None` gradient /
konkretny `RuntimeError`/`AttributeError`), zrozumiały przez odwołanie do
tego, czego się uczysz w tym module (`eval()` vs `no_grad()` vs graf
gradientów), nie przez zgadywanie szczegółów implementacji hooków.

## 11. Raport końcowy — Twoje zadanie

Napisz krótki raport (markdown, w notebooku), odpowiadający **wprost** na:

1. Jak dobrze model klasyfikuje poszczególne klasy? (konkretne liczby)
2. Które pomyłki są najczęstsze? (konkretne pary, z confusion matrix)
3. Czy model bywa mocno pewny błędnych decyzji? (konkretny przykład)
4. Na jakie regiony wejścia wskazuje Grad-CAM? (dla Twoich 3 przykładów)
5. Czy analiza przez perturbacje (occlusion) wspiera tę interpretację?
6. Czy zachowanie modelu zmienia się po Twojej wybranej modyfikacji wejścia
   (sekcja 9)?
7. Co należałoby sprawdzić przed wykorzystaniem tego modelu w praktyce?

**Ważne**: przykłady z sekcji 6/8 są **wybranymi ilustracjami**, nie
reprezentatywną próbą całego zbioru — nie wyciągaj ogólnych wniosków o
jakości modelu z 3 heatmap. Ogólne wnioski (punkty 1-2) oprzyj na **całym**
zbiorze testowym (sekcje 3-4), nie na wybranych przykładach.

## Podział czasu (realistyczny, zweryfikowany względem rzeczywistego kosztu CPU)

| Blok | Czas |
|---|---|
| Sekcja 0 (konfiguracja, model) | 5 min |
| Sekcja 1 (dane, podział) | 10 min |
| Sekcja 2 (trening głowicy) | 5 min |
| Sekcja 3 (ewaluacja, confusion matrix) | 10 min |
| Sekcja 4 (najczęstsze błędy) | 8 min |
| Sekcja 5 (confidence) | 8 min |
| Sekcja 6 (wybór próbek) | 5 min |
| Sekcja 7 (Grad-CAM) | 13 min |
| Sekcja 8 (porównanie + occlusion) | 12 min |
| Sekcja 9 (podgrupy LUB odporność) | 8 min |
| Sekcja 10 (zadanie diagnostyczne) | 10 min |
| Sekcja 11 (raport) | 10 min |

Suma: **104 min** rdzenia + **16-minutowy** bufor (razem 120 min). Koszt
CPU: ekstrakcja cech (~2600 obrazów) ok. 30s **jednorazowo**; trening
głowicy (30 epok na prekomputowanych cechach) <1s; Grad-CAM/occlusion dla
kilku obrazów — sekundy. Budżet czasowy idzie w implementację Grad-CAM,
analizę błędów i pisanie raportu, nie w czekanie na model.

## Kryteria oceny

- Podział train/val/test zweryfikowany jako rozłączny; głowica trenowana
  **wyłącznie** na `train_idx`.
- Metryki (accuracy, macro-F1, per-class) policzone na **niezależnym**
  `test_idx`, nie na danych użytych do treningu.
- Grad-CAM zaimplementowany samodzielnie (hook-based), działa na
  zamrożonym backbonie (z poprawnym `requires_grad` na wejściu),
  zweryfikowany (skończone wartości, niezerowa rozpiętość).
- Wybór próbek do analizy (sekcja 6) jest **jawny i odtwarzalny** — nie
  "wybrałem, co wyglądało dobrze".
- Analiza podgrup/odporności (sekcja 9) jawnie sprawdza alternatywne
  wyjaśnienia (konfundowanie klasą), nie wyciąga wniosku o "niesprawiedliwości"
  bez podstawy.
- Zadanie diagnostyczne identyfikuje **obie** przyczyny błędu, nie tylko
  pierwszą oczywistą.
- Raport odpowiada na wszystkie 7 pytań, odwołując się do konkretnych
  liczb/przykładów, z jasnym rozróżnieniem wniosków ogólnych (cały zbiór)
  od ilustracji (wybrane przykłady).

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/14_evaluation_interpretability/`) i nie jest częścią
tego repozytorium.
