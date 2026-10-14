# Laboratorium 02: klasyfikacja tabularna z MLP

Czas: ok. 2 godziny. Bazuje na [`lectures/02_mlp/lecture.md`](../../lectures/02_mlp/lecture.md)
(perceptron, MLP, funkcje aktywacji), ale **nie powtarza** eksperymentu z
wykładu — inne dane, własna implementacja `nn.Module` (nie `nn.Sequential`),
porównanie architektur/aktywacji i analiza błędnych klasyfikacji, których
wykład nie robił.

## 1. Problem i dane

Zadanie: klasyfikacja binarna punktów na płaszczyźnie, które leżą na dwóch
**współśrodkowych okręgach** (pierścień zewnętrzny — klasa 0, okrąg
wewnętrzny — klasa 1), z szumem gaussowskim. To inny rodzaj nieliniowej
separowalności niż półksiężyce z wykładu: tutaj granica decyzyjna, której
potrzeba, jest zamkniętą krzywą (okręgiem), nie pojedynczą "falą".

Dane są generowane lokalnie w notebooku (sekcja 1, kod gotowy — nie jest
częścią zadania) i dzielone na zbiór treningowy/testowy (gotowe, sekcja 1).

## 2. Metryki — Twoje zadanie

Zaimplementuj (TODO, sekcja 2) funkcję liczącą z predykcji i etykiet:
**TP, FP, TN, FP, TN, FN**, a z nich **accuracy, precision, recall**
(klasa 1 — okrąg wewnętrzny — jako klasa pozytywna). Te funkcje są używane
przez resztę notebooka, także do oceny baseline'u.

## 3. Baseline (gotowe)

Pojedyncza warstwa liniowa (`nn.Linear(2, 1)`) — ten sam "model liniowy
zawodzi na nieliniowym problemie", co na wykładzie, tylko na innych danych.
Zobaczysz, że na okręgach model liniowy wypada **wyraźnie gorzej** niż na
półksiężycach z wykładu (bliżej przypadkowego zgadywania) — to nie błąd,
to konsekwencja kształtu danych: żadna prosta nie wydzieli sensownie
pierścienia od środka.

## 4. Twój MLP — Twoje zadanie

Zaimplementuj (TODO, sekcja 4) **własną klasę** `MLP(nn.Module)`
(nie `nn.Sequential`, w odróżnieniu od wykładu) z konfigurowalną liczbą
neuronów w warstwie ukrytej i konfigurowalną funkcją aktywacji, żeby można
jej było użyć do porównania z sekcji 6. Zaimplementuj też (TODO, sekcja 5)
funkcję trenującą (`forward → strata (BCEWithLogitsLoss) → backward →
optimizer.step()`), analogiczną do tej z wykładu, ale działającą na
dowolnym przekazanym jej modelu `nn.Module`.

## 5. Eksperyment (gotowe, korzysta z Twojego kodu)

Notebook trenuje jeden domyślny MLP Twoją klasą/funkcją i porównuje go z
baseline'em.

## 6. Porównanie architektur/aktywacji — Twoje zadanie

Wybierz **co najmniej trzy** konfiguracje swojego MLP, różniące się liczbą
neuronów w warstwie ukrytej i/lub funkcją aktywacji (TODO, sekcja 6).
Włącz do porównania przynajmniej jedną konfigurację, co do której
podejrzewasz, że będzie **zbyt słaba** (np. bardzo mała liczba neuronów) —
samo porównanie "większe zawsze lepsze" niczego nie uczy. Wytrenuj każdą
konfigurację i zbierz metryki testowe w jednej tabeli/wydruku.

## 7. Ewaluacja

Zinterpretuj (komórka Markdown, TODO) tabelę z sekcji 6: która konfiguracja
była najlepsza, czy któraś była wyraźnie za słaba, i — jeśli tak — dlaczego
(w kategoriach tego, co ta architektura może, a czego nie może
reprezentować, nawiązując do dowodu z wykładu).

## 8. Analiza błędnych klasyfikacji — Twoje zadanie

Notebook rysuje (gotowe) granicę decyzyjną Twojego najlepszego modelu z
zaznaczonymi błędnie sklasyfikowanymi punktami testowymi. Odpowiedz
(Markdown, TODO, 3–5 zdań): gdzie leżą błędnie sklasyfikowane punkty
względem granicy między pierścieniem a okręgiem? Czy to są punkty "z
natury trudne" (blisko prawdziwej granicy, ofiary szumu), czy raczej dowód,
że model systematycznie czegoś nie rozumie? Jak to rozstrzygniesz, patrząc
na wykres?

## 9. Wnioski — Twoje zadanie

Odpowiedz (Markdown, TODO):

- Czy wynik eksperymentu z sekcji 6 pasuje do tego, czego uczy dowód o
  złożeniu przekształceń afinicznych z wykładu? Czym różni się to, co
  pokazuje ten eksperyment, od tego, co pokazywał dowód?
- Gdybyś miał/miała jedno zdanie rady dla kogoś projektującego architekturę
  MLP do nowego problemu klasyfikacji, co by to było — na podstawie tego,
  co zaobserwowałeś/aś w tym laboratorium, nie z ogólników?

## Kryteria oceny

- Metryki, klasa `MLP` i funkcja trenująca działają (notebook wykonuje się
  od danej komórki do końca bez błędu).
- Porównanie w sekcji 6 obejmuje realnie różne konfiguracje (nie trzy razy
  to samo z inną nazwą zmiennej) i zawiera przynajmniej jedną wyraźnie
  słabszą.
- Interpretacja w sekcji 7 poprawnie łączy wynik liczbowy ze zdolnością
  reprezentacyjną architektury, nie tylko go opisuje.
- Analiza błędów w sekcji 8 odwołuje się do konkretnego położenia błędnych
  punktów na wykresie, nie do ogólników w stylu "model czasem się myli".
- Odpowiedzi we wnioskach są uzasadnione, nie jednowersowe.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/02_mlp/`) i nie jest częścią tego repozytorium.
