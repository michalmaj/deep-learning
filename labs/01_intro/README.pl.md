Język: [English](README.md) | **Polski**

# Laboratorium 01. Podstawy PyTorch, tensory, urządzenia i pierwszy model

**Czas:** około 2 godzin.

Laboratorium rozwija mechanizm przedstawiony w
[`lectures/01_intro/lecture.md`](../../lectures/01_intro/lecture.md), czyli
regresję liniową, gradient i pętlę treningową. Eksperyment obejmuje wiele
cech oraz przypadek, w którym model liniowy *nie wystarcza*.

## 1. Problem

Na wykładzie przedstawiono model jednej zmiennej $\hat{y} = wx + b$.
Rzeczywiste zbiory danych rzadko ograniczają się do jednej cechy. W tym
laboratorium model zostanie rozszerzony do regresji liniowej **wielu
zmiennych** w postaci macierzowej.

$$\hat{y} = Xw + b$$

Macierz $X$ ma kształt $(N, 3)$ i zawiera trzy cechy. Wektor $w$ ma
kształt $(3,)$.

Celem jest przewidzenie ciągłej zmiennej $y$ na podstawie cech $x_1$, $x_2$
i $x_3$. Jedna z nich wpływa na proces generowania danych w sposób
**nieliniowy**. Oprócz wytrenowania modelu trzeba więc **wykryć i
zinterpretować** wynikający z tego systematyczny błąd.

## 2. Dataset

Dane są syntetyczne i powstają lokalnie w sekcji 1 notebooka `lab.ipynb`.
Laboratorium nie wymaga pobierania dodatkowych plików. Kod generujący dane
jest gotowy i nie stanowi części zadania.

## 3. Metryki i podział danych

Po wygenerowaniu danych należy uzupełnić dwie funkcje używane w dalszej
części laboratorium. Odpowiednie miejsca oznaczono jako TODO w sekcji 2.

- funkcję dzielącą indeksy przykładów na zbiór treningowy i walidacyjny
  (ok. 80/20),
- metryki **MSE** (błąd średniokwadratowy) i **$R^2$**
  ($1 - SS_{\text{res}}/SS_{\text{tot}}$).

Te same funkcje są później wykorzystywane przez baseline, trening i
ewaluację.

## 4. Baseline

Przed treningiem potrzebny jest punkt odniesienia. Notebook zawiera gotowy
baseline, który jako predykcję zwraca średnią wartość $y$ ze zbioru
treningowego. Nie korzysta przy tym z $x$, dlatego poprawnie działający
model powinien osiągnąć lepszy wynik. Wartości MSE i $R^2$ są obliczane za
pomocą funkcji z sekcji 3.

## 5. Model i trening

W dalszych komórkach należy uzupełnić miejsca oznaczone jako `# TODO`.
Do tego czasu kod zgłasza `NotImplementedError`. Zakres obejmuje trzy
elementy.

1. **`forward(X, w, b)`**, czyli predykcję modelu liniowego w postaci
   macierzowej. Jest to uogólnienie `w * x + b` z wykładu na iloczyn
   macierzowy.
2. **Rozwiązanie w postaci zamkniętej** (*normal equation*)
   $w^\* = (X^\top X)^{-1} X^\top y$, z kolumną jedynek dla biasu. Stanowi
   ono niezależny od gradientu punkt odniesienia dla modelu trenowanego
   gradientowo.
3. **Pętlę treningową opartą na spadku gradientu**, która rozszerza pętlę z
   wykładu na wektorowe $w$. Nie używa `nn.Module` ani `optim`, więc kroki
   `forward → strata → backward → aktualizacja` trzeba zapisać bezpośrednio.

Dostarczony kod obsługuje generowanie danych, wybór urządzenia, baseline i
wykresy. Nie wymaga zmian.

## 6. Eksperyment

Eksperyment polega na wytrenowaniu modelu i porównaniu straty baseline'u,
modelu trenowanego gradientowo oraz rozwiązania w postaci zamkniętej.
Model trenowany gradientowo i rozwiązanie w postaci zamkniętej powinny
prowadzić do **zbliżonych** wyników, ponieważ problem jest wypukły i ma to
samo optimum. Duża różnica sygnalizuje błąd w pętli treningowej, na
przykład niewłaściwy współczynnik uczenia, krok aktualizacji albo liczbę
epok.

## 7. Ewaluacja

MSE i $R^2$ wytrenowanego modelu należy obliczyć na zbiorze
**walidacyjnym**. Metryki ze zbioru treningowego nie pokazują, jak model
radzi sobie z nowymi danymi.

## 8. Analiza błędów

Gotowy kod rysuje reszty (*residuals*, $\hat{y} - y$) względem każdej z
trzech cech w zbiorze walidacyjnym. Dla jednej cechy pojawia się
**systematyczny, nieprzypadkowy wzorzec**, a nie losowy szum wokół zera.

W komórce Markdown należy w 3–5 zdaniach wskazać tę cechę, opisać kształt
wzorca i wyjaśnić jego związek z ograniczeniami modelu liniowego. Nie
trzeba odgadywać rzeczywistego wzoru, którego nie podano w zadaniu. Analiza
powinna rozpoznać na podstawie reszt taki rodzaj zależności, którego model
liniowy nie potrafi przedstawić.

## 9. Wnioski

Ostatnia sekcja notebooka zawiera dwa pytania do krótkiej odpowiedzi w
komórce Markdown.

- Czy ten sam problem dałoby się rozwiązać lepiej, dodając nową *cechę*
  (np. kwadrat jednej z istniejących) bez zmiany samego modelu liniowego?
  Z czego wynika odpowiedź?
- Dlaczego porównanie z rozwiązaniem w postaci zamkniętej jest użytecznym
  testem poprawności pętli treningowej?

## Kryteria oceny

- `forward`, podział danych, metryki, rozwiązanie w postaci zamkniętej i
  pętla treningowa działają (notebook wykonuje się od danej komórki do
  końca bez błędu).
- Strata modelu trenowanego gradientowo jest bliska stracie rozwiązania w
  postaci zamkniętej (różnica rzędu kilku procent, nie wielokrotności).
- Metryki MSE/$R^2$ raportowane w sekcji 7 są liczone na zbiorze
  **walidacyjnym**, nie treningowym.
- Interpretacja wykresu reszt poprawnie identyfikuje, która cecha ma
  nieliniowy wpływ, i wiąże to z ograniczeniem modelu liniowego (nie z
  "błędem w danych" czy "zbyt małym $N$").
- Odpowiedzi we wnioskach są uzasadnione, nie jednowersowe.

Pełne rozwiązanie wzorcowe jest przechowywane lokalnie przez prowadzącego
(`decisions/solutions/01_intro/`) i nie jest częścią tego repozytorium.
