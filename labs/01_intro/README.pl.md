# Laboratorium 01: podstawy PyTorch — tensory, urządzenia, pierwszy model

Czas: ok. 2 godziny. Bazuje na mechanizmie z [`lectures/01_intro/lecture.md`](../../lectures/01_intro/lecture.md)
(regresja liniowa, gradient, pętla treningowa), ale **nie jest tym samym
eksperymentem** — tu uogólniasz go na wiele cech i trafiasz na sytuację,
w której model liniowy *nie wystarcza*.

## 1. Problem

Na wykładzie trenowaliśmy model jednej zmiennej: $\hat{y} = wx + b$. W
praktyce dane rzadko mają jedną cechę. W tym laboratorium zbudujesz
regresję liniową **wielu zmiennych** w postaci macierzowej:

$$\hat{y} = Xw + b$$

gdzie $X$ ma kształt $(N, 3)$ (trzy cechy), a $w$ — kształt $(3,)$.

Zadanie: przewidzieć ciągłą zmienną $y$ na podstawie trzech cech $x_1, x_2,
x_3$. Jedna z tych cech wejdzie do prawdziwego procesu generującego dane w
sposób **nieliniowy** — Twoim zadaniem będzie nie tylko wytrenować model,
ale **wykryć i zinterpretować**, że model liniowy ma w tym miejscu
systematyczny błąd.

## 2. Dataset

Dane są syntetyczne i generowane lokalnie w notebooku (`lab.ipynb`,
sekcja 1) — nie pobierasz niczego. Kod generujący dane jest gotowy i
podany: to nie jest część zadania.

## 3. Metryki i podział danych — Twoje zadanie

Zaraz po wygenerowaniu danych notebook prosi Cię o dwie funkcje, które
będą używane przez całą resztę laboratorium (TODO, sekcja 2):

- funkcję dzielącą indeksy przykładów na zbiór treningowy i walidacyjny
  (ok. 80/20),
- metryki **MSE** (błąd średniokwadratowy) i **$R^2$**
  ($1 - SS_{\text{res}}/SS_{\text{tot}}$).

Robimy to na początku, bo baseline, trening i ewaluacja — wszystko dalej —
korzysta z tych samych dwóch narzędzi.

## 4. Baseline

Zanim trenujemy jakikolwiek model, potrzebujemy punktu odniesienia. W
notebooku dostajesz gotowy baseline: predykcja = średnia z $y$ ze zbioru
treningowego. To jest "model", który nie patrzy na $x$ wcale — każdy
sensowny model musi go pobić. Notebook liczy jego MSE/$R^2$ Twoimi
funkcjami z sekcji 3.

## 5. Model i trening — Twoje zadanie

W dalszych komórkach notebooka uzupełniasz (`# TODO`, podniesie
`NotImplementedError`, dopóki nie uzupełnisz):

1. **`forward(X, w, b)`** — predykcję modelu liniowego w postaci
   macierzowej (uogólnienie `w * x + b` z wykładu na iloczyn macierzowy).
2. **Rozwiązanie w postaci zamkniętej** (*normal equation*) —
   $w^\* = (X^\top X)^{-1} X^\top y$ (z kolumną jedynek dla biasu) — jako
   niezależny od gradientu punkt odniesienia dla Twojego modelu trenowanego
   gradientowo.
3. **Pętlę treningową metodą spadku gradientu** — uogólnienie pętli z
   wykładu na wektorowe $w$ (nadal bez `nn.Module`/`optim`: piszesz
   `forward → strata → backward → aktualizacja` sam).

Dostarczony kod (generowanie danych, device handling, baseline, wykresy)
działa od razu — nie musisz (i nie powinieneś) go zmieniać.

## 6. Eksperyment

Wytrenuj model i porównaj trzy liczby: stratę baseline'u, stratę modelu
wytrenowanego gradientowo i stratę rozwiązania w postaci zamkniętej. Strata
z gradientu powinna być **bliska** rozwiązaniu w postaci zamkniętej (to
jest problem wypukły — obie metody szukają tego samego optimum). Jeśli
różnica jest duża, to sygnał, że coś w Twojej implementacji pętli
treningowej (uczenie, krok aktualizacji, liczba epok) wymaga poprawy — nie
ignoruj tego rozjazdu.

## 7. Ewaluacja

Policz MSE i $R^2$ swojego wytrenowanego modelu na zbiorze
**walidacyjnym**, nie treningowym — raportowanie metryk na danych, które
model widział podczas treningu, nie mówi niczego o tym, jak model
generalizuje.

## 8. Analiza błędów — Twoje zadanie

Narysuj wykres reszt (*residuals*, $\hat{y} - y$) w funkcji każdej z trzech
cech na zbiorze walidacyjnym (kod do wykresu jest gotowy). Jedna z tych
trzech cech pokaże **systematyczny, nieprzypadkowy wzorzec** w rozkładzie
reszt — nie losowy szum wokół zera.

**Zadanie (komórka Markdown, 3–5 zdań):** zidentyfikuj, która cecha to jest,
opisz kształt wzorca i wyjaśnij — odwołując się do tego, czego model
liniowy *strukturalnie* nie jest w stanie reprezentować — dlaczego taki
wzorzec się pojawia. Nie chodzi o zgadnięcie "prawdziwego wzoru" (nie
zobaczysz go wprost w zadaniu), a o poprawne rozpoznanie *rodzaju* problemu
(czego liniowy model nie może wyrazić) z samego wykresu reszt.

## 9. Wnioski — Twoje zadanie

Notebook kończy się krótką sekcją, w której odpowiadasz (markdown) na dwa
pytania:

- Czy ten sam problem dałoby się rozwiązać lepiej, dodając nową *cechę*
  (np. kwadrat jednej z istniejących) bez zmiany samego modelu liniowego?
  Uzasadnij.
- Dlaczego porównanie z rozwiązaniem w postaci zamkniętej jest użytecznym
  testem poprawności Twojej pętli treningowej?

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

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/01_intro/`) i nie jest częścią tego repozytorium.
