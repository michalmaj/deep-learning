Język: [English](lecture.en.md) | **Polski**

# Wykład 14 — Jak oceniać i interpretować modele, którym chcemy zaufać?

## Cele wykładu

1. Przejść od pojedynczej liczby (accuracy) do systematycznego audytu błędów
   modelu klasyfikującego.
2. Rozumieć precision, recall, F1, macro/micro-averaging i rolę nierównowagi
   klas — oraz umieć je policzyć ręcznie i zweryfikować programowo.
3. Rozumieć różnicę między confidence modelu a rzeczywistą wiarygodnością
   predykcji — i co mierzy (i czego nie mierzy) kalibracja.
4. Umieć wyprowodzić i samodzielnie zaimplementować Grad-CAM, z pełną
   kontrolą wymiarów i poprawności gradientów.
5. Rozumieć, że Grad-CAM (i attribution w ogólności) pokazuje wrażliwość, nie
   dowód przyczynowy — i wiedzieć, jak to sprawdzić (occlusion, sanity checks).
6. Przygotować systematyczny raport analizy błędów, z którego wynika
   konkretna rekomendacja inżynierska.

## Dlaczego to jest ważne

Od modułu 6 trenujemy modele i mierzymy accuracy. Ale accuracy jest **jedną**
liczbą, zagregowaną po całym zbiorze — ukrywa, **które** klasy/przykłady
model zawodzi, **jak bardzo** jest pewny swoich błędów, i **czy** jego
"spojrzenie" na dane (attribution) ma sens. Ten wykład nie wprowadza nowej
architektury — wprowadza **dyscyplinę**: jak zadawać modelowi pytania, na
które accuracy nie odpowiada, i jak nie oszukać się własnymi narzędziami
interpretowalności.

Trzy doświadczenia z wcześniejszych modułów są tu bezpośrednio przydatne:
model z większą liczbą parametrów nie musi generalizować lepiej (moduł 5);
model generatywny może mieć dobrą metrykę, mimo że prosty baseline jest
równie skuteczny (moduł 12 — AE vs trywialny baseline na detekcji anomalii);
pojedynczy wynik nie dowodzi ogólnego twierdzenia (moduł 13 — jeden przebieg
MMD nie decyduje, jedna heatmapa nie decyduje).

## Ewaluacja jako część procesu inżynierskiego

**Train / validation / test — rola każdego podziału:**

- **Train**: uczenie parametrów $\theta$.
- **Validation**: wybór konfiguracji (hiperparametry, próg decyzyjny,
  wczesne zatrzymanie) — **nigdy** finalna ocena.
- **Test**: **jedna**, końcowa ocena wybranego już modelu — nigdy używany do
  decyzji o konfiguracji.

❌ "Mogę dobrać próg decyzyjny (albo hiperparametr, albo przykłady do analizy)
na zbiorze testowym, skoro i tak chcę zobaczyć, jak dobrze to działa."

✅ To jest **data leakage protokołu ewaluacyjnego** — używanie testu do
**jakiejkolwiek** decyzji przed finalnym raportem unieważnia jego rolę jako
niezależnego oszacowania. Zobaczymy to dosłownie w laboratorium (zadanie
diagnostyczne modułu 12 uczyło tego dla progu anomalii — tu ta sama zasada
dotyczy progu decyzyjnego, wyboru przykładów do analizy error i wyboru metody
interpretacji).

## Accuracy, precision, recall, F1 — matematyka

Notacja: $C$ = liczba klas (reużyte z modułu 07 — "liczba klas Twojego
zadania"; **inna** rola niż $C$ = liczba kanałów obrazu $(C,H,W)$, moduł 06
— nigdy w tym samym kontekście, ale explicit rozróżnienie: tu $C$ to
rozmiar confusion matrix, nie wymiar tensora).

**Przykład rozwiązany** (3 klasy, $N=20$ przykładów, celowo nierównomierne:
klasa 0 — 10, klasa 1 — 8, klasa 2 — 2):

Confusion matrix (wiersze = prawda, kolumny = predykcja):

| | pred 0 | pred 1 | pred 2 |
|---|---|---|---|
| **true 0** | 9 | 1 | 0 |
| **true 1** | 0 | 8 | 0 |
| **true 2** | 2 | 0 | 0 |

$$\text{accuracy} = \frac{\sum_c \text{TP}_c}{N} = \frac{9+8+0}{20} = 0{,}85$$

Dla klasy $c$: $\text{TP}_c$ = diagonalna; $\text{FP}_c$ = suma kolumny $c$
minus $\text{TP}_c$; $\text{FN}_c$ = suma wiersza $c$ minus $\text{TP}_c$.

$$\text{precision}_c = \frac{\text{TP}_c}{\text{TP}_c+\text{FP}_c}, \qquad \text{recall}_c = \frac{\text{TP}_c}{\text{TP}_c+\text{FN}_c}, \qquad F1_c = \frac{2\cdot\text{precision}_c\cdot\text{recall}_c}{\text{precision}_c+\text{recall}_c}$$

| klasa | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0 | 9 | 2 | 1 | 0,818 | 0,900 | 0,857 |
| 1 | 8 | 1 | 0 | 0,889 | 1,000 | 0,941 |
| 2 | 0 | 0 | 2 | **NaN** (0/0) | **0,000** | 0,000 |

Zweryfikowane programowo w notebooku — dokładnie te liczby.

❌ "Bardzo wysoka accuracy oznacza, że model działa dobrze na każdej klasie."

✅ **Nieprawda — ten przykład to pokazuje bezpośrednio.** Accuracy $=0{,}85$
(wysoka), ale klasa 2 (mniejszościowa) ma recall $=0$ — model **nigdy**
poprawnie nie rozpoznaje tej klasy, a wysoka accuracy to maskuje, bo klasa 2
to tylko $2/20=10\%$ przykładów.

**Macro vs micro averaging:**

$$\text{macro-F1} = \frac{1}{C}\sum_{c=1}^C F1_c = \frac{0{,}857+0{,}941+0{,}000}{3} = 0{,}599$$

Micro-F1 (dla klasyfikacji multi-class single-label) liczony jest z
**zsumowanych** TP/FP/FN po wszystkich klasach — co dla tego przypadku
(każdy przykład ma dokładnie jedną predykcję i jedną prawdziwą etykietę)
**redukuje się matematycznie do accuracy**:

$$\text{micro-F1} = \frac{\sum_c \text{TP}_c}{\sum_c \text{TP}_c + \sum_c \text{FP}_c} = \frac{\sum_c \text{TP}_c}{N} = \text{accuracy} = 0{,}85$$

(bo $\sum_c \text{FP}_c = \sum_c \text{FN}_c$ dla single-label multi-class —
każdy błąd jest FP dla jednej klasy i FN dla innej, więc sumy się znoszą).

❌ "Macro-F1 i micro-F1 to dwa równoważne sposoby zapisu tej samej liczby."

✅ **Nieprawda — ten przykład pokazuje różnicę $0{,}60$ vs $0{,}85$.** Macro-F1
**nie waży** klas liczebnością — każda klasa wnosi tyle samo, niezależnie od
tego, czy ma 2 czy 1000 przykładów. Micro-F1 (= accuracy tutaj) **waży**
niejawnie przez liczebność — duże klasy dominują wynik. Dla nierównowagi
klas, macro-F1 **ujawnia** problem klasy mniejszościowej, który accuracy/
micro-F1 **maskuje**. Weighted-F1 (waży F1 każdej klasy jej liczebnością)
to trzecia opcja, pośrednia — $0{,}805$ dla tego przykładu.

## Logity, softmax, argmax, ranking, próg — rozróżnienie

Łańcuch (rozszerzenie modułu 02/08/11): $z = f_\theta(x) \in \mathbb R^C$
(logity) $\to \hat y = \text{softmax}(z) \in [0,1]^C$ (prawdopodobieństwa,
sumują się do 1) $\to \hat c = \arg\max_c \hat y_c$ (predykowana etykieta).
To są **trzy różne obiekty**: logity nie są prawdopodobieństwami (mogą być
ujemne, nie sumują się do 1); argmax to **decyzja**, nie **miara pewności**;
ranking przykładów (np. po $\max_c \hat y_c$) to kolejna, osobna operacja,
używana przy ROC-AUC/PR-AUC.

❌ "ROC-AUC i PR-AUC stosuje się bezpośrednio do klasyfikatora wieloklasowego
tak, jak do binarnego, bez żadnej modyfikacji."

✅ **Nieprawda.** ROC-AUC/PR-AUC są zdefiniowane dla **binarnej** decyzji
(pozytyw/negatyw) w funkcji progu. Dla $C>2$ klas, stosuje się je **per
klasa**, w ramach schematu *one-vs-rest* (klasa $c$ vs "wszystkie inne"), nie
jako jedna, natywnie wieloklasowa krzywa. Moduł 12's lab już zbudował pełną
maszynerię ROC-AUC/PR-AUC (ręcznie, bez `scikit-learn`) dla **binarnego**
zadania detekcji anomalii — ten sam kod działałby tu, zastosowany $C$ razy
(raz na klasę), nie rozwijamy tego pełnego omówienia tutaj (poza rdzeniem
tego wykładu).

**Próg decyzyjny i koszty błędów.** Przy binarnej decyzji (np. "czy
to klasa X"), wybór progu zmienia **balans** między false positives i false
negatives — jak w module 12's labie (próg wybrany **tylko** z danych
walidacyjnych, nigdy testowych). Koszt FP i FN **nie musi być symetryczny**
— zależy od zastosowania (inżynierska decyzja, nie statystyczna).

## Kalibracja

❌ "Jeśli softmax dla predykcji wynosi $0{,}95$, to ta KONKRETNA predykcja
jest poprawna z prawdopodobieństwem 95%."

✅ **Nieprawda — to błąd kategorii.** Softmax $0{,}95$ to wyjście **jednego**
forward passu dla **jednego** przykładu — nie jest to częstość empiryczna
ani gwarancja. **Kalibracja** jest własnością **populacji** predykcji, nie
jednej predykcji: model jest skalibrowany, jeśli **wśród wszystkich**
predykcji z confidence $\approx p$, **rzeczywiście** ok. $p$ fraction jest
poprawna.

**Confidence** = $\max_c \hat y_c$ (reużycie już ustalonego $\hat y=
\text{softmax}(z)$, moduł 08 — tu: maksymalny wpis wektora). **Reliability
diagram**: grupujemy predykcje w biny confidence $B_m$ (**uwaga o
notacji**: koliduje z $B_t$ = mini-batch, moduł 04 — różny indeks (m=bin
confidence vs t=krok treningowy), różny koncept, zbyt standardowe w
literaturze kalibracji, by zmieniać), dla każdego binu liczymy **średnią
accuracy** i **średnią confidence**; skalibrowany model ma punkty bliskie
przekątnej (acc≈conf).

**Expected Calibration Error:**

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \big|\text{acc}(B_m) - \text{conf}(B_m)\big|$$

**Przykład liczbowy** (10 predykcji, 2 biny): 4 predykcje confidence$=0{,}95$
(3 poprawne, 1 błędna → acc$=0{,}75$); 6 predykcji confidence$=0{,}55$ (2
poprawne, 4 błędne → acc$=0{,}3333$).

$$\text{ECE} = \frac{4}{10}|0{,}75-0{,}95| + \frac{6}{10}|0{,}3333-0{,}55| = 0{,}08+0{,}13 = 0{,}21$$

Zweryfikowane programowo — dokładnie $0{,}21$.

❌ "ECE jest obiektywną, jednoznaczną miarą — nie zależy od tego, jak ją
policzysz."

✅ **Nieprawda — zmierzone, nie tylko zadeklarowane.** Na syntetycznym
przykładzie z ciągłym rozkładem confidence: ECE stabilne ($\approx 0{,}157$)
dla liczby binów $5,10,15,20$, ale **skacze** do $0{,}175$ dla $50$ binów
(rzadkie biny stają się szumowe — niewiele przykładów na bin, średnia
niestabilna). Ta sama dystrybucja, **różny rozmiar próby**: ECE
$0{,}193\to0{,}164\to0{,}118\to0{,}157$ dla $n=20,50,100,200$ — wyraźna
zależność od liczności próby, nie jedna, stabilna liczba.

❌ "Model może mieć wysoką accuracy i automatycznie być dobrze skalibrowany
(albo: musi być źle skalibrowany, bo sieci neuronowe są znane z
nadmiernej pewności)."

✅ **Żadne z tych uogólnień nie jest bezpieczne — trzeba zmierzyć.** Model z
tego wykładu (Eksperyment B) jest **dobrze** skalibrowany (ECE$\approx 0{,}01$)
— nie wymuszamy tu narracji o nadmiernej pewności tylko dlatego, że to
częstszy wynik w literaturze. **Ale** nawet dobra kalibracja **populacyjna**
nie gwarantuje, że **każda** pojedyncza predykcja jest wiarygodna — ten sam
model miał błędne predykcje z confidence do $0{,}9996$ (patrz "Co
obserwujemy"). To są **dwa różne poziomy** oceny (populacja vs przykład),
nie jeden wniosek.

(Temperature scaling — technika kalibracji logitów **po** treningu,
dzielenia $z$ przez skalar $T_{\text{cal}}$ przed softmax, dobierany na
walidacji — **nota o notacji**: to **inna** temperatura niż $\tau$ z modułu
11 (temperatura **samplingu**, zmienia losowość generowania tekstu) — tu
temperatura **kalibracji**, zmienia ostrość softmax dla **oceny
wiarygodności**, nie dla generowania. Nie rozwijamy pełnej implementacji w
rdzeniu tego wykładu — rozszerzenie, jeśli zostanie czas.)

## Grad-CAM: matematyka

**Najważniejsza część interpretowalności tego wykładu.** Kontekst: CNN z
wybraną warstwą konwolucyjną, dla której chcemy zrozumieć, na jakie regiony
wejścia "reaguje" predykcja konkretnej klasy.

**Notacja.** $A^k \in \mathbb R^{H\times W}$ — mapa aktywacji **kanału $k$**
wybranej warstwy (**uwaga o notacji**: zapis **bez nawiasów**, w
odróżnieniu od $A^{(l)}$, moduł 03 — tam nawiasy oznaczają **ogólny indeks
warstwy** dla uśrednionej, batched reprezentacji; tu $A^k$ to **konkretna**
mapa **jednego, wybranego** layera, indeksowana **kanałem**, nie warstwą —
inny obiekt matematyczny). $z_c$ — logit klasy $c$ (reużycie $z$=logity,
moduł 02+, z indeksem klasy — **nie** $y^c$, nowy symbol byłby tu
niepotrzebny, i **nie** softmax $\hat y_c$ — to jest istotne rozróżnienie,
nie stylistyczne (wyjaśnienie poniżej).

**Dlaczego logit, nie softmax.** Softmax $\hat y_c = \exp(z_c) /
\sum_{c'} \exp(z_{c'})$ normalizuje przez sumę po **wszystkich** klasach —
gradient $\partial \hat y_c/\partial A^k$ więc niesie w sobie wpływ
**wszystkich** innych logitów (przez mianownik), nie tylko klasy $c$.
Gradient logitu $\partial z_c/\partial A^k$ "widzi" tylko klasę $c$,
bez tej domieszki. Dodatkowo, gdy softmax jest bliski $0$ albo $1$
(model bardzo pewny), jego gradient **saturuje** blisko zera — Grad-CAM
liczony na softmaksie traciłby sygnał właśnie tam, gdzie predykcja jest
najbardziej jednoznaczna. Z tych dwóch powodów (zanieczyszczenie przez
inne klasy, saturacja) oryginalna praca Grad-CAM (Selvaraju i in., 2017)
celuje w logit $z_c$, nie w $\hat y_c$.

**Krok 1 — gradient wyniku względem map aktywacji:**

$$\frac{\partial z_c}{\partial A^k} \in \mathbb R^{H\times W}$$

**Krok 2 — wagi kanałów przez globalne uśrednienie gradientu (global average
pooling po wymiarach przestrzennych):**

$$\alpha_k^c = \frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W \frac{\partial z_c}{\partial A^k_{ij}}$$

(**uwaga o notacji**: to **czwarte** znaczenie $\alpha$ w tym kursie —
wagi attention $\alpha_{ij}$, moduł 10; interpolacja VAE, bare $\alpha$,
moduł 12; harmonogram dyfuzji $\alpha_t$, moduł 13 — tu $\alpha_k^c$,
dwupoziomowy indeks sub+superscript, niepowtórzony wcześniej. Zbyt
standardowe w pracy Grad-CAM, by zmieniać nazwę.)

**Krok 3 — ważona suma map aktywacji + ReLU:**

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right) \in \mathbb R^{H\times W}$$

ReLU **tutaj** (na ważonej sumie, nie na pojedynczych $A^k$) odzwierciedla,
że interesują nas tylko regiony o **pozytywnym** wpływie na $z_c$ —
negatywny wpływ (region "przeciw" klasie $c$) jest ucinany do zera, nie
zachowany jako wartość ujemna.

**Krok 4 — przeskalowanie do rozdzielczości obrazu** (interpolacja, zwykle
bilinearna, z $(H,W)$ warstwy do $(H_{\text{img}}, W_{\text{img}})$ obrazu
wejściowego) i normalizacja do $[0,1]$ (z obsługą przypadku zerowej
rozpiętości — niżej).

### Mały przykład liczbowy

Dwa kanały, mapa $2\times 2$ każdy. $A^0 = \begin{pmatrix}1&2\\3&4\end{pmatrix}$,
$A^1=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ — **hipotetyczne, ręcznie
wybrane** wartości, do zilustrowania wzoru, nie wynik żadnego treningu.
Przyjmijmy równie hipotetyczny, ręcznie wybrany gradient
$\partial z_c/\partial A^k$: dla kanału 0,
$\begin{pmatrix}-0{,}2&-0{,}2\\-0{,}2&0{,}2\end{pmatrix}$, dla kanału 1,
$\begin{pmatrix}0{,}2&0{,}4\\0{,}4&0{,}2\end{pmatrix}$.

Krok 2 (globalne uśrednienie, $1/(HW)=1/4$):
$\alpha_0^c = (-0{,}2-0{,}2-0{,}2+0{,}2)/4 = -0{,}1$,
$\alpha_1^c = (0{,}2+0{,}4+0{,}4+0{,}2)/4 = 0{,}3$ — zauważ, że $\alpha_k^c$
**może być ujemne** (kanał "argumentujący przeciw" klasie $c$, nie tylko
"za" nią).

Ważona suma: $\alpha_0^c A^0 + \alpha_1^c A^1 =
-0{,}1\begin{pmatrix}1&2\\3&4\end{pmatrix} +
0{,}3\begin{pmatrix}0&1\\1&0\end{pmatrix} =
\begin{pmatrix}-0{,}1&0{,}1\\0{,}0&-0{,}4\end{pmatrix}$. Po ReLU:
$\begin{pmatrix}0&0{,}1\\0{,}0&0\end{pmatrix}$ — tylko jedna komórka (prawy
górny róg) ma pozytywny wpływ na $z_c$ po uwzględnieniu obu kanałów; resztę
ReLU zeruje.

Cały ten przykład jest **w pełni ręcznie policzalny** krok po kroku z samej
definicji $\alpha_k^c$ podanej wyżej — nie pochodzi z żadnego konkretnego
treningu czy notebooka. Niezależnie od niego, zgodność praktycznej,
hook-based implementacji (forward/backward hooks) z niezależnym, ręcznym
obliczeniem przez `retain_grad()` jest sprawdzona w zestawie testów
(`tests/test_lecture_14_notebook.py`, nie w tym notebooku, na osobnej, małej,
losowo zainicjalizowanej sieci) — **w granicach tolerancji numerycznej**
(`atol=1e-5`), nie jako identyczność bitowa.

### Trzy różne cele — trzy różne pytania

❌ "Grad-CAM pokazuje jedną, uniwersalną mapę 'tego, na co model patrzy'."

✅ **Nieprawda — wybór $c$ zmienia pytanie.** Grad-CAM dla:

1. **klasy przewidzianej** $\hat c=\arg\max_c z_c$ — "co wsparło **tę**
   decyzję?"
2. **klasy rzeczywistej** $y$ (jeśli $y\neq\hat c$) — "co **by** wsparło
   poprawną odpowiedź, której model nie wybrał?"
3. **dowolnej innej klasy** wskazanej przez analizującego — "co wsparłoby
   **tę konkretną, hipotetyczną** decyzję?"

To są **trzy różne mapy dla tego samego obrazu** — używanie jednej zamiast
drugiej bez uzasadnienia to mylenie pytań.

### Obsługa mapy o zerowej rozpiętości

Jeśli ważona suma $\sum_k \alpha_k^c A^k$ jest **niepozytywna wszędzie**,
ReLU zeruje **całą** mapę — normalizacja $(x-\min)/(\max-\min)$ dzieliłaby
przez $\approx 0$. Poprawna obsługa: sprawdzić $(\max-\min)>\epsilon$ przed
dzieleniem, inaczej zwrócić mapę zer (zweryfikowane programowo — nie
teoretyczny edge case, patrz "Co obserwujemy": to się zdarza **często**,
nie tylko w teorii).

## Granice interpretacji

- Grad-CAM pokazuje **określoną formę wrażliwości** predykcji na aktywacje
  **wybranej** warstwy — nie "to, co model myśli" w ogólności.
- Heatmapa **nie jest** maską segmentacji — niska rozdzielczość warstwy
  konwolucyjnej (np. $7\times7$ przed interpolacją do $28\times28$ albo
  $224\times224$) oznacza, że **żadna** interpolacja nie odzyskuje precyzji
  co do piksela, której tam nie było.
- Wysoka wartość attribution **nie dowodzi**, że model rozpoznaje
  semantycznie poprawny obiekt — tylko że ten region **koreluje** z wyższym
  $z_c$ przy tej **konkretnej** parametryzacji gradientu.
- Zgodność mapy z ludzką intuicją **nie gwarantuje** poprawności wyjaśnienia
  — zgodna z intuicją mapa może wynikać z przypadkowej korelacji, nie z
  mechanizmu, który faktycznie napędza decyzję.
- Metoda **zależy** od architektury, wybranej warstwy, wybranej klasy i
  sposobu liczenia gradientów — zmiana każdego z tych elementów może dać
  inną mapę dla **tego samego** obrazu.
- Duże/wygładzone obszary (typowe po interpolacji z małej mapy) **nie
  oznaczają** precyzyjnej lokalizacji cechy.

## Occlusion sensitivity: niezależna weryfikacja przez perturbację

Zamiast gradientu, **fizycznie** zasłaniamy fragment obrazu (zastępując go
sensowną wartością bazową, zgodną z preprocessingiem — np. znormalizowany
"szary" piksel dla modelu z ImageNet-normalizacją, nie surowe zero) i
mierzymy **zmianę** logitu $z_c$ analizowanej klasy. Wynik: heatmapa "ile
stracił logit, gdy zasłoniliśmy ten region" — koncepcyjnie inny sygnał niż
gradient (pyta "co się stanie, jeśli to usunę", nie "jak zmienia się wynik
lokalnie przy infinitezymalnej zmianie aktywacji").

❌ "Occlusion sensitivity jest metodą bez ograniczeń, bo nie używa
gradientów."

✅ **Ma własne, inne ograniczenia.** Maskowanie **tworzy** zmodyfikowane
wejścia, które mogą **drastycznie** odbiegać od naturalnego rozkładu danych
treningowych (duży, jednolity szary kwadrat na zdjęciu satelitarnym to coś,
czego model nigdy nie widział podczas treningu) — zmiana wyniku może
odzwierciedlać reakcję na **nienaturalne** wejście, nie czystą "ważność"
zasłonionego regionu.

## Sanity checks dla saliency maps

❌ "Metoda attribution, która wygląda sensownie, jest tym samym wiarygodna."

✅ **Trzeba to zweryfikować, nie zakładać.** Jeden test: policz Grad-CAM dla
wytrenowanego modelu, **następnie** zresetuj losowo wybrane wagi (np.
klasyfikator końcowy) i policz Grad-CAM **ponownie** dla tego samego obrazu
i tej samej klasy. Jeśli metoda jest **wrażliwa na model** (jak powinna
być), mapa powinna się **zmienić** — jeśli nie zmienia się wcale,
attribution może nie odzwierciedlać faktycznie wyuczonych parametrów, tylko
coś strukturalnego (np. sam kształt obrazu). Nie zakładamy z góry, w jakim
stopniu zmiana nastąpi — **mierzymy** (np. średnia różnica absolutna między
mapami, nie korelacja — korelacja jest niezdefiniowana, gdy jedna z map jest
stała/zerowa, co się zdarza, patrz "obsługa mapy o zerowej rozpiętości").

## `model.eval()` vs `torch.no_grad()` vs `torch.inference_mode()`

❌ "Grad-CAM można policzyć w tym samym trybie co zwykłą inferencję bez
gradientów — wystarczy `model.eval()`."

✅ **Nieprawda, zmierzony kontrast:**

- `model.eval()` **samo** nie wpływa na śledzenie gradientu — wpływa tylko
  na zachowanie `BatchNorm`/`Dropout`. `backward()` działa normalnie.
- `torch.no_grad()`: `backward()` **rzuca** `RuntimeError` ("element 0 of
  tensors does not require grad and does not have a grad_fn") —
  Grad-CAM **wymaga** aktywnego grafu gradientów.
- `torch.inference_mode()`: ten sam błąd **wewnątrz** kontekstu, ale
  dodatkowo tensory stworzone tam **nie mogą** nawet dostać
  `requires_grad_(True)` **poza** kontekstem ("Setting requires_grad=True on
  inference tensor outside InferenceMode is not allowed") — silniejsze,
  bardziej "zaraźliwe" ograniczenie niż `no_grad()`.

**Subtelniejszy, realnie odkryty przypadek**: dla modelu z **zamrożonym**
backbone'em (wszystkie wagi `requires_grad=False`, tylko głowica trenowana
— jak w laboratorium), Grad-CAM na warstwie backbone'u **failuje** (gradient
= `None`) **nawet bez** `no_grad()`, **jeśli** wejściowy tensor $x$ też nie
ma `requires_grad=True` — graf autogradu nigdy nie powstaje wzdłuż
zamrożonych wag, gdy nic "wcześniej" nie wymaga gradientu. Fix: `x =
x.requires_grad_(True)` **na wejściu** (nie na wagach backbone'u) — nie
zmienia, czego się uczymy, tylko zapewnia, że graf gradientów faktycznie
istnieje do hooka.

## Co obserwujemy

Wszystkie liczby pochodzą z rzeczywistego wykonania `examples.ipynb` (CPU,
ten komputer) — Fashion-MNIST, podzbiór 8000 train / 2000 test, `SmallCNN`
z modułu 06 (z poprawką: osobne instancje `relu1`/`relu2`, niezbędne dla
poprawnego działania hooków Grad-CAM — moduł 06 nigdy nie potrzebował
hooków, więc oryginalna wersja z jedną współdzieloną instancją nigdy nie
objawiała tego problemu tam).

- **Eksperyment A.** Trening (10 epok, Adam, CrossEntropyLoss): test
  accuracy $=0{,}8745$, macro-F1 $=0{,}8765$ (zbliżone — klasy w tym
  podzbiorze są w przybliżeniu zbalansowane, inaczej niż w ręcznym
  3-klasowym przykładzie wyżej). Per-class F1 w zakresie $0{,}70$–$0{,}97$ —
  realna zmienność, nie jedna liczba. Najczęściej mylona para: **Coat→
  Pullover** (34 przypadki), druga: **T-shirt/top→Shirt** (22) — obie to
  znane, wizualnie uzasadnione konfuzje tego zbioru, nie przypadek.
- **Eksperyment B.** Mean confidence: poprawne $=0{,}9082$, błędne
  $=0{,}6517$ — rozkłady różne, jak oczekiwano. ECE $=0{,}0099$ — **model
  jest dobrze skalibrowany**, piszemy to wprost, nie wymuszamy narracji o
  nadmiernej pewności. Mimo to: **29** błędnych predykcji miały confidence
  $>0{,}9$, maksymalnie $0{,}9996$ — dobra kalibracja **populacyjna** nie
  eliminuje pojedynczych, bardzo pewnych błędów.
- **Eksperyment C.** Zero-map rate (relu2, 32 kanały): **ok. 65%**
  predykcji dało CAM o zerowej rozpiętości (ważona suma ujemna wszędzie) —
  **nieskorelowane** z confidence ($0{,}854$ vs $0{,}849$ confidence,
  degenerate vs non-degenerate). Wybrane, non-degenerate przykłady:
  poprawna predykcja (idx=2, Trouser→Trouser, conf=1,0000); pomyłka
  (idx=12, Sneaker→Bag, conf=0,4602); myląca wysoka-confidence pomyłka
  (idx=23, Ankle boot→Sandal, conf=0,9996 — **to jest** ten sam przykład z
  "max confidence wśród błędnych predykcji" z Eksperymentu B).
- **Eksperyment D.** Occlusion sensitivity (patch 4×4, stride 4, baseline=0,
  zgodny z zakresem $[0,1]$ `ToTensor()`): dla przykładu poprawnego (idx=2,
  Trouser), base_logit$=9{,}644$, zakres zmiany logitu po zasłonięciu
  patchy: $[-0{,}975,\ 5{,}369]$ — regiony o dużym spadku logitu **częściowo**
  pokrywają się z regionami Grad-CAM, nie identycznie (różne pytania, patrz
  wyżej). Czas: $0{,}007$s na obraz (49 patchy) — pomijalny.
- **Eksperyment E.** Sanity check (losowy reset wag `fc`, ten sam obraz i
  klasa): **mean absolute difference** między CAM przed/po $=0{,}1384$ —
  realna, zmierzona zmiana, mimo że conv-features są niezmienione (tylko
  końcowa klasyfikacja zmieniona losowo). Korelacja jako metryka **byłaby**
  niezdefiniowana (`NaN`), gdyby trafiono na jeden z degenerate (zerowych)
  CAM-ów z ok. 65% przypadków zidentyfikowanych w Eksperymencie C — dlatego
  mean absolute difference (odporne na stałe/zerowe mapy) jest tu **lepszym**
  domyślnym wyborem niż korelacja, zweryfikowane przez samą częstość
  degeneracji, nie tylko teoretyczną możliwość.

## Częste błędne przekonania

(Zbiorcze podsumowanie — każde ma pełny ❌/✅ blok wyżej)

1. Wysoka accuracy nie oznacza dobrego działania na każdej klasie.
2. Wysoki confidence nie jest gwarancją poprawności pojedynczej predykcji.
3. Model może być źle skalibrowany mimo wysokiej accuracy — i odwrotnie,
   dobrze skalibrowany model wciąż ma pewne, błędne predykcje.
4. ECE zależy od liczby binów i rozmiaru próby — zmierzone, nie teoretyczne.
5. Confusion matrix nie wyjaśnia sama z siebie przyczyn pomyłek.
6. Grad-CAM nie jest segmentacją ani ground truth.
7. Grad-CAM nie jest dowodem przyczynowego wyjaśnienia predykcji.
8. Ładna wizualizacja nie oznacza, że metoda interpretowalności jest
   wiarygodna — trzeba sanity-checkować.
9. Wagi attention (moduł 10) nie są zamiennikiem Grad-CAM i nie stanowią
   uniwersalnego wyjaśnienia modelu — różne metody, różne założenia.
10. Wyniki testowe nie powinny służyć do strojenia modelu/progu/metody.
11. Jeden przykład błędu nie dowodzi systematycznej słabości modelu.
12. Odporność na pojedynczą transformację nie oznacza odporności na
    rzeczywisty distribution shift.
13. Dobry wynik offline nie gwarantuje równie dobrego działania po
    wdrożeniu (rozwinięte w module 15).
14. Grad-CAM wymaga aktywnego grafu gradientów — `model.eval()` samo nie
    wystarcza, a zamrożony backbone bez `requires_grad` na wejściu failuje
    inaczej niż `torch.no_grad()`, ale z tego samego powodu.

## Implikacje inżynierskie

- **Jakość ogólna vs jakość per-class** to dwie różne decyzje — model
  "dobry ogólnie" może być nieakceptowalny dla jednej, ważnej klasy.
- **Koszty FP/FN są decyzją biznesową**, nie statystyczną — próg dobrany
  "optymalnie" dla jednej metryki może być źle dobrany dla rzeczywistego
  kosztu błędu.
- **Kalibracja** ma praktyczne znaczenie tam, gdzie confidence jest
  używane do decyzji (np. "automatycznie akceptuj tylko predykcje >0,95") —
  źle skalibrowany model podejmuje te decyzje na złej podstawie.
- **Przegląd trudnych przykładów** (wybranych jawną, nie-cherry-picked
  procedurą) jest częścią procesu, nie jednorazową ciekawostką.
- **Ograniczenia metod interpretowalności muszą być dokumentowane** —
  "ładna heatmapa" bez sanity check to niewystarczające uzasadnienie decyzji.
- **Powtarzalność raportów ewaluacyjnych** — ten sam podział danych, te same
  metryki, żeby porównania między wersjami modelu były uczciwe.
- **Ryzyko selektywnego pokazywania dobrych przypadków** — raport musi
  pokazywać reprezentatywny obraz, nie tylko efektowne przykłady.
- **Dokumentowanie znanych ograniczeń modelu** (model cards, krótko) —
  część odpowiedzialnego wdrożenia, nie opcjonalny dodatek.
- Ewaluacja **przed** wdrożeniem (ten moduł) i monitorowanie **po**
  wdrożeniu (moduł 15) są powiązane, ale **to nie ten sam proces** — dobry
  wynik offline nie gwarantuje braku potrzeby monitorowania.

## Co warto zapamiętać

- Train/validation/test mają ściśle różne role — test nigdy nie wpływa na
  żadną decyzję przed finalnym raportem.
- Accuracy ukrywa per-class słabości; macro-F1 je ujawnia (nie waży klas
  liczebnością), micro-F1 = accuracy dla multi-class single-label.
- Confidence ≠ wiarygodność jednej predykcji; kalibracja to własność
  **populacji** predykcji (ECE), nie gwarancja dla każdego przykładu.
- Grad-CAM: $\alpha_k^c$ = GAP gradientu $z_c$ po $A^k$; mapa = ReLU(ważona
  suma); wymaga **aktywnego grafu gradientów**, nie tylko `eval()`.
- Trzy różne cele Grad-CAM (predykowana/rzeczywista/dowolna klasa) = trzy
  różne pytania.
- Occlusion sensitivity weryfikuje Grad-CAM **innym** rodzajem sygnału
  (perturbacja, nie gradient) — zgodność nie jest gwarantowana.
- Sanity checks (np. losowy reset wag) sprawdzają, czy metoda w ogóle
  reaguje na model, nie tylko na kształt wejścia.

## Pytania dla studentów

1. Masz klasyfikator 5-klasowy z accuracy $0{,}92$. Jakie dodatkowe liczby
   poprosiłbyś/abyś o zobaczenie, zanim uznasz model za "dobry"?
2. Dlaczego micro-F1 dla multi-class single-label equals accuracy, a dla
   multi-label (więcej niż jedna prawdziwa etykieta per przykład) już nie?
3. Masz dwa modele o identycznym ECE. Czy to znaczy, że są "tak samo
   wiarygodne" dla pojedynczych predykcji? Uzasadnij.
4. Wyjaśnij, dlaczego Grad-CAM dla klasy przewidzianej i dla klasy
   rzeczywistej mogą wskazywać **różne** regiony obrazu, nawet dla tego
   samego, błędnie klasyfikowanego przykładu.
5. Masz Grad-CAM i occlusion sensitivity, które **nie zgadzają się** co do
   najważniejszego regionu obrazu. Jak zdecydowałbyś/łabyś, co to znaczy —
   i czego **nie** możesz z tego wnioskować?

## Bibliografia

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
  Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*. ICCV. arXiv:1610.02391 (praca źródłowa
  Grad-CAM).
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B.
  (2018). *Sanity Checks for Saliency Maps*. NeurIPS. arXiv:1810.03292
  (randomizacja wag/etykiet jako test wrażliwości metod attribution).
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration
  of Modern Neural Networks*. ICML. arXiv:1706.04599 (ECE, reliability
  diagram, temperature scaling).
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST*.
  arXiv:1708.07747 (zobacz też `data/registry.yml`).
- PyTorch documentation: *Autograd mechanics* (`torch.no_grad`,
  `torch.inference_mode`), `https://pytorch.org/docs/stable/notes/autograd.html`.
