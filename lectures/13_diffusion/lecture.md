# Wykład 13 — Modele dyfuzyjne: uczenie generowania przez odszumianie

## Cele wykładu

1. Zrozumieć forward diffusion jako kontrolowany, matematycznie precyzyjny
   proces dodawania szumu — nie jako coś, czego model musi się uczyć.
2. Umieć wyprowadzić i policzyć zamkniętą postać $x_t = \sqrt{\bar\alpha_t}
   x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ dla zadanego $x_0$, $t$, $\varepsilon$.
3. Rozumieć, że model przewiduje **szum**, nie bezpośrednio dane — i jak ta
   predykcja jest wykorzystywana w reverse diffusion.
4. Umieć samodzielnie zaimplementować mały DDPM: harmonogram, trening,
   próbkowanie odwrotne.
5. Obserwować powstawanie struktury danych z czystego szumu, nie
   rekonstrukcję podanej próbki.
6. Zbadać wpływ harmonogramu szumu, liczby kroków i jakości predykcji szumu
   na jakość generowania — **zmierzony**, nie zadeklarowany.
7. Krytycznie oceniać jakość generatora — nie tylko "czy wykres wygląda
   dobrze".
8. Porównać dyfuzję z VAE (moduł 12) i generowaniem autoregresyjnym
   (moduł 11) bez ogłaszania jednego podejścia zwycięzcą.

## Dlaczego to jest ważne

Moduł 11 uczył generowania **sekwencyjnego, token po tokenie**: każdy
kolejny element zależy od poprzednich, w jednym przebiegu modelu na token.
Moduł 12 uczył generowania **przez próbkowanie z jawnie zaprojektowanej
przestrzeni latentnej**: jeden przebieg dekodera, jedna próbka $\mathbf z$ →
jeden wygenerowany przykład. Dyfuzja wprowadza **trzeci, jakościowo inny**
mechanizm: zaczynamy od czystego szumu i **iteracyjnie** go przekształcamy,
wielokrotnie wywołując tę samą, stosunkowo prostą sieć. To nie jest wariant
VAE czy autoregresji — to osobny sposób rozkładania trudnego zadania
("wygeneruj realistyczną próbkę") na wiele łatwiejszych podzadań
("usuń odrobinę szumu, znając poziom zaszumienia").

Nie trenujemy tu Stable Diffusion, nie pobieramy wag, nie używamy
biblioteki `diffusers` jako zależności — mechanizm DDPM jest w pełni widoczny
w kodzie, na danych 2D, żeby nic nie było "czarną skrzynką".

## Intuicja

❌ "Diffusion to algorytm, który wielokrotnie stosuje filtr wygładzający/
odszumiający do obrazu, jak w edycji zdjęć."

✅ Klasyczne odszumianie (np. filtr Gaussa, median filter) **usuwa znany typ
zakłócenia z KONKRETNEGO, ustalonego obrazu** — nie tworzy nowych danych.
Model dyfuzyjny robi coś fundamentalnie innego: uczy się, **jak w ogóle
wyglądają** dane treningowe (ich rozkład), a następnie generowanie polega na
**wielokrotnym** wywołaniu tej wiedzy, zaczynając od **czystego, losowego**
punktu startowego — nie od zaszumionej wersji istniejącego przykładu, który
mamy odtworzyć. To różnica między "napraw TO zdjęcie" i "wymyśl NOWE zdjęcie,
wiedząc, jak wyglądają zdjęcia tego typu".

**Kluczowy pomysł (dwie strony tego samego procesu):**

1. **Forward diffusion**: zepsuj dane treningowe, dodając szum w kontrolowany,
   znany sposób, krok po kroku, aż stracą całą strukturę.
2. **Reverse diffusion**: naucz sieć odwracać **jeden mały krok** tego
   procesu — a potem zastosuj tę nauczoną operację wielokrotnie, zaczynając
   od czystego szumu, żeby "wyhodować" nową próbkę.

## Krótkie porównanie rodzin modeli generatywnych

| Rodzina | Jak generuje | Czego się uczy |
|---|---|---|
| Autoregresyjne (moduł 11) | Element po elemencie, warunkowo na poprzednich | $P_\theta(x_t\mid x_{<t})$ |
| VAE (moduł 12) | Jedna próbka $\mathbf z\sim p(\mathbf z)$ → jeden przebieg dekodera | Probabilistyczny model latentny, ELBO |
| GAN (wspomniane krótko, nierozwijane) | Jeden przebieg generatora z szumu, trenowany przeciwstawnie z dyskryminatorem | Niejawny rozkład danych, gra generator-dyskryminator |
| Diffusion (ten wykład) | Iteracyjne odwracanie zaszumiania, wiele przebiegów tej samej sieci | Przewidywanie szumu $\varepsilon_\theta(x_t,t)$ |

Nie rozwijamy tu wyprowodzenia adversarial training (GAN) — to osobny temat,
wymagający własnej teorii gry i stabilizacji treningu, poza zakresem tego
modułu.

## Forward diffusion: matematyka

**Notacja.** $x_0$ — czysta próbka danych (podkreślenie: "czysta", przed
jakimkolwiek zaszumieniem). $x_t$ — próbka po $t$ krokach zaszumiania.

**Uwaga o notacji ($x_t$, trzecie znaczenie w tym kursie).** Moduł 9: $x_t$ =
ciągły wektor wejścia RNN w chwili $t$ (indeks = **pozycja w sekwencji**).
Moduł 11: $x_t$ = dyskretny identyfikator tokenu w pozycji $t$ (już
disambiguowane od modułu 9). Tu: $x_t$ = ciągły punkt/obraz po $t$ krokach
zaszumienia — indeks $t$ oznacza **poziom szumu dla JEDNEJ próbki**, nie
pozycję w sekwencji wielu elementów. Oś indeksowania jest inna, mimo tej
samej litery.

$\beta_t$ — wariancja szumu dodawanego w kroku $t$ ($0<\beta_t<1$).

**Uwaga o notacji ($\beta$, trzecie znaczenie).** Moduł 4: $\beta_1,\beta_2$
— współczynniki momentum Adama. Moduł 12: bare $\beta$ — waga KL w Beta-VAE.
Tu: $\beta_t$, **zawsze z indeksem $t$** — odróżnia to od obu poprzednich
(bare w module 12, podwójny indeks $1,2$ w module 4). Zbyt standardowe w
literaturze DDPM (Ho, Jain, Abbeel, 2020), żeby wymyślać nowy symbol.

$\alpha_t := 1-\beta_t$, $\bar\alpha_t := \prod_{s=1}^t \alpha_s$ (skumulowany
iloczyn).

**Uwaga o notacji ($\alpha$, trzecie znaczenie).** Moduł 10: wagi attention
$\alpha_{ij}$ (podwójny indeks). Moduł 12: współczynnik interpolacji VAE,
bare $\alpha\in[0,1]$. Tu: $\alpha_t$/$\bar\alpha_t$, **jednoindeksowane po
czasie** — wzorzec indeksowania (bare / $_{ij}$ / $_t$) rozróżnia wszystkie
trzy użycia.

**Proces forward** (łańcuch Markowa — każdy krok zależy **tylko** od
poprzedniego, nie od całej historii):

$$q(x_t \mid x_{t-1}) = N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)$$

To jest **proces fiksowany, bez uczonych parametrów** — $\beta_t$ ustalamy z
góry (harmonogram, niżej), nie uczymy. **Kluczowa różnica względem modułu
12**: VAE's $q_\phi(\mathbf z\mid x)$ jest **uczonym** przybliżonym
posteriorem (stąd indeks $\phi$); tu forward $q$ **nie ma** żadnego indeksu
parametrów — to zwykła, zdefiniowana formuła, nie coś, czego sieć się uczy.

### Wyprowodzenie zamkniętej postaci

Podstawowa własność: suma niezależnych zmiennych Gaussowskich jest
Gaussowska, z sumującymi się wariancjami (po odpowiednim przeskalowaniu).
Rozpisując rekurencję $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,z_t$
($z_t\sim N(0,I)$ niezależne dla każdego kroku) i podstawiając wielokrotnie:

$$x_t = \sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}\,x_0 + (\text{suma przeskalowanych niezależnych szumów})$$

Suma niezależnych Gaussów o wariancjach $(1-\alpha_t)\bar\alpha_t/\alpha_t,
\ldots$ sumuje się do wariancji $1-\bar\alpha_t$ (dowód indukcyjny — nie
powtarzamy tu pełnego rachunku, wynik jest standardowy w literaturze DDPM).
Końcowy wynik, **zamknięta postać**:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \varepsilon\sim N(0,I)$$

**Uwaga o notacji ($\varepsilon$).** Ten sam glif i ta sama rola koncepcyjna,
co w module 12 (szum reparametryzacji VAE: ustalony standardowy szum
Gaussowski, wykorzystywany do jawnej, różniczkowalnej konstrukcji próbki) —
to **kontynuacja**, nie kolizja. Diffusion rozszerza ten sam pomysł z
jednego kroku (VAE) na cały łańcuch.

**Dlaczego to jest tak użyteczne podczas treningu.** Bez zamkniętej postaci,
żeby uzyskać $x_t$ dla treningu, trzeba by symulować **cały** łańcuch $t$
kroków ($x_0\to x_1\to\cdots\to x_t$), co jest kosztowne i niepotrzebne.
Zamknięta postać pozwala **wylosować $t$ i $\varepsilon$ niezależnie, i
skonstruować $x_t$ w jednym kroku** — dokładnie to, co robimy w pętli
treningowej (Eksperyment B).

### Przykład liczbowy

$x_0 = [2{,}0,\ {-}1{,}0]$ (punkt 2D), $\varepsilon=[0{,}5,\ {-}0{,}3]$
(ustalony, dla weryfikacji), harmonogram liniowy $\beta\in[0{,}0001,\ 0{,}02]$,
$T=10$:

| $t$ | $\bar\alpha_t$ | $x_t$ (zamknięta postać) |
|---|---|---|
| 1 | $0{,}9999$ | $[2{,}0049,\ {-}1{,}0030]$ |
| 3 | $0{,}9931$ | $[2{,}0347,\ {-}1{,}0215]$ |
| 6 | $0{,}9667$ | $[2{,}0577,\ {-}1{,}0380]$ |
| 10 | $0{,}9037$ | $[2{,}0564,\ {-}1{,}0437]$ |

Zweryfikowane programowo w notebooku (Eksperyment A) — te same liczby, nie
tylko przeliczone na papierze. Widać: dla małych $t$, $x_t\approx x_0$
(niewielka zmiana); z rosnącym $t$, $\bar\alpha_t$ spada, punkt coraz
bardziej "dryfuje" i zwiększa wariancję — ale przy $T=10$ i tym konkretnym
harmonogramie, $\bar\alpha_{10}=0{,}90$ wciąż oznacza **silny** sygnał
oryginalnych danych (patrz sekcja o harmonogramie niżej — $T=10$ jest tu
tylko do ręcznej weryfikacji, nie do faktycznego treningu).

### Geometryczna utrata struktury

W notebooku (Eksperyment A) pokazujemy, jak zbiór punktów (mieszanina
Gaussów na okręgu) stopniowo "rozmazuje się" w jednorodny obłok z rosnącym
$t$ — struktura (osiem odseparowanych skupisk) zanika płynnie, nie
nagle.

❌ "Dla dowolnego harmonogramu i dowolnej (wystarczająco dużej) liczby
kroków $T$, rozkład $x_T$ jest dokładnie $N(0,I)$."

✅ **Nieprawda w ogólności — to zależy od konkretnego $\bar\alpha_T$.**
Zmierzone w notebooku: dla harmonogramu liniowego z $T=100$,
$\beta\in[0{,}0001,0{,}02]$: $\bar\alpha_{100}=0{,}364$ — **istotny rezydualny
sygnał**, rozkład $x_{100}$ **nie jest** bliski czystemu szumowi (wariancja
dodana to tylko $1-0{,}364=0{,}636$, nie $1$). Dla tego samego harmonogramu z
$T=1000$: $\bar\alpha_{1000}\approx 0{,}00004$ — praktycznie czysty szum.
**Wybór $T$ i harmonogramu $\beta$ wspólnie decydują**, jak bliski $N(0,I)$
jest rozkład końcowy — nie jest to automatyczna konsekwencja "wystarczająco
dużej liczby kroków" bez odniesienia do konkretnych wartości $\beta$.

## Noise schedule

**Harmonogram liniowy**: $\beta_t$ rośnie **liniowo** z $t$, od $\beta_1$
(małe) do $\beta_T$ (większe, ale wciąż $<1$).

❌ "Nazwa 'linear schedule' jednoznacznie mówi, co w tym harmonogramie jest
liniowe."

✅ Nazwa musi **wskazywać wielkość**, która zmienia się liniowo — tu: $\beta_t$
samo, **nie** $\bar\alpha_t$ (który zmienia się nieliniowo, nawet przy
liniowym $\beta_t$, bo jest iloczynem skumulowanym) i **nie** SNR (sygnał-do-
szumu, niżej). Różne prace definiują "liniowość" dla różnych wielkości —
zawsze trzeba sprawdzić, o którą wielkość chodzi.

**Harmonogram cosinusowy (cosine schedule, intuicja)**: definiuje się
$\bar\alpha_t$ **bezpośrednio** przez funkcję cosinus (nie $\beta_t$ wprost),
tak by sygnał ($\bar\alpha_t$) zanikał **wolniej** w środkowej części procesu
i **szybciej** pod koniec, w porównaniu do harmonogramu liniowego. Nie
wyprowadzamy tu dowodu optymalności żadnego z nich — to projektowa decyzja
empiryczna.

Zmierzone w notebooku: dla $T=100$, $\bar\alpha_t$ przy $t=25,50,75,100$:

| harmonogram | $t=25$ | $t=50$ | $t=75$ | $t=100$ |
|---|---|---|---|---|
| liniowy | $0{,}939$ | $0{,}777$ | $0{,}567$ | $0{,}364$ |
| cosinusowy | $0{,}847$ | $0{,}494$ | $0{,}144$ | $\approx 0$ |

Cosinusowy trzyma sygnał **wyżej** w pierwszej połowie, a potem **gwałtowniej**
go niszczy pod koniec — inny "tempo" destrukcji sygnału, nie tylko inna
formuła.

**Signal-to-noise ratio (SNR)**: $\text{SNR}_t = \bar\alpha_t / (1-\bar\alpha_t)$
— stosunek "ile zostało z sygnału" do "ile jest szumu" w kroku $t$. Duże
SNR → mało zaszumione; SNR→0 → praktycznie czysty szum. Harmonogram
definiuje całą **trajektorię** SNR od $t=0$ do $t=T$, nie tylko wartość
końcową.

## Reverse diffusion i DDPM: matematyka

**Dlaczego odwracanie wymaga informacji o rozkładzie danych.** Prawdziwy
$q(x_{t-1}\mid x_t)$ (bez warunkowania na $x_0$) zależy od **całego**
rozkładu danych $q(x_0)$ — w ogólności nieobliczalny w zamkniętej postaci
(podobny problem do nieobliczalnego $p(\mathbf z\mid x)$ w module 12).
Rozwiązanie: **aproksymujemy** go uczonym rozkładem $p_\theta(x_{t-1}\mid
x_t)$ — tu, **ponownie reużywając notację $p_\theta$ z modułu 12** (dekoder
VAE), ale teraz dla **jednego kroku odwrotnego procesu**, nie dla
"dekodowania z latentu".

### Model przewidujący szum

Zamiast uczyć sieć, która wprost przewiduje $x_{t-1}$ (albo $x_0$), DDPM
uczy sieć **przewidującą szum** $\varepsilon$, który został dodany:

$$\varepsilon_\theta(x_t, t) \approx \varepsilon \quad \text{(szum faktycznie użyty do skonstruowania } x_t \text{)}$$

❌ "Przewidywanie szumu jest celem samym w sobie — model dyfuzyjny 'chce'
nauczyć się rozpoznawać szum."

✅ Przewidywanie szumu jest **sposobem parametryzacji** modelu generatywnego
— matematycznie równoważnym (przez zamkniętą postać wyżej) przewidywaniu
$x_0$ albo bezpośrednio średniej rozkładu odwrotnego. Wybrano tę
parametryzację, bo **empirycznie** daje stabilniejszy trening (Ho, Jain,
Abbeel, 2020) — cel końcowy to wciąż **generowanie danych**, nie
"rozpoznawanie szumu" jako osobna umiejętność.

### Uproszczona funkcja celu

$$L_{\text{simple}} = \mathbb E_{x_0,t,\varepsilon}\big[\lVert \varepsilon - \varepsilon_\theta(x_t,t)\rVert^2\big], \qquad x_t = \sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$$

**Skąd bierze się sygnał treningowy (krok po kroku):**

1. Losujemy $x_0$ z danych treningowych (zwykły batch).
2. Losujemy $t\sim \text{Uniform}\{1,\ldots,T\}$ — **inny $t$ dla każdego
   przykładu w batchu**, nie jeden $t$ na cały batch.
3. Losujemy $\varepsilon\sim N(0,I)$ — ten konkretny szum, który "ukryjemy"
   w $x_t$.
4. Konstruujemy $x_t$ zamkniętą postacią (bez symulowania łańcucha).
5. Model przewiduje $\varepsilon_\theta(x_t,t)$ — **musi znać $t$** (niżej),
   bo zadanie "jaki szum trzeba odjąć" zależy od tego, **jak dużo** szumu
   już jest.
6. Porównujemy przewidywany i rzeczywisty szum przez MSE — sygnał
   treningowy to **różnica** między tym, co model przewidział, a tym, co
   faktycznie dodaliśmy (a więc wiemy z konstrukcji, nie z obserwacji).

❌ "Dobry (niski) noise prediction loss gwarantuje dobre próbki generowania."

✅ **Nieprawda — to osobne pytania.** Niski $L_{\text{simple}}$ mówi tylko,
że model dobrze przewiduje szum **dla pojedynczych, niezależnych $(x_t,t)$
par z rozkładu treningowego** — nie mówi nic bezpośrednio o jakości
**wielokrokowego, iteracyjnego** procesu generowania, gdzie małe błędy mogą
się kumulować przez wiele kroków. Zmierzone w notebooku (Eksperyment D):
strata spada **umiarkowanie** między epoką 20 i 200 ($0{,}392\to0{,}336$,
ok. 14% względnej redukcji), ale fraction-near-mode rośnie **dramatycznie**
w tym samym przedziale ($0{,}462\to0{,}886$) — mała różnica w loss, ogromna
różnica w jakości.

### Związek z wariacyjnym sformułowaniem DDPM

$L_{\text{simple}}$ **nie** jest arbitralnie wymyślone — wywodzi się z tej
samej logiki ELBO co w module 12 (dolne ograniczenie na log-prawdopodobieństwo,
teraz dla **łańcucha** zmiennych latentnych $x_1,\ldots,x_T$ zamiast
jednego $\mathbf z$). Pełny wariacyjny cel DDPM to ważona suma dywergencji
KL między $q$ i $p_\theta$ na każdym kroku łańcucha.

❌ "Uproszczony MSE $L_{\text{simple}}$ jest automatycznie identyczny z pełnym
ujemnym ELBO."

✅ **Nieprawda.** Ho, Jain i Abbeel (2020) pokazują, że $L_{\text{simple}}$
to pełny $-$ELBO **z usuniętymi wagami zależnymi od $t$** na poszczególnych
członach KL (każdy krok ważony jednakowo, niezależnie od tego, ile
"naprawdę" waży w pełnym wyprowodzeniu wariacyjnym). Autorzy zgłaszają, że
ta uproszczona, przeważona wersja **empirycznie** daje lepsze próbki niż
pełny, "poprawny" ELBO — ale to jest świadome odejście od dokładnego
wyprowodzenia wariacyjnego, analogiczne do $\beta$-VAE's odejścia od
standardowego ELBO (moduł 12), nie "to samo, tylko inaczej napisane".

### Aktualizacja odwrotnego procesu (jawna konwencja)

Przyjmujemy **jedną, explicit konwencję** (DDPM, wariancja ustalona jako
$\beta_t$ — nie jedyna możliwa, ale najprostsza, i ta, którą implementujemy):

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\right)$$

$$x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z, \qquad z\sim N(0,I),\ \sigma_t=\sqrt{\beta_t}, \quad \text{dla } t>1$$

$$x_0 = \mu_\theta(x_1,1) \quad \text{(ostatni krok: BEZ dodawania szumu)}$$

**Dlaczego ostatni krok jest inny.** Prawdziwa wariancja posteriora
$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ przy $t=1$
daje $\tilde\beta_1=0$ (bo $\bar\alpha_0:=1$ z definicji — nie ma "kroku
przed $x_0$") — formalnie, w ostatnim kroku nie powinno się dodawać żadnego
losowego składnika. Zweryfikowane programowo: $\tilde\beta_0=0{,}00000000$
dla naszego harmonogramu.

❌ "Formuła $\mu_\theta$ dla sąsiednich kroków $t$ i $t{-}1$ jest automatycznie
poprawna, jeśli przeskoczymy kilka kroków na raz (np. przy mniejszej liczbie
kroków próbkowania)."

✅ **Nieprawda — potwierdzone eksperymentalnie w sposób dramatyczny.**
Zmierzone w notebooku (Eksperyment D, model wytrenowany z $T=1000$): poprawne
**respacing** (przerzedzenie sekwencji kroków z przeliczonym efektywnym
$\alpha_{\text{eff}}=\bar\alpha_t/\bar\alpha_{\text{prev}}$ dla **tej
konkretnej** przerzedzonej pary kroków) degraduje się **płynnie** z mniejszą
liczbą kroków próbkowania (pokrycie modów: 8/8 zawsze, fraction-near-mode
$0{,}836\to0{,}798\to0{,}730\to0{,}596$ dla $100\to50\to20\to10$ kroków).
Naiwne przeskakiwanie (reużycie formuły **jednostkowego** kroku
$\beta_t/\alpha_t$ dla **dużego** skoku) degraduje się **katastroficznie**
(fraction-near-mode $0{,}032\to0{,}006\to0{,}008\to0{,}000$, pokrycie spada z
7/8 do 0/8 modów) — ten sam model, tylko poprawność procedury próbkowania
decyduje o różnicy między "trochę gorsze" i "praktycznie niedziałające".

**Zgodność indeksowania z kodem.** W kodzie (0-indeksowany `t_idx` od `0` do
`T-1`, odpowiadający matematycznemu $t=t_{\text{idx}}+1$) `t_idx=0` to
ostatni krok odwrotny (brak dodanego szumu), `t_idx=T-1` to pierwszy krok
(start od czystego szumu $x_T$). Zweryfikowane programowo, że `alphas_bar`
przy `t_idx` i formuła dają zgodne wyniki z ręcznym wyprowodzeniem.

### Warunkowanie czasem

❌ "Mała sieć MLP może przewidywać szum równie dobrze bez znajomości
aktualnego poziomu zaszumienia $t$ — w końcu widzi zaszumiony punkt $x_t$,
to powinno wystarczyć."

✅ **Nieprawda — zmierzony, nie tylko deklarowany efekt.** $x_t$ **sam**
nie mówi modelowi, **ile** szumu w nim jest względem oryginalnego sygnału —
ten sam punkt $x_t$ mógłby powstać z bliskiego $x_0$ przy dużym $t$ (duża
wariancja szumu) albo z odległego $x_0$ przy małym $t$ (mała wariancja) —
zależność $\varepsilon$ od $x_t$ **zmienia się** z $t$, więc model **musi**
znać $t$, żeby wiedzieć, **jakiej** zależności się aktualnie uczyć/używać.
Zmierzone ($T=1000$): model **bez** warunkowania czasem (identyczna
architektura, ignorująca $t$) osiągnął wyraźnie wyższą (gorszą) stratę
końcową ($0{,}638$ vs $0{,}337$) i **drastycznie** gorsze próbki — fraction-
near-mode spadł z $0{,}868$ do $0{,}026$, a średnia odległość do
najbliższego środka komponentu wybuchła z $0{,}51$ do $24{,}4$ (próbki
praktycznie rozbiegają się podczas 1000 iteracyjnych kroków bez informacji o
aktualnym poziomie szumu) — różnica mierzalna i dramatyczna, nie tylko
intuicyjna.

**Intuicja embeddingu czasu.** Prosty, znormalizowany skalar $t/T$ też by
"działał" w zasadzie, ale sinusoidalny embedding (ten sam pomysł co
pozycyjne embeddingi w module 10 — rozszerzony tu na **poziom szumu**, nie
**pozycję w sekwencji**) daje sieci gładką, wieloczęstotliwościową
reprezentację $t$, łatwiejszą do wykorzystania przez kolejne warstwy
liniowe niż jeden surowy skalar. Nie implementujemy pełnego U-Net — mały
MLP z konkatenacją $[x, \text{sinusoidal\_embedding}(t)]$ jest w pełni
wystarczający dydaktycznie dla danych 2D.

## Co obserwujemy

Wszystkie liczby niżej pochodzą z rzeczywistego wykonania `examples.ipynb`
(CPU, ten komputer) — mieszanina 8 Gaussów na okręgu (radius=4, std=0.3 per
komponent), $2000$ próbek treningowych.

- **Eksperyment A.** Forward diffusion zweryfikowane: zamknięta postać
  zgodna z symulacją krok-po-kroku statystycznie (średnia/wariancja na
  $50\,000$ próbach Monte Carlo dla $t=301$: stepwise mean$=[1{,}886,
  0{,}625]$/closed mean$=[1{,}886,0{,}634]$ vs oczekiwane $[1{,}883,0{,}628]$
  — zgodne w granicach błędu Monte Carlo, nie identyczność pojedynczych
  próbek). Geometrycznie: 8 odseparowanych skupisk rozmazuje się płynnie w
  jednorodny obłok z rosnącym $t$. $\bar\alpha_T$: $T=1000$ liniowy
  →$0{,}0000404$ (czysty szum), $T=100$ liniowy→$0{,}364$ (istotny
  rezydualny sygnał), $T=100$ cosinusowy→$0{,}0000$ (szybsza destrukcja
  sygnału na końcu).
- **Eksperyment B.** Trening ($T=1000$, 200 epok, MLP $2{+}16\to128\to128\to2$,
  2000 próbek): czas $0{,}85$s na CPU. Strata spada z $0{,}86$ (epoka 1) do
  $0{,}31$ (epoka 200), nie monotonicznie gładko — typowe dla tego zadania.
- **Eksperyment C.** Reverse diffusion od $x_T\sim N(0,I)$: po $T=1000$
  krokach, wygenerowane próbki pokrywają wszystkie 8 modów rozkładu
  treningowego (fraction-near-mode $0{,}868$, pokrycie 8/8, średnia
  odległość do najbliższego środka $0{,}515$ vs component std $0{,}3$),
  czas generowania $500$ próbek $0{,}18$s. Trajektorie pojedynczych punktów
  widoczne jako płynne, nie skokowe przejścia od szumu do struktury.
- **Eksperyment D.** Respaced correct vs naive wrong step-skipping (model
  $T=1000$, $n\_steps\in\{100,50,20,10\}$): correct fraction-near-mode
  $0{,}836\to0{,}798\to0{,}730\to0{,}596$ (pokrycie 8/8 zawsze); naive wrong
  $0{,}032\to0{,}006\to0{,}008\to0{,}000$ (pokrycie spada z 7/8 do 0/8).
  No-time-conditioning: strata końcowa $0{,}638$ (bez) vs $0{,}337$ (z) —
  fraction-near-mode $0{,}026$ vs $0{,}868$, średnia odległość do środka
  $24{,}4$ vs $0{,}515$ (próbki bez warunkowania czasem praktycznie
  rozbiegają się podczas 1000 iteracyjnych kroków). Undertrained snapshoty:
  strata $0{,}498\to0{,}392\to0{,}336$ (epoka 5/20/200), fraction-near-mode
  $0{,}142\to0{,}462\to0{,}886$ — loss i jakość **nie** poprawiają się w tym
  samym tempie.
- **Eksperyment E.** MMD² na znanych przypadkach kontrolnych: ta sama
  dystrybucja $N(0,I)$ vs $N(0,I)$ →$0{,}00044$; przesunięta dystrybucja
  →$1{,}144$ (poprawnie rozróżnia, zweryfikowane przed użyciem na prawdziwych
  danych). Na danych z Eksperymentu C: MMD²(test, wygenerowane)$=0{,}0052$
  vs MMD²(test, świeża niezależna próbka rzeczywista)$={-}0{,}0004$ (blisko
  zera, jak dla tej samej dystrybucji) — wygenerowane próbki mierzalnie
  bliższe rozkładowi rzeczywistemu niż losowy szum, ale nie identyczne z
  nową, niezależną próbką rzeczywistą.

## Częste błędne przekonania

(Zbiorcze podsumowanie — każde ma pełny ❌/✅ blok wyżej lub w notebooku)

1. Forward diffusion nie jest procesem, którego model musi się uczyć — to
   fiksowana, zdefiniowana z góry formuła.
2. Przewidywanie szumu jest sposobem parametryzacji modelu generatywnego,
   nie celem samym w sobie.
3. Dobry noise prediction loss nie gwarantuje dobrych próbek generowania.
4. Reverse diffusion nie jest zwykłym "odjęciem szumu" od danych — to
   iteracyjny, probabilistyczny proces z precyzyjnie zdefiniowaną formułą.
5. Generowanie nie wymaga podania rzeczywistej próbki do rekonstrukcji —
   startuje od czystego, niezależnego szumu.
6. Większa liczba kroków nie gwarantuje automatycznie lepszej jakości (choć
   zbyt mała liczba kroków, zwłaszcza bez poprawnego respacing, szkodzi).
7. Pominięcie/przeskoczenie timestepów wymaga matematycznie poprawnej,
   odpowiednio przeliczonej procedury — nie można reużyć formuły
   sąsiednich kroków dla dużych skoków.
8. Model nie musi bezbłędnie odtwarzać wszystkich modów rozkładu danych.
9. Wygenerowane próbki podobne wizualnie do treningowych nie dowodzą
   automatycznie memorization (zapamiętania konkretnych przykładów).
10. Mniejszy training loss nie musi oznaczać lepszego modelu generatywnego.
11. DDPM i DDIM to różne procedury próbkowania (niżej), nie dwie nazwy
    tego samego algorytmu.
12. Latent diffusion nie oznacza, że generowanie jest "darmowe" obliczeniowo
    — wciąż wymaga wielu przebiegów sieci, tylko w mniejszej przestrzeni.
13. Modele dyfuzyjne nie są automatycznie lepsze od VAE czy modeli
    autoregresyjnych — to zależy od zadania, zmierzone, nie z góry dane.

## DDPM vs DDIM (krótki przegląd)

DDPM (ten wykład): próbkowanie **stochastyczne** — na każdym kroku dodajemy
losowy składnik $\sigma_t z$ (poza ostatnim krokiem). Dwa przebiegi
generowania z tego samego $x_T$ dają **różne** końcowe próbki.

DDIM (*Denoising Diffusion Implicit Models*, Song, Meng, Ermon, 2021):
alternatywna procedura próbkowania, **deterministyczna** (albo z regulowaną
ilością stochastyczności), wykorzystująca ten **sam** wytrenowany model
$\varepsilon_\theta$, ale inną formułę aktualizacji — pozwala na znacznie
**mniejszą liczbę kroków próbkowania** przy porównywalnej jakości, bez
konieczności ponownego treningu.

❌ "DDPM i DDIM to dwie nazwy tego samego algorytmu próbkowania."

✅ **Nieprawda** — różnią się formułą aktualizacji i stochastycznością,
mimo że mogą korzystać z **tego samego** wytrenowanego modelu
$\varepsilon_\theta$. Nie implementujemy DDIM w tym module (poza zakresem),
ale respaced sampling (Eksperyment D) jest koncepcyjnie pokrewny: obie
techniki pokazują, że liczba kroków **próbkowania** nie musi być identyczna
z liczbą kroków **treningu** $T$, jeśli procedura jest odpowiednio
zdefiniowana.

## Dyfuzja w przestrzeni pikseli a latent diffusion

To, co zaimplementowaliśmy, działa **bezpośrednio** w przestrzeni danych
(2D punkty; dla obrazów — bezpośrednio na pikselach). **Latent diffusion**
(np. Stable Diffusion) najpierw koduje dane do **mniejszej** przestrzeni
reprezentacji (przez enkoder, konceptualnie jak enkoder AE/VAE z modułu 12),
wykonuje **cały** proces dyfuzji w tej mniejszej przestrzeni, a na końcu
**dekoduje** wynik z powrotem do pełnej przestrzeni danych (np. pikseli).

❌ "Latent diffusion oznacza, że generowanie jest obliczeniowo darmowe
(albo trywialnie szybkie)."

✅ **Nieprawda.** Latent diffusion **zmniejsza** koszt każdego pojedynczego
kroku dyfuzji (mniejsza przestrzeń = mniejsza sieć na krok) — ale wciąż
wymaga **wielu** przebiegów sieci odszumiającej (dokładnie jak nasz
eksperyment, tylko w innej przestrzeni), **plus** koszt kodowania/
dekodowania między przestrzenią danych i przestrzenią latentną. To
optymalizacja kosztu **per krok**, nie eliminacja iteracyjnej natury
dyfuzji.

## Związek ze współczesnymi generatorami obrazów (koncepcyjnie)

- **U-Net** (albo inna sieć odszumiająca) zastępuje nasz mały MLP —
  architektura dopasowana do obrazów (konwolucje, moduł 6), ale **rola**
  jest identyczna: przewidzieć szum $\varepsilon_\theta(x_t,t)$.
- **Warunkowanie na tekście**: dodatkowe wejście do sieci odszumiającej
  (embedding tekstu, koncepcyjnie jak embeddingi z modułu 8 i mechanizm
  attention z modułu 10, łączące reprezentację tekstu z reprezentacją
  obrazu w każdym kroku).
- **Classifier-free guidance**: technika sterowania próbkowaniem — model
  trenowany **z** i **bez** warunku (np. tekstu), a podczas generowania
  łączy się obie predykcje, żeby wzmocnić wpływ warunku. Nie wymaga
  osobnego klasyfikatora (wbrew starszej technice *classifier guidance*).
- **Latent diffusion**: opisane wyżej — łączy enkoder/dekoder (moduł 12)
  z iteracyjnym procesem dyfuzji.
- **Kompromis liczba kroków / koszt**: mniej kroków = szybsze generowanie,
  ale (bez odpowiedniej procedury, jak respacing/DDIM) gorsza jakość —
  dokładnie ten kompromis zmierzyliśmy w Eksperymencie D, tylko w 2D.

Nie implementujemy text-to-image, nie pobieramy wag, nie dodajemy
`diffusers` jako zależności obowiązkowej tego modułu — biblioteka może
służyć (lokalnie, opcjonalnie) jako punkt odniesienia przy dalszej,
samodzielnej eksploracji, nie jako część wymaganego materiału.

## Implikacje inżynierskie

- **Koszt generowania ≠ koszt jednego przebiegu sieci.** Nawet mała sieć
  generuje relatywnie wolno, jeśli trzeba ją wywołać $T$ razy **sekwencyjnie**
  dla **jednej** próbki (każdy krok zależy od wyniku poprzedniego) — koszt
  całkowity to (koszt jednego przebiegu) $\times$ (liczba kroków), nie
  koszt jednego przebiegu.
- **Liczba kroków jest bezpośrednim przełącznikiem jakość/czas** —
  zmniejszenie liczby kroków próbkowania (poprawnie, przez respacing/DDIM)
  przyspiesza generowanie, z pewnym kosztem jakości; brak odpowiedniej
  procedury przy zmniejszaniu kroków psuje wynik znacznie bardziej, niż
  "rozsądnie mniej dokładny" wynik.
- **Batching przy generowaniu** amortyzuje koszt: generowanie $N$ próbek
  naraz (batch) kosztuje nieproporcjonalnie mniej niż $N$ razy koszt
  generowania jednej próbki, bo każdy krok sieci przetwarza cały batch
  równolegle.
- **Koszt treningu vs koszt generowania to różne budżety** — trening (w tym
  module: raz, ~1s) jest **jednorazowy**; generowanie (wiele wywołań
  sieci) jest kosztem **powtarzalnym**, płaconym za każdą nową próbkę.
- **Skalowanie z 2D do obrazów** nie zmienia mechanizmu, zmienia **koszt
  per krok** (większa sieć, więcej pamięci) — sam fakt potrzeby wielu
  kroków zostaje.
- **Walidacja jakości generowanych danych wymaga metryk, nie tylko
  wykresów** — i żadna pojedyncza metryka nie jest kompletna (Eksperyment
  E pokazuje to wprost: metryka, która nie rozróżnia kształtu rozkładu,
  może "faworyzować" gorszy model).
- **Diversity i coverage są trudne do oddzielenia** od jakości pojedynczych
  próbek — model może generować bardzo realistyczne, ale mało zróżnicowane
  próbki (pokrywające tylko część modów) — osobne pytanie od "czy pojedyncza
  próbka wygląda dobrze".

## Co warto zapamiętać

- Forward diffusion: $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$
  — fiksowany proces, zamknięta postać pozwala próbkować $x_t$ bez
  symulowania całego łańcucha.
- Model uczy się przewidywać szum $\varepsilon_\theta(x_t,t)$ — **musi**
  znać $t$, bo zależność szumu od $x_t$ zmienia się z poziomem zaszumienia.
- $L_{\text{simple}}$ to uproszczona, przeważona wersja $-$ELBO dla
  łańcucha — nie dowolnie wymyślona strata, ale też nie dokładnie pełny
  ELBO.
- Reverse diffusion: $\mu_\theta(x_t,t)$ z jawnej formuły, ostatni krok bez
  dodanego szumu ($\tilde\beta_0=0$ formalnie).
- Mniejsza liczba kroków próbkowania wymaga matematycznie poprawnej
  procedury (respacing) — naiwne przeskakiwanie katastrofalnie degraduje
  jakość, zmierzone, nie tylko zadeklarowane.
- Dobra strata treningowa nie gwarantuje dobrych próbek — jakość generowania
  trzeba zmierzyć osobno, wieloma metrykami.
- Diffusion, VAE i modele autoregresyjne to trzy różne, ważne podejścia —
  żadne nie jest uniwersalnie najlepsze.

## Pytania dla studentów

1. Masz $x_0=[1{,}0,2{,}0]$, harmonogram liniowy $T=50$,
   $\beta\in[0{,}0001,0{,}02]$. Jak policzyłbyś/łabyś $\bar\alpha_{25}$ bez
   symulowania 25 kroków łańcucha? Co by się zmieniło, gdybyś użył innego,
   ustalonego $\varepsilon$?
2. Dlaczego model przewidujący szum **musi** otrzymać $t$ jako wejście, a
   model klasyfikujący obrazy (moduł 6) nie potrzebuje analogicznej
   informacji?
3. Masz wytrenowany model DDPM z $T=1000$. Chcesz generować próbki w $50$
   kroków, nie $1000$. Jaka jest **poprawna** procedura, i czym różni się od
   "po prostu zrób co 20. krok"?
4. Wyjaśnij, dlaczego niski noise prediction loss na zbiorze treningowym
   nie jest sam w sobie dowodem, że model generuje dobre próbki.
5. Masz dwie metryki oceny generatora: jedna mówi "model A lepszy", druga
   "model B lepszy". Jak zdecydowałbyś/łabyś, której bardziej ufać w
   konkretnym zadaniu?

## Bibliografia

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic
  Models*. NeurIPS 33. arXiv:2006.11239 (główna praca źródłowa DDPM, cel
  $L_{\text{simple}}$, formuła reverse step).
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit
  Models*. ICLR. arXiv:2010.02502 (DDIM, próbkowanie deterministyczne z
  mniejszą liczbą kroków).
- Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion
  Probabilistic Models*. ICML. arXiv:2102.09672 (cosine schedule, respaced
  sampling).
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015).
  *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML
  (pierwsza praca wprowadzająca ramy probabilistyczne dyfuzji).
- Rombach, R., i in. (2022). *High-Resolution Image Synthesis with Latent
  Diffusion Models*. CVPR. arXiv:2112.10752 (latent diffusion, kontekst dla
  sekcji "Związek ze współczesnymi generatorami").
