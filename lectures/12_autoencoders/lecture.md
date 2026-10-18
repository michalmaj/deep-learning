# Wykład 12 — Autoenkodery (AE), wariacyjne autoenkodery (VAE) i przestrzeń latentna

## Cele wykładu

1. Rozumieć uczenie przez rekonstrukcję jako formę uczenia **bez etykiet**
   (*self-supervised*) i wiedzieć, kiedy taka rekonstrukcja wymusza, a kiedy
   **nie** wymusza użytecznej reprezentacji.
2. Umieć formalnie zdefiniować autoenkoder (AE): enkoder, bottleneck,
   dekoder, stratę rekonstrukcji — i policzyć mały przykład liczbowy.
3. Rozumieć różnicę między AE (deterministyczny wektor **e**) a VAE
   (probabilistyczny rozkład nad **z**) — i **dlaczego** ta różnica ma
   znaczenie dla generowania nowych danych.
4. Umieć wyprowadzić ELBO na poziomie odpowiednim dla podstawowej
   probabilistyki/analizy, rozróżniając maksymalizację ELBO od minimalizacji
   straty treningowej (= $-$ELBO).
5. Znać wzór KL dla diagonalnego rozkładu Gaussa, reparameterization trick, i
   rozumieć, dlaczego jest konieczny dla propagacji gradientu przez losowanie.
6. Krytycznie oceniać zastosowanie błędu rekonstrukcji do detekcji anomalii —
   wiedzieć, czym jest, a czym **nie jest** taka procedura.

## Dlaczego to jest ważne

Moduł 8 uczył reprezentacji (embeddingów) poprzez zadanie **z nadzorem lub
pół-nadzorem** powiązane z podobieństwem. Moduł 11 budował model generatywny
tekstu poprzez **autoregresję** — przewidywanie następnego tokenu. Ten wykład
pyta o **trzeci** sposób uczenia reprezentacji i generowania danych:
**rekonstrukcja**. Sieć nie ma dostępu do żadnej etykiety — ma tylko dane
wejściowe $x$ i zadanie "odtwórz $x$ z własnej, wewnętrznej, zwykle mniejszej
reprezentacji". To pozwala stawiać pytania, które nie pojawiły się wcześniej:
czy taka reprezentacja jest *automatycznie* użyteczna? Czy model, który
dobrze rekonstruuje, jest automatycznie dobrym modelem generatywnym? VAE
odpowiada "nie" na obie wersje tego pytania w sposób sformalizowany — i to
jest główny powód, by uczyć się VAE **po** AE, nie zamiast AE.

Nie rozwijamy tu diffusion models (kolejny, osobny sposób generowania danych
— moduł 13) i nie wyprowadzamy ponownie architektury Transformer (moduł 10)
czy mechanizmu autoregresji (moduł 11) — enkodery/dekodery w tym module są
prostymi sieciami MLP, nie Transformerami.

## Intuicja: uczenie przez rekonstrukcję

❌ "Sieć, która nauczy się dobrze rekonstruować dane wejściowe, musi nauczyć
się czegoś 'głębokiego' o danych."

✅ Sieć uczy się **tylko tego, co jest potrzebne, żeby zminimalizować stratę
rekonstrukcji** na danym zbiorze treningowym, przy danych ograniczeniach
architektury. Jeśli ograniczenia są słabe (np. bottleneck wystarczająco duży),
sieć może nauczyć się **trywialnego** mapowania (prawie tożsamości) — bez
żadnej kompresji, bez żadnej "zrozumiałej" struktury. Zobaczymy to
bezpośrednio w Eksperymencie A (sekcja "Overcomplete vs undercomplete" niżej)
— to nie jest wyjątek teoretyczny, tylko zmierzone zjawisko.

Dobra reprezentacja **może** wyłonić się z rekonstrukcji — ale wyłania się
tylko wtedy, gdy architektura **wymusza** nietrywialną kompresję (bottleneck
mniejszy niż wymiar danych) albo gdy dodajemy inne ograniczenia/regularyzację.
Nic z tego nie jest gwarantowane samym faktem "sieć uczy się odtwarzać dane".

## Autoenkoder (AE): formalny model

**Notacja.** $x \in \mathbb R^{d_x}$ — dane wejściowe ($d_x$ rozszerza
wzorzec $d$-z-indeksem z modułów 08-10, tu: wymiar wejścia). **e** $\in
\mathbb R^{d_e}$ (pogrubione, moduł 08: reprezentacja/embedding) — reprezentacja
w bottlenecku, wymiar $d_e$. $\hat x$ — rekonstrukcja (wzorzec $\hat y$ =
predykcja, moduł 08, tu zastosowany do całego wejścia, nie do etykiety).

$$\mathbf e = f_{\text{enc}}(x), \qquad \hat x = f_{\text{dec}}(\mathbf e)$$

Strata rekonstrukcji (najczęściej MSE dla danych ciągłych):

$$L_{\text{AE}} = \frac{1}{N}\sum_{n=1}^N \lVert x^{(n)} - \hat x^{(n)} \rVert_2^2$$

($L$ = strata, zgodnie z konwencją kursu od modułu 01; $N$ = rozmiar batcha/
zbioru, zgodnie z konwencją modułów 01/02/06/07).

### Mały przykład liczbowy

Weźmy $x = [1{,}0,\ 0{,}5,\ -0{,}5,\ 0{,}0] \in \mathbb R^4$ ($d_x=4$),
liniowy enkoder bez biasu do $d_e=2$ z wagami

$$W_{\text{enc}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 0 \end{pmatrix}$$

$$\mathbf e = W_{\text{enc}} x = \begin{pmatrix} 1\cdot 1{,}0 \\ 1\cdot 0{,}5 + 1\cdot(-0{,}5) \end{pmatrix} = \begin{pmatrix} 1{,}0 \\ 0{,}0 \end{pmatrix}$$

Dekoder, też liniowy, $W_{\text{dec}} = W_{\text{enc}}^\top$ (dekoder próbujący
"odwrócić" projekcję):

$$\hat x = W_{\text{dec}}\, \mathbf e = \begin{pmatrix}1\\0\\0\\0\end{pmatrix}\cdot 1{,}0 + \begin{pmatrix}0\\1\\1\\0\end{pmatrix}\cdot 0{,}0 = \begin{pmatrix}1{,}0\\0{,}0\\0{,}0\\0{,}0\end{pmatrix}$$

$$L_{\text{AE}} = (1{,}0-1{,}0)^2+(0{,}5-0{,}0)^2+(-0{,}5-0{,}0)^2+(0{,}0-0{,}0)^2 = 0{,}5$$

Zweryfikowane programowo w notebooku (Eksperyment A, wstępna komórka):
dokładnie te same liczby, nie tylko przeliczone na papierze. Ten konkretny
dekoder **nie** jest optymalny dla tych danych (druga współrzędna $\mathbf e$
"ucięła" informację o różnicy między $x_2$ i $x_3$ nieodwracalnie) — pokazuje
to wprost, że sama architektura enkoder/dekoder **nie gwarantuje** dobrej
rekonstrukcji; trzeba **wytrenować** wagi tak, by stratę minimalizować, co
robimy dalej w notebooku na prawdziwych obrazach.

## Undercomplete vs overcomplete: skąd bierze się kompresja

- **Undercomplete AE**: $d_e < d_x$. Bottleneck jest **węższy** niż wejście —
  sieć **musi** stracić część informacji (nie da się odwracalnie zmapować
  $\mathbb R^{d_x}$ do $\mathbb R^{d_e}$ i z powrotem, gdy $d_e<d_x$, dla
  ogólnych danych pełnego rzędu). To jest mechanizm, który **wymusza**
  kompresję.
- **Overcomplete AE**: $d_e \geq d_x$. Bottleneck **nie** ogranicza
  teoretycznej zdolności do odwracalnego mapowania — bez dodatkowej
  regularyzacji (np. sparsity, szum na wejściu/denoising, ograniczenie normy)
  sieć może nauczyć się (prawie) tożsamości.

❌ "Autoenkoder zawsze kompresuje dane do mniejszej, użytecznej reprezentacji."

✅ **Nieprawda w ogólności** — zmierzone wprost na syntetycznych danych
pełnego rzędu w $\mathbb R^8$: overcomplete AE ($d_e=32$) osiągnął MSE
rekonstrukcji $\approx 0$ (trywialne mapowanie identycznościowe, zero
kompresji), podczas gdy undercomplete AE ($d_e=2$) na **tych samych danych**
nie mógł zejść poniżej MSE $\approx 0{,}69$ — bottleneck **musiał** wymusić
utratę informacji. Kompresja nie jest własnością "bycia autoenkoderem" — jest
konsekwencją **konkretnego** ograniczenia architektury na **konkretnych**
danych. Zweryfikowane w notebooku (Eksperyment A, komórka weryfikacyjna).

## Autoenkoder liniowy i PCA

Specjalny, pouczający przypadek: enkoder i dekoder **liniowe, bez
aktywacji**, strata MSE. W tym przypadku (Baldi & Hornik, 1989) optymalne
rozwiązanie **rozpina tę samą podprzestrzeń** (*subspace*), co $d_e$
pierwszych głównych składowych PCA (moduł 08: `torch.linalg.eigh`) — przy
pełnej konwergencji treningu.

❌ "Liniowy autoenkoder automatycznie nauczy się dokładnie wektorów własnych
PCA (tej samej bazy, w tej samej kolejności, co wariancja)."

✅ **Nieprawda w tej silnej formie.** AE nauczy się bazy, która rozpina **tę
samą podprzestrzeń** — ale ta baza nie musi być ortonormalna, nie musi być
uporządkowana według wariancji, i może być **inną** bazą tej samej
podprzestrzeni (enkoder i PCA mogą "obrócić" tę przestrzeń inaczej).
Rekonstrukcja (błąd MSE) będzie taka sama jak dla optymalnej projekcji PCA na
tę podprzestrzeń — ale surowe wagi enkodera generalnie **nie** są równe
wektorom własnym PCA wprost.

Zweryfikowane programowo na syntetycznych danych o znanym, ukrytym rzędzie 3
w $\mathbb R^{10}$: po **dostatecznie długim** treningu (8000 epok, lr=0,02)
AE osiągnął MSE rekonstrukcji $0{,001764}$ — **zgodne** z optymalnym MSE PCA
rzędu 3 na tych danych — a kąty główne (*principal angles*) między
podprzestrzenią rozpiętą przez wagi enkodera i podprzestrzenią PCA wyniosły
**poniżej $0{,}2°$** dla wszystkich trzech wymiarów (identyczna podprzestrzeń
w granicach błędu numerycznego).

❌ "Ta równoważność AE/PCA zachodzi automatycznie, niezależnie od tego, jak
długo trenujemy."

✅ **Nieprawda — wymaga wystarczającej konwergencji.** Ten sam eksperyment,
zatrzymany **wcześniej** (2000 epok, mniejszy learning rate), dał podobny
błąd rekonstrukcji, ale kąty główne do **$19{,}8°$** — podprzestrzeń była
jeszcze wyraźnie "krzywa" względem PCA, mimo niemal takiego samego MSE.
Niski błąd rekonstrukcji **nie** dowodzi sam z siebie, że AE "znalazł" tę
samą podprzestrzeń co PCA — to osobne, zmierzone fakty, nie jeden i ten sam.

## Dlaczego AE nie jest automatycznie dobrym modelem generatywnym

AE ma dekoder $f_{\text{dec}}: \mathbb R^{d_e} \to \mathbb R^{d_x}$ — można
podać mu **dowolny** wektor $\mathbf e$ i dostać "obraz". Problem: AE **nie
uczy się**, jakie wektory $\mathbf e$ są "sensowne" (odpowiadają
rzeczywistym danym) — bottleneck tylko minimalizuje błąd rekonstrukcji **dla
wektorów, które faktycznie powstały z kodowania prawdziwych danych
treningowych**. Przestrzeń $\mathbb R^{d_e}$ poza regionami faktycznie
zajętymi przez zakodowane dane treningowe jest **niezdefiniowana** — dekoder
może dać tam cokolwiek, zwykle coś niesensownego.

❌ "Mogę wylosować przypadkowy wektor $\mathbf e$ i podać go do dekodera AE,
żeby wygenerować nowy, sensowny przykład."

✅ AE **nie ma** mechanizmu, który organizowałby przestrzeń $\mathbf e$ w
jakiś konkretny, znany rozkład (np. $N(0,I)$), z którego moglibyśmy losować.
Właśnie **to** jest motywacja probabilistycznego podejścia VAE, niżej — VAE
jawnie **projektuje** przestrzeń latentną tak, by miała znany, ustalony
rozkład "z góry" (prior), z którego losowanie ma sens.

## VAE: model probabilistyczny

Zamiast deterministycznego $\mathbf e = f_{\text{enc}}(x)$, VAE modeluje
**łączny rozkład** obserwacji $x$ i zmiennej latentnej $\mathbf z \in
\mathbb R^{d_z}$ (pogrubione **z** — **nowe** użycie w tym kursie; zwykłe,
niepogrubione $z$ to logity, moduł 02+, używane też jako $z_t$ w module 11 —
oba mogą się pojawić w tym samym akapicie, np. przy dekoderze z
`binary_cross_entropy_with_logits`, stąd rozróżnienie pogrubieniem jest tu
istotne, nie kosmetyczne):

- **Prior** $p(\mathbf z)$ — rozkład "z góry" nad przestrzenią latentną,
  zwykle $\mathbf z \sim N(0, I)$.
- **Dekoder (model obserwacji)** $p_\theta(x \mid \mathbf z)$ — jak bardzo
  prawdopodobne jest zaobserwowanie $x$, mając dany kod $\mathbf z$.
- **Prawdziwy posterior** $p(\mathbf z \mid x)$ — rozkład nad kodami,
  **mając dany konkretny** $x$. To jest to, czego chcielibyśmy znać przy
  kodowaniu — ale jest **nieobliczalny** (*intractable*) w ogólności:

$$p(\mathbf z \mid x) = \frac{p_\theta(x \mid \mathbf z)\,p(\mathbf z)}{p(x)}, \qquad p(x) = \int p_\theta(x \mid \mathbf z)\,p(\mathbf z)\, d\mathbf z$$

Mianownik $p(x)$ wymaga całkowania po **całej** przestrzeni $\mathbf z$ — dla
sieci neuronowej jako $p_\theta(x\mid \mathbf z)$ ta całka nie ma postaci
zamkniętej i jest obliczeniowo niewykonalna w wysokich wymiarach.

**Wnioskowanie wariacyjne (variational inference), intuicja.** Skoro
$p(\mathbf z\mid x)$ jest nieobliczalny, **przybliżamy** go wygodną rodziną
rozkładów $q_\phi(\mathbf z \mid x)$ (tu: diagonalny Gauss, niżej) — nazywaną
**przybliżonym posteriorem** (*approximate posterior*) — i uczymy parametry
$\phi$ (sieć enkodera) tak, by $q_\phi(\mathbf z\mid x)$ był **jak najbliżej**
prawdziwego $p(\mathbf z \mid x)$, w sensie dywergencji KL (niżej).
$q_\phi(\mathbf z\mid x)$ nigdy nie jest **identyczny** z $p(\mathbf z\mid
x)$ — to przybliżenie z ograniczonej rodziny rozkładów, nie dokładne
odtworzenie.

## ELBO: wyprowadzenie

Chcemy maksymalizować log-prawdopodobieństwo danych $\log p_\theta(x)$, ale
jest ono nieobliczalne (wymaga tej samej całki co wyżej). Wprowadzamy
dowolny rozkład $q_\phi(\mathbf z \mid x)$ i piszemy:

$$\log p_\theta(x) = \log \int p_\theta(x,\mathbf z)\, d\mathbf z = \log \int q_\phi(\mathbf z\mid x) \frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\, d\mathbf z = \log \mathbb E_{q_\phi(\mathbf z\mid x)}\!\left[\frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right]$$

Z nierówności Jensena ($\log$ jest wklęsły, $\log \mathbb E[\cdot] \geq
\mathbb E[\log(\cdot)]$):

$$\log p_\theta(x) \geq \mathbb E_{q_\phi(\mathbf z\mid x)}\!\left[\log \frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right] =: \mathcal L(\theta,\phi; x)$$

$\mathcal L$ (script L, mathcal — **nowy symbol w tym kursie**, celowo
odróżniony od zwykłego $L$ = strata treningowa od modułu 01) to **ELBO**
(*Evidence Lower BOund*) — dolne ograniczenie na log-prawdopodobieństwo
danych, które **da się** policzyć (albo oszacować próbkowaniem). Rozpisując
$p_\theta(x,\mathbf z) = p_\theta(x\mid \mathbf z)\,p(\mathbf z)$:

$$\mathcal L(\theta,\phi;x) = \underbrace{\mathbb E_{q_\phi(\mathbf z\mid x)}\big[\log p_\theta(x\mid \mathbf z)\big]}_{\text{człon rekonstrukcji}} - \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big)}_{\text{człon KL}}$$

**Dwa składniki, dwie różne role:**

1. **Oczekiwane log-prawdopodobieństwo rekonstrukcji** — jak dobrze dekoder
   odtwarza $x$ z próbki $\mathbf z\sim q_\phi(\mathbf z\mid x)$. Im wyższe,
   tym lepsza rekonstrukcja.
2. **Dywergencja KL** między przybliżonym posteriorem i priorem — jak
   daleko $q_\phi(\mathbf z\mid x)$ odszedł od $p(\mathbf z)$. Im **niższe**,
   tym bliżej do uporządkowanego, znanego rozkładu z góry.

❌ "Maksymalizujemy ELBO i minimalizujemy stratę treningową — to ta sama
operacja, ten sam znak."

✅ **Maksymalizujemy** $\mathcal L$ (ELBO) — ale w praktyce frameworki (i ten
kurs) **minimalizują** stratę. Dlatego definiujemy stratę treningową jako
**ujemne** ELBO:

$$L_{\text{VAE}}(\theta,\phi;x) := -\mathcal L(\theta,\phi;x) = -\mathbb E_{q_\phi(\mathbf z\mid x)}\big[\log p_\theta(x\mid \mathbf z)\big] + D_{\mathrm{KL}}\!\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big)$$

Minimalizacja $L_{\text{VAE}}$ $\equiv$ maksymalizacja $\mathcal L$ $\equiv$
(z dołu) maksymalizacja $\log p_\theta(x)$. Trzy różne sformułowania **tego
samego** celu — ale tylko środkowe/ostatnie to coś, co faktycznie
maksymalizujemy; to, co kod **minimalizuje**, to $L_{\text{VAE}}$, ze znakiem
odwrotnym.

❌ "«Strata rekonstrukcji + strata KL» to dowolnie wymyślona funkcja straty,
złożona z dwóch niezależnie wybranych kawałków."

✅ **Nieprawda** — to jest **konkretna, wyprowodzona** postać $-$ELBO dla
**konkretnych** założeń probabilistycznych (prior $N(0,I)$, określony
$p_\theta(x\mid\mathbf z)$, $q_\phi$ z rodziny Gaussów diagonalnych — niżej).
Zmiana np. rozkładu dekodera (Bernoulli vs Gauss, niżej) zmienia **konkretną
postać** członu rekonstrukcji — ale wciąż w ramach tego samego wyprowodzenia,
nie przez dowolne dopisanie nowego termu.

## Gaussowska przestrzeń latentna

Standardowy wybór: prior $p(\mathbf z) = N(0, I)$, przybliżony posterior —
**diagonalny** Gauss:

$$q_\phi(\mathbf z \mid x) = N(\boldsymbol\mu(x),\ \mathrm{diag}(\boldsymbol\sigma^2(x)))$$

Enkoder (sieć $\phi$) **nie** zwraca jednego wektora **e** jak w AE — zwraca
**dwa** wektory, parametry rozkładu: $\boldsymbol\mu \in \mathbb R^{d_z}$
(średnia — symbol wolny, bez kolizji w tym kursie) i $\log\boldsymbol\sigma^2
\in \mathbb R^{d_z}$ (logarytm wariancji, **`logvar`** w kodzie).

**Uwaga o notacji.** Nie używamy bare $\sigma$ dla odchylenia standardowego w
prozie tego wykładu — $\sigma(\cdot)$ (zawsze z argumentem w nawiasie) jest
od modułu 02 zarezerwowane dla funkcji aktywacji sigmoid. Tu $\sigma^2$ (albo
$\log\sigma^2$) pojawia się **wyłącznie** jako forma złożona (z indeksem
`²` albo `log`), nigdy bare — a w kodzie i większości tekstu używamy
bezpośrednio `logvar`, zgodnie z konwencją biblioteki referencyjnej (Kingma
& Welling, 2013) i PyTorch.

### Dlaczego enkoder zwraca `mu`/`logvar`, nie gotową próbkę

Gdybyśmy chcieli próbkować $\mathbf z \sim N(\boldsymbol\mu,
\mathrm{diag}(\boldsymbol\sigma^2))$ **wprost**, operacja losowania nie ma
gradientu względem $\boldsymbol\mu,\boldsymbol\sigma^2$ — nie można
propagować błędu "przez" losowe losowanie tak jak przez zwykłą warstwę.

### Reparameterization trick

Rozwiązanie: przenosimy całą losowość do **osobnej** zmiennej $\varepsilon$
niezależnej od parametrów sieci, i piszemy $\mathbf z$ jako
**deterministyczną** funkcję $\boldsymbol\mu$, `logvar` i $\varepsilon$:

$$\mathbf z = \boldsymbol\mu + \exp(0{,}5 \cdot \text{logvar}) \odot \boldsymbol\varepsilon, \qquad \boldsymbol\varepsilon \sim N(0, I)$$

($\odot$ = mnożenie po współrzędnych). **Uwaga o notacji.** $\varepsilon$
(regularny epsilon) oznacza tu szum reparametryzacji — inny glif niż
$\varepsilon$-stała numerycznego bezpieczeństwa z modułu 08 (`\varepsilon`,
wariantowy epsilon) — różne zastosowania, różne glify, nota dezambiguacyjna
przy pierwszym użyciu, zgodnie z ustaloną w kursie zasadą.

❌ "Reparameterization trick usuwa losowość z modelu VAE."

✅ **Nieprawda** — losowość **wciąż jest obecna** (próbkujemy
$\boldsymbol\varepsilon$ na każdym forward passie). Trick **zmienia tylko
to, gdzie w grafie obliczeniowym ta losowość wchodzi** — $\boldsymbol\varepsilon$
jest **wejściem** (bez gradientu, nie trzeba go liczyć), a $\boldsymbol\mu$ i
`logvar` są **zwykłymi, różniczkowalnymi** wyjściami sieci. Gradient płynie
przez $\boldsymbol\mu$ i `logvar` dokładnie tak jak przez każdą inną warstwę
— losowanie samo w sobie nie jest już "na ścieżce" propagacji gradientu.

**Przykład liczbowy (2 wymiary latentne), zweryfikowany względem PyTorch.**
$\boldsymbol\mu=[0{,}5,\ {-}0{,}3]$, logvar$=[{-}0{,}2,\ 0{,}4]$, ustalone
$\boldsymbol\varepsilon=[1{,}0,\ {-}0{,}5]$ (normalnie losowane, tu ustalone
dla weryfikacji):

$$\exp(0{,}5\cdot(-0{,}2)) = 0{,}9048, \qquad \exp(0{,}5\cdot 0{,}4) = 1{,}2214$$
$$\mathbf z = [0{,}5 + 0{,}9048\cdot 1{,}0,\ \ {-}0{,}3 + 1{,}2214\cdot({-}0{,}5)] = [1{,}4048,\ {-}0{,}9107]$$

Zweryfikowane programowo: `torch.allclose` z wartością obliczoną wzorem,
bit-w-granicach-tolerancji numerycznej (nie bit-exact w sensie `==`, bo
operacje na `float32`). Dla straty $\mathcal L = \sum_i z_i^2$, policzony
gradient: $\partial \mathcal L/\partial \boldsymbol\mu = 2\mathbf z$
(zweryfikowane programowo: `mu.grad` $=[2{,}8097,\ {-}1{,}8214]$, dokładnie
$2\mathbf z$), a $\partial\mathcal L/\partial\text{logvar} = \mathbf
z\cdot\boldsymbol\varepsilon\cdot\exp(0{,}5\cdot\text{logvar})$
(zweryfikowane programowo: `logvar.grad` $=[1{,}2711,\ 0{,}5562]$, zgodne z
tym wzorem). Obie wartości gradientu są skończone (`torch.isfinite`) — to
jest bezpośredni, policzony dowód, że gradient **faktycznie płynie** przez
reparametryzację do obu parametrów rozkładu.

### Dywergencja KL dla diagonalnego Gaussa vs $N(0,I)$

Dla $q = N(\mu_i, \sigma_i^2)$ (jeden wymiar $i$) i priora $N(0,1)$, wzór
analityczny (standardowy wynik dla dwóch Gaussów jednowymiarowych):

$$D_{\mathrm{KL}}(N(\mu_i,\sigma_i^2)\,\|\,N(0,1)) = \frac{1}{2}\left(\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2\right)$$

Sumujemy po wszystkich $d_z$ wymiarach (niezależność współrzędnych w
diagonalnym Gaussie):

$$D_{\mathrm{KL}}\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big) = \frac{1}{2}\sum_{i=1}^{d_z}\left(\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2\right) = -\frac{1}{2}\sum_{i=1}^{d_z}\left(1 + \text{logvar}_i - \mu_i^2 - e^{\text{logvar}_i}\right)$$

**Przykład liczbowy, zweryfikowany względem `torch.distributions`.** Dla
$\boldsymbol\mu=[0{,}5,{-}0{,}3]$, logvar$=[{-}0{,}2,0{,}4]$: KL per wymiar
$=[0{,}1344,\ 0{,}0909]$, suma $=0{,}2253$. Zweryfikowane programowo:
`torch.allclose` między tym wzorem i
`torch.distributions.kl_divergence(Normal(mu,sigma), Normal(0,1))` z
`atol=1e-6` — zgodność w granicach tolerancji numerycznej (nie bit-exact w
sensie `==`). Nieujemność KL (własność dywergencji KL w ogólności)
zweryfikowana na $1000$ losowych parach $(\boldsymbol\mu,\text{logvar})$ —
zawsze $\geq 0$ (z dokładnością do błędu numerycznego), nigdy ujemna.

## Dyscyplina skalowania strat: sum vs mean

To jest **praktyczny**, łatwy do przeoczenia błąd: strata rekonstrukcji jest
sumą/średnią po **pikselach** (wymiar $d_x$), a KL jest sumą/średnią po
**wymiarach latentnych** ($d_z$) — i obie są potem agregowane po **batchu**
($N$). Jeśli te konwencje nie są **jawne i konsekwentne**, relatywna waga
między dwoma członami straty zmienia się **niezamierzenie**.

Konwencja użyta w notebooku tego wykładu (jawnie, dla porównywalności):
suma po pikselach dla rekonstrukcji, suma po wymiarach latentnych dla KL,
**średnia po batchu** dla obu:

```python
recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / x.shape[0]
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
```

❌ "Użycie `reduction='mean'` zamiast `reduction='sum'` dla KL to nieistotny
szczegół implementacyjny."

✅ **Nieprawda — zmierzony, realny efekt.** Zamiana sumy na średnią po
wymiarach latentnych ($d_z=8$ w naszym eksperymencie) dzieli efektywną wagę
KL przez $d_z$ — model trenowany z tą "błędną" wersją osiągnął **gorszą**
(mniejszą) regularyzację: KL $3{,}76$ vs $14{,}4$ dla poprawnej wersji, przy
złudnie **"lepszej"** (niższej) stracie rekonstrukcji ($221{,}6$ vs
$230{,}7$) — bo model miał efektywnie dużo "swobodniejszy" dostęp do
przestrzeni latentnej. To nie jest tylko ostrzeżenie teoretyczne — zmierzone
w notebooku (komórka weryfikacyjna, nie w głównych Eksperymentach A-D).

### Wybór straty rekonstrukcji i założenie o rozkładzie obserwacji

Człon rekonstrukcji ELBO to $\mathbb E_q[\log p_\theta(x\mid\mathbf z)]$ —
**konkretna postać** zależy od założonego $p_\theta(x\mid\mathbf z)$:

- **Bernoulli** (pikselowe wartości w $[0,1]$ jako prawdopodobieństwa) →
  $-\log p_\theta(x\mid\mathbf z)$ sprowadza się do **binary cross-entropy**
  per piksel. Założenie: każdy piksel to niezależna zmienna Bernoulliego —
  **przybliżenie** dla danych skali szarości, nie dokładny model (piksele w
  praktyce nie są binarne ani niezależne), ale standardowe i wystarczające
  dla KMNIST/Fashion-MNIST w tym kursie.
- **Gauss o ustalonej wariancji** → $-\log p_\theta(x\mid\mathbf z)$
  sprowadza się (do addytywnej stałej) do **MSE**. Założenie: szum
  obserwacji jest Gaussowski o **ustalonej** (nie uczonej) wariancji —
  inaczej MSE nie jest dokładnym odpowiednikiem log-prawdopodobieństwa.

Używamy `binary_cross_entropy_with_logits` (nie osobno sigmoid + BCE) —
numerycznie stabilniejsze (jedna, połączona operacja, bez jawnego
`log(sigmoid(...))`, który może dać `-inf` dla ekstremalnych logitów).

## Beta-VAE: waga KL jako hiperparametr

Rozszerzenie: ważymy człon KL współczynnikiem $\beta$ (**reużyte** z modułu
04, gdzie $\beta$ oznacza momentum/współczynniki Adama — inny temat,
dezambiguacja kontekstem, zgodnie z ustaloną w kursie zasadą dla symboli
zbyt standardowych w literaturze, by je zmieniać):

$$L_{\beta\text{-VAE}} = -\mathbb E_{q_\phi}[\log p_\theta(x\mid\mathbf z)] + \beta \cdot D_{\mathrm{KL}}\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big)$$

❌ "Beta-VAE z $\beta\neq 1$ to wciąż ten sam ELBO, tylko 'dostrojony'."

✅ **Nieprawda.** Dla $\beta=1$ odzyskujemy standardowy ELBO dla modelu
probabilistycznego opisanego wyżej. Dla $\beta\neq 1$, $L_{\beta\text{-VAE}}$
**nie jest** $-$ELBO **tego samego** modelu probabilistycznego — to inna,
zmodyfikowana funkcja celu, motywowana empirycznie (chęcią silniejszej
regularyzacji przestrzeni latentnej), nie wyprowodzona z tych samych założeń
probabilistycznych. To rozróżnienie jest ważne: $\beta$-VAE nie "ulepsza"
wyprowodzenia ELBO, **odchodzi** od niego.

❌ "Zwiększanie $\beta$ zawsze daje bardziej interpretowalne / rozplecione
(*disentangled*) czynniki w przestrzeni latentnej."

✅ **Nieprawda w ogólności** — to była **hipoteza/motywacja** oryginalnej
pracy o $\beta$-VAE, nie gwarancja. Większe $\beta$ silniej "ściska" $q_\phi$
w stronę priora $N(0,I)$ — to **mierzalnie** redukuje KL (zobaczymy w
Eksperymencie D), ale redukcja KL **nie jest tym samym** co interpretowalność
czynników latentnych, i w skrajnym przypadku ($\beta$ bardzo duże) prowadzi
do **posterior collapse** (niżej), czyli **utraty** użytecznej informacji w
$\mathbf z$ — odwrotność "lepszej reprezentacji".

### Posterior collapse

Gdy KL $\to 0$ dla **wszystkich** przykładów, $q_\phi(\mathbf z\mid x)
\approx p(\mathbf z)$ niezależnie od $x$ — enkoder "poddaje się" i ignoruje
wejście, dekoder musi generować $\hat x$ praktycznie **niezależnie** od
konkretnego $x$. To jest degenerowany przypadek: niska strata KL, ale
rekonstrukcja drastycznie się pogarsza, bo $\mathbf z$ nie niesie już
informacji o $x$. Zobaczymy tendencję w tym kierunku w Eksperymencie D przy
dużym $\beta$ (bez pełnego kolapsu, w granicach budżetu epok tego kursu).

## Przestrzeń latentna: rekonstrukcja, interpolacja, generowanie — trzy różne operacje

❌ "Rekonstrukcja przykładu, interpolacja między przykładami i generowanie z
priora to w zasadzie to samo działanie dekodera."

✅ Dekoder jest **tą samą** funkcją we wszystkich trzech przypadkach — ale
**wejście** do dekodera pochodzi z trzech **różnych** źródeł:

1. **Rekonstrukcja**: $\mathbf z = \boldsymbol\mu(x)$ (albo próbka z
   $q_\phi(\mathbf z\mid x)$) dla **konkretnego, rzeczywistego** $x$.
2. **Interpolacja**: $\mathbf z = (1-\alpha)\boldsymbol\mu(x_a) +
   \alpha\boldsymbol\mu(x_b)$ dla $\alpha\in[0,1]$, między kodami **dwóch
   rzeczywistych** przykładów.
3. **Generowanie**: $\mathbf z \sim p(\mathbf z) = N(0,I)$ — próbka **z
   priora**, niezwiązana z żadnym konkretnym, rzeczywistym przykładem.

Zmierzone w notebooku (Eksperyment C): różnica pikselowa między rekonstrukcją
rzeczywistego obrazu i obrazem wygenerowanym z losowej próbki priora jest
**znacząca** (nie bliska zeru) — potwierdza, że to rzeczywiście różne
operacje, nie tylko różne nazwy tej samej czynności.

❌ "Interpolacja w przestrzeni latentnej VAE jest zawsze semantycznie
gładka, dla każdej pary przykładów."

✅ **Nie jest to gwarantowane.** Gładkość interpolacji zależy od tego, jak
dobrze model zorganizował przestrzeń latentną — dla małego modelu, krótkiego
treningu, albo dla par przykładów z odległych regionów przestrzeni, kroki
pośrednie mogą wyglądać na "rozmazane" przejścia bez wyraźnego sensu
wizualnego, nie na płynną, interpretowalną transformację. Sprawdzamy to
empirycznie w Eksperymencie C (różnice pikselowe między kolejnymi krokami
interpolacji), nie zakładamy z góry.

❌ "Dwuwymiarowa wizualizacja przestrzeni latentnej pokazuje wszystkie
własności modelu VAE."

✅ Model z **dwuwymiarową** przestrzenią latentną (Eksperyment C) jest
wybrany **specjalnie**, żeby dało się go łatwo zwizualizować — ale to
drastyczne ograniczenie pojemności modelu w porównaniu do np. $16$-wymiarowej
wersji z Eksperymentu B. Wykres 2D jest pouczający dla intuicji, ale **nie**
reprezentuje tego, jak wyglądałaby/działała przestrzeń latentna o wyższej
liczbie wymiarów — to dwa osobne eksperymenty z osobnymi wnioskami, nie jeden
wykres "pokazujący wszystko".

## Detekcja anomalii przez błąd rekonstrukcji — wprowadzenie (rozwinięte w laboratorium)

Intuicja: jeśli AE/VAE jest trenowany **tylko** na danych "normalnych", może
rekonstruować je lepiej niż dane nietypowe (bo nigdy nie widział wzorców
anomalii podczas treningu) — wysoki błąd rekonstrukcji **może** sugerować
anomalię.

❌ "Wysoki błąd rekonstrukcji jest wiarygodną, pełną definicją anomalii."

✅ **Nieprawda** — błąd rekonstrukcji to **jeden, niedoskonały** sygnał, nie
pełny model gęstości prawdopodobieństwa danych. Model może **dobrze**
rekonstruować przykład anomalii, który jest wizualnie/strukturalnie podobny
do klasy normalnej (zobaczymy to jako **zmierzony**, nie hipotetyczny
przypadek w laboratorium) — i może **źle** rekonstruować pewne przykłady
normalne, które są nietypowe w ramach swojej własnej klasy. Rozwijamy to w
pełni w laboratorium, z konkretnymi metrykami i analizą błędów.

❌ "Model generatywny (VAE) jest automatycznie lepszym detektorem anomalii
niż deterministyczny AE."

✅ **Nie jest to gwarantowane** — VAE optymalizuje inny cel (ELBO, z
regularyzacją KL), nie "optymalną separację normalne/anomalie". Który model
daje lepszą separację błędu rekonstrukcji między klasami jest **pytaniem
empirycznym** dla konkretnego zadania, nie czymś, co można zadeklarować z
góry na podstawie "VAE > AE" w ogólności.

## Związek z modułami 08 i 11

- **Moduł 08**: embeddingi uczone były przez zadanie związane z
  **podobieństwem/klasyfikacją** — tu reprezentacja (**e** albo **z**) wynika
  z **rekonstrukcji**, zupełnie innego celu treningowego. Dwa różne sposoby
  dojścia do "reprezentacji wektorowej danych" — nie warianty tego samego
  zadania.
- **Moduł 11**: model autoregresyjny generuje dane **sekwencyjnie, token po
  tokenie**, bez jawnej, ciągłej przestrzeni latentnej. VAE generuje dane
  **naraz**, z jawnej, ciągłej przestrzeni $\mathbf z$, z której można
  próbkować i interpolować — dwa różne, osobne paradygmaty modeli
  generatywnych, nie jeden "ogólny" sposób generowania danych.

## Co obserwujemy

Wszystkie liczby niżej pochodzą z rzeczywistego wykonania `examples.ipynb`
(CPU, ten komputer — nie uniwersalny benchmark), zbiór KMNIST, podzbiór
$8000$ obrazów treningowych / $2000$ testowych. Zobacz notebook po pełne
wyjście (wykresy, dodatkowe seeded).

- **Eksperyment A (AE, wpływ wymiaru latentnego).** $20$ epok, $3$ ziarna
  losowości: $d_e=2 \to$ MSE testowe $\approx 0{,}083$; $d_e=8 \to \approx
  0{,}063$; $d_e=32 \to \approx 0{,}052$. W tym konkretnym, skończonym runie
  **monotonicznie** lepsze z większym $d_e$ — ale traktujemy to jako
  obserwację tego eksperymentu, nie uniwersalne prawo (większy bottleneck
  może w innych warunkach przeuczyć się albo nie pomóc, zgodnie z
  ostrzeżeniem wyżej).
- **Eksperyment B (VAE, $d_z=16$).** Przez $20$ epok strata rekonstrukcji
  spadła z $\approx 384$ do $\approx 220$, a KL **wzrosła** z $\approx 1{,}6$
  do $\approx 16{,}8$ — model w trakcie treningu "odkrywa", że lepsza
  rekonstrukcja wymaga wykorzystania większej części przestrzeni latentnej
  (większego odejścia od priora), nie tylko minimalizacji jednego członu.
- **Eksperyment C (VAE, $d_z=2$).** Interpolacja między przykładem klasy $2$
  i przykładem klasy $9$ (7 kroków): różnice pikselowe między kolejnymi
  krokami $[0{,}0293,\ 0{,}0402,\ 0{,}0536,\ 0{,}0566,\ 0{,}0501,\ 0{,}0411]$
  — stabilne, bez nagłych skoków, lokalnie gładkie przejście, mimo małego,
  2-wymiarowego modelu. Rekonstrukcja rzeczywistego obrazu vs generacja z
  losowej próbki priora: średnia różnica pikselowa $0{,}0881$ — nietrywialna,
  jednoznacznie różne wyniki, nie przypadkowo podobne.
- **Eksperyment D (beta sweep, $d_z=8$).** $\beta=0$: KL $\approx 207$ (brak
  regularyzacji — przestrzeń latentna praktycznie nieukształtowana przez
  prior, zachowanie bliskie zwykłemu AE). $\beta=1$: KL $\approx 15$,
  rekonstrukcja $\approx 223$ (punkt odniesienia). $\beta=5$: KL $\approx
  5{,}6$. $\beta=25$: KL $\approx 0{,}67$ (silna kompresja w stronę priora),
  ale rekonstrukcja pogorszona do $\approx 322$ — oba zjawiska ("zbyt słaba"
  i "zbyt silna" regularyzacja) **rzeczywiście wystąpiły** w tym
  eksperymencie, nie trzeba było korygować narracji ani szukać innej
  konfiguracji.
- **Weryfikacja mean-vs-sum (KL reduction).** Opisana wyżej w sekcji o
  dyscyplinie skalowania strat — zmierzony, nie tylko zadeklarowany efekt.

## Częste błędne przekonania

(Zbiorcze podsumowanie — każde ma pełny ❌/✅ blok wyżej, w odpowiedniej
sekcji merytorycznej)

1. AE nie musi nauczyć się semantycznie interpretowalnej reprezentacji —
   zależy od architektonicznych ograniczeń, nie od samego faktu rekonstrukcji.
2. Dobra rekonstrukcja nie implikuje automatycznie dobrego modelu
   generatywnego (AE nie organizuje przestrzeni latentnej do próbkowania).
3. VAE nie generuje nowych danych poprzez dodawanie szumu do obrazów
   wejściowych — generuje poprzez próbkowanie z priora i dekodowanie.
4. Prior $p(\mathbf z)$ i przybliżony posterior $q_\phi(\mathbf z\mid x)$ to
   różne rozkłady — model minimalizuje ich odległość (KL), nie zrównuje je.
5. Reparameterization trick nie usuwa losowości — zmienia, gdzie w grafie
   obliczeniowym ta losowość wchodzi, umożliwiając propagację gradientu.
6. Minimalizacja samej straty rekonstrukcji nie jest pełnym celem treningowym
   VAE — człon KL jest integralną częścią $-$ELBO, nie opcjonalnym dodatkiem.
7. Wyższa dywergencja KL nie jest automatycznie oznaką "lepszego" modelu —
   to tylko miara odległości od priora, nie miara jakości.
8. Beta-VAE nie gwarantuje rozplecionych (*disentangled*) czynników
   latentnych — to była hipoteza motywująca, nie dowiedziona własność.
9. Próbkowanie z priora i rekonstrukcja konkretnego wejścia to dwie różne
   operacje, z różnymi wejściami do dekodera.
10. Niski błąd rekonstrukcji nie gwarantuje, że przykład pochodzi z
    rozkładu danych treningowych — model może dobrze rekonstruować
    nietypowe/nowe dane, jeśli są strukturalnie podobne do danych treningowych.
11. Wyższa wizualna jakość wygenerowanego obrazu nie dowodzi większej
    probabilistycznej wierności modelu (to odrębne kryteria oceny).
12. Dwuwymiarowa przestrzeń latentna jest użyteczna do wizualizacji, ale
    może ograniczać jakość rekonstrukcji względem przestrzeni o większym
    wymiarze — to kompromis, nie "darmowa" wygoda.
13. Model generatywny (VAE) nie jest automatycznie dobrym detektorem
    anomalii — to osobne zadanie z osobnymi wymaganiami.

## Implikacje inżynierskie

- **Koszt enkodera/dekodera** rośnie z rozmiarem sieci i wymiarem danych, nie
  z samym faktem "bycia VAE" — architektura MLP użyta tu jest celowo mała
  (CPU, kilka sekund na pełny trening), nie reprezentuje kosztu
  produkcyjnych modeli generatywnych na dużych obrazach.
- **Wymiar latentny to kompromis koszt/jakość/interpretowalność** — większy
  $d_z$/$d_e$ zwykle (w tym eksperymencie: zmierzono) daje lepszą
  rekonstrukcję, ale zmniejsza interpretowalność wizualizacji i zwiększa
  koszt pamięci/obliczeń przy dużej skali.
- **AE jako kompresja ma granice** — kompresja jest stratna (lossy) i
  ograniczona przez to, czego model faktycznie się nauczył na danych
  treningowych; nie jest to zamiennik ogólnego algorytmu kompresji.
- **Generowanie nowych danych z VAE wymaga zaprojektowanej przestrzeni
  latentnej** (prior + KL) — to jest właśnie różnica praktyczna względem AE,
  nie tylko teoretyczna.
- **Próg detekcji anomalii musi być wybierany z danych walidacyjnych, nigdy
  testowych** — koszt błędu (przepuszczona anomalia vs fałszywy alarm) jest
  decyzją biznesową/inżynierską, nie czysto statystyczną (rozwinięte w
  laboratorium).
- **Dryft danych** — model AE/VAE trenowany na danych z jednego okresu może
  przestać dobrze odzwierciedlać "normalność" danych z innego okresu; próg i
  model wymagają okresowej rewalidacji w praktycznym wdrożeniu.
- Denoising autoencoders i inne warianty AE (nie rozwijane tu szerzej) dodają
  regularyzację poprzez rekonstrukcję **z zaszumionego** wejścia — krótka
  wzmianka, nie osobny blok.

## Co warto zapamiętać

- AE: $\mathbf e = f_{\text{enc}}(x)$, $\hat x = f_{\text{dec}}(\mathbf e)$,
  strata rekonstrukcji — kompresja jest **wymuszona** tylko przez bottleneck
  węższy niż dane, nie przez sam fakt treningu na rekonstrukcję.
- VAE: probabilistyczny model z priorem $p(\mathbf z)$, dekoderem
  $p_\theta(x\mid\mathbf z)$ i przybliżonym posteriorem $q_\phi(\mathbf
  z\mid x)$ — enkoder zwraca $\boldsymbol\mu$/`logvar`, nie gotową próbkę.
- ELBO $\mathcal L$ (maksymalizowane) $=$ człon rekonstrukcji $-$ KL; strata
  treningowa $L_{\text{VAE}} = -\mathcal L$ (minimalizowana) — to konkretne,
  wyprowodzone wyrażenie, nie dowolnie złożona funkcja.
- $D_{\mathrm{KL}}(N(\mu,\sigma^2)\|N(0,1)) = \frac12(\sigma^2+\mu^2-1-\log\sigma^2)$
  per wymiar, zweryfikowane względem `torch.distributions`, zawsze $\geq 0$.
- Reparameterization: $\mathbf z = \boldsymbol\mu + \exp(0{,}5\cdot
  \text{logvar})\odot\boldsymbol\varepsilon$ — przenosi losowość do wejścia
  niezależnego od parametrów, umożliwiając propagację gradientu.
- Sum vs mean reduction zmienia relatywną wagę KL/rekonstrukcji — musi być
  jawna i konsekwentna, inaczej wynik treningu zmienia się niezamierzenie.
- Rekonstrukcja, interpolacja i generowanie z priora to trzy różne operacje
  z trzema różnymi źródłami wejścia do dekodera.
- Błąd rekonstrukcji to niedoskonały sygnał anomalii — nie pełny model
  gęstości, i nie gwarancja, że VAE "automatycznie" lepiej wykrywa anomalie
  niż AE.

## Pytania dla studentów

1. Masz AE z $d_e > d_x$ (overcomplete) wytrenowany bez żadnej dodatkowej
   regularyzacji. Jakiego zachowania się oczekujesz i dlaczego? Jak by się to
   zmieniło, gdybyś dodał silną regularyzację $L_2$ na wagi enkodera?
2. Wyjaśnij, dlaczego $-$ELBO ma postać "strata rekonstrukcji + KL", a nie
   dowolną inną kombinację dwóch strat. Co zmieniłoby się formalnie, gdybyś
   zmienił założony rozkład dekodera z Bernoulliego na Gaussa?
3. Masz wytrenowany VAE z $d_z=32$. Opisz dokładnie, czym różni się
   (1) rekonstrukcja konkretnego obrazu testowego, (2) interpolacja między
   dwoma obrazami testowymi, (3) generowanie nowego obrazu — w kontekście
   tego, co dokładnie podajesz do dekodera w każdym przypadku.
4. Dlaczego bardzo duże $\beta$ w Beta-VAE może prowadzić do posterior
   collapse? Jak rozpoznałbyś/rozpoznałabyś ten efekt, patrząc wyłącznie na
   wykresy straty rekonstrukcji i KL w czasie treningu?
5. Masz model AE trenowany tylko na "normalnych" obrazach, który ma niski
   błąd rekonstrukcji na pewnej klasie anomalii. Czy to oznacza, że model
   jest "zepsuty"? Jak byś to zbadał/zbadała i jak byś to zakomunikował/a
   jako ograniczenie systemu, nie błąd implementacji?

## Bibliografia

- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*.
  arXiv:1312.6114 (praca źródłowa VAE, ELBO, reparameterization trick).
- Baldi, P., & Hornik, K. (1989). *Neural networks and principal component
  analysis: Learning from examples without local minima*. Neural Networks,
  2(1), 53-58 (równoważność liniowego AE i PCA).
- Higgins, I., i in. (2017). *beta-VAE: Learning Basic Visual Concepts with
  a Constrained Variational Framework*. ICLR (Beta-VAE, hipoteza
  disentanglementu).
- Clanuwat, T., i in. (2018). *Deep Learning for Classical Japanese
  Literature*. arXiv:1812.01718 (KMNIST — zobacz też `data/registry.yml`).
- An, J., & Cho, S. (2015). *Variational Autoencoder based Anomaly
  Detection using Reconstruction Probability*. SNU Data Mining Center,
  Technical Report (detekcja anomalii przez błąd/prawdopodobieństwo
  rekonstrukcji — kontekst dla laboratorium).
- PyTorch documentation: `torch.distributions.kl_divergence`,
  `https://pytorch.org/docs/stable/distributions.html#torch.distributions.kl.kl_divergence`.
- PyTorch documentation: `torch.nn.functional.binary_cross_entropy_with_logits`,
  `https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html`.
