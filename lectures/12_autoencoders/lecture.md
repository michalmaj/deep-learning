Język: [English](lecture.en.md) | **Polski**

# Wykład 12. Autoenkodery, VAE i przestrzeń latentna

## Cele wykładu

Po tym wykładzie powinny być jasne następujące zagadnienia:

1. uczenie przez rekonstrukcję jako forma uczenia bez etykiet,
2. budowa autoenkodera i rola bottlenecku,
3. różnica między deterministycznym AE a probabilistycznym VAE,
4. wyprowadzenie ELBO i związek ze stratą treningową,
5. reparametryzacja oraz wzór na KL dla diagonalnego rozkładu Gaussa,
6. ograniczenia detekcji anomalii opartej na błędzie rekonstrukcji.

## Po co uczyć się rekonstrukcji

W module 8 reprezentacje powstawały w zadaniu związanym z podobieństwem lub
klasyfikacją. Moduł 11 wprowadził generowanie autoregresyjne, czyli
przewidywanie kolejnych tokenów. Rekonstrukcja jest trzecim podejściem. Model
otrzymuje dane $x$ bez etykiety i ma je odtworzyć z wewnętrznej, zazwyczaj
mniejszej reprezentacji.

Taki cel nie gwarantuje użytecznej reprezentacji. Sieć uczy się tylko tego, co
pozwala zmniejszyć błąd rekonstrukcji. Przy zbyt dużym bottlenecku może niemal
skopiować wejście. Dopiero ograniczona pojemność albo dodatkowa regularyzacja
wymusza wybór informacji istotnych dla odtworzenia danych.

W tym module używane są małe sieci MLP. Modele dyfuzyjne pojawiają się w module
13, a architektura Transformera i autoregresja zostały omówione odpowiednio w
modułach 10 i 11.

## Autoenkoder

Autoenkoder składa się z enkodera, reprezentacji pośredniej i dekodera:

$$mathbf e = f_{\text{enc}}(x), \qquad \hat x = f_{\text{dec}}(\mathbf e).$$

$x \in \mathbb R^{d_x}$ oznacza wejście, $\mathbf e \in \mathbb R^{d_e}$
reprezentację w bottlenecku, a $\hat x$ rekonstrukcję. Dla danych ciągłych
często stosuje się MSE:

$$L_{\text{AE}} = \frac{1}{N}\sum_{n=1}^N
\lVert x^{(n)} - \hat x^{(n)} \rVert_2^2.$$

### Przykład liczbowy

Niech

$$x = [1{,}0,\ 0{,}5,\ -0{,}5,\ 0{,}0] \in \mathbb R^4$$

oraz

$$W_{\text{enc}} =
\begin{pmatrix}1&0&0&0\\0&1&1&0\end{pmatrix}.$$

Liniowy enkoder bez biasu daje

$$\mathbf e = W_{\text{enc}}x =
\begin{pmatrix}1{,}0\\0{,}0\end{pmatrix}.$$

Przy $W_{\text{dec}}=W_{\text{enc}}^\top$ rekonstrukcja wynosi

$$\hat x = [1{,}0,\ 0{,}0,\ 0{,}0,\ 0{,}0],$$

a suma kwadratów błędów jest równa

$$L_{\text{AE}} = 0 + 0{,}25 + 0{,}25 + 0 = 0{,}5.$$

Wynik został odtworzony programowo w notebooku. Druga współrzędna kodu utraciła
informację o różnicy między $x_2$ i $x_3$. Sama architektura enkoder-dekoder nie
zapewnia więc dobrej rekonstrukcji. Wagi muszą zostać wytrenowane dla danych i
wybranej funkcji straty.

## Skąd bierze się kompresja

W autoenkoderze undercomplete zachodzi $d_e < d_x$. Węższy bottleneck wymusza
utratę części informacji dla ogólnych danych pełnego rzędu. W wersji
overcomplete, gdzie $d_e \geq d_x$, taki mechanizm nie istnieje. Bez
regularyzacji model może nauczyć się mapowania bliskiego tożsamości.

W notebooku porównano oba przypadki na syntetycznych danych w $\mathbb R^8$.
Model z $d_e=32$ osiągnął MSE bliskie zeru, natomiast model z $d_e=2$ nie
zszedł poniżej około $0{,}69$. Kompresja wynikała z konkretnego ograniczenia
architektury, a nie z samej nazwy modelu.

Stosuje się również inne ograniczenia, na przykład rzadkość aktywacji,
regularyzację normy lub rekonstrukcję zaszumionego wejścia w denoising
autoencoderze.

## Liniowy autoenkoder a PCA

Dla liniowego enkodera i dekodera, bez aktywacji i ze stratą MSE, globalne
optimum rozpina tę samą podprzestrzeń co pierwsze $d_e$ składowych PCA. Jest to
wynik matematyczny dla rozwiązania optymalnego, a nie gwarancja dotycząca
dowolnego stanu treningu.

Wagi enkodera nie muszą być identyczne z wektorami własnymi PCA. Mogą tworzyć
inną bazę tej samej podprzestrzeni, bez ortonormalności i uporządkowania według
wariancji. Zgodność podprzestrzeni należy mierzyć kątami głównymi, a nie przez
bezpośrednie porównanie znaków i kolejności wektorów.

Przed osiągnięciem zbieżności możliwy jest niski błąd rekonstrukcji bez pełnej
zgodności z podprzestrzenią PCA. MSE i kąty główne opisują dwa różne aspekty
stanu treningu.

## Dlaczego zwykły AE nie wystarcza do generowania

Do dekodera AE można przekazać dowolny wektor $\mathbf e$, ale trening nie
określa, które obszary $\mathbb R^{d_e}$ odpowiadają rzeczywistym danym.
Rekonstrukcja uczy dekoder tylko na kodach uzyskanych z przykładów
treningowych. Losowy punkt poza tymi obszarami może dać bezsensowny wynik.

Potrzebny jest zatem znany rozkład przestrzeni latentnej, z którego można
próbkować. Taką rolę pełni prior w VAE.

## Wariacyjny autoenkoder

VAE opisuje łączny rozkład obserwacji $x$ i zmiennej latentnej
$\mathbf z \in \mathbb R^{d_z}$. Składają się na niego:

- prior $p(\mathbf z)$, zwykle $N(0,I)$,
- dekoder $p_\theta(x\mid\mathbf z)$,
- posterior $p(\mathbf z\mid x)$.

Z twierdzenia Bayesa

$$p(\mathbf z\mid x) =
\frac{p_\theta(x\mid\mathbf z)p(\mathbf z)}{p(x)}, \qquad
p(x)=\int p_\theta(x\mid\mathbf z)p(\mathbf z)\,d\mathbf z.$$

Całka po przestrzeni latentnej jest na ogół nieobliczalna dla dekodera
neuronowego. Dlatego posterior przybliża się rozkładem
$q_\phi(\mathbf z\mid x)$ z wygodnej rodziny. Enkoder o parametrach $\phi$
uczy się tak, aby przybliżenie było bliskie prawdziwemu posteriorowi.

## ELBO

Celem jest maksymalizacja $\log p_\theta(x)$. Po wprowadzeniu
$q_\phi(\mathbf z\mid x)$ można zapisać

$$\log p_\theta(x)=
\log \mathbb E_{q_\phi(\mathbf z\mid x)}
\left[\frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right].$$

Z nierówności Jensena wynika dolne ograniczenie

$$\log p_\theta(x) \geq
\mathbb E_{q_\phi(\mathbf z\mid x)}
\left[\log\frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right]
=: \mathcal L(\theta,\phi;x).$$

$\mathcal L$ jest ELBO, czyli *Evidence Lower Bound*. Po rozłożeniu rozkładu
łącznego otrzymuje się

$$\mathcal L(\theta,\phi;x)=
\underbrace{\mathbb E_{q_\phi}\left[\log p_\theta(x\mid\mathbf z)\right]}
_{\text{rekonstrukcja}}
-
\underbrace{D_{\mathrm{KL}}
\left(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\right)}_{\text{KL}}.$$

Pierwszy człon premiuje dobrą rekonstrukcję. Drugi ogranicza odległość
przybliżonego posteriora od priora. ELBO jest maksymalizowane, natomiast kod
treningowy zwykle minimalizuje jego wartość przeciwną:

$$L_{\text{VAE}}=-\mathcal L
=-\mathbb E_{q_\phi}\left[\log p_\theta(x\mid\mathbf z)\right]
+D_{\mathrm{KL}}
\left(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\right).$$

Suma straty rekonstrukcji i KL nie jest dowolną konstrukcją. Wynika z
konkretnych założeń o priorze, posteriorze przybliżonym i rozkładzie
obserwacji. Zmiana modelu obserwacji zmienia postać pierwszego członu.

## Gaussowska przestrzeń latentna

Standardowy VAE przyjmuje

$$p(\mathbf z)=N(0,I), \qquad
q_\phi(\mathbf z\mid x)=
N\left(\boldsymbol\mu(x),
\operatorname{diag}(\boldsymbol\sigma^2(x))\right).$$

Enkoder zwraca dwa wektory: średnią $\boldsymbol\mu$ i logarytm wariancji
$\log\boldsymbol\sigma^2$, oznaczany w kodzie jako `logvar`. Przechowywanie
logarytmu ułatwia zachowanie dodatniej wariancji i stabilne obliczenia.

### Reparametryzacja

Bezpośrednie losowanie nie daje zwykłej ścieżki gradientu względem parametrów
rozkładu. Losowość przenosi się więc do niezależnej zmiennej
$\boldsymbol\varepsilon$:

$$\mathbf z = \boldsymbol\mu
+\exp(0{,}5\cdot\text{logvar})\odot\boldsymbol\varepsilon,
\qquad \boldsymbol\varepsilon\sim N(0,I).$$

Losowość pozostaje w modelu, lecz $\mathbf z$ staje się różniczkowalną funkcją
`mu` i `logvar`. Gradient może dzięki temu dotrzeć do enkodera.

Dla $\boldsymbol\mu=[0{,}5,-0{,}3]$, `logvar` $=[-0{,}2,0{,}4]$ i
$\boldsymbol\varepsilon=[1{,}0,-0{,}5]$ otrzymuje się

$$\exp(-0{,}1)=0{,}9048, \qquad \exp(0{,}2)=1{,}2214,$$

$$\mathbf z=[1{,}4048,-0{,}9107].$$

Dla $\mathcal L=\sum_i z_i^2$ zachodzi
$\partial\mathcal L/\partial\boldsymbol\mu=2\mathbf z$. Notebook potwierdza
wartości `mu.grad` $=[2{,}8097,-1{,}8214]$ oraz `logvar.grad`
$=[1{,}2711,0{,}5562]$. Oba gradienty są skończone.

### Dywergencja KL

Dla jednego wymiaru

$$D_{\mathrm{KL}}
\left(N(\mu_i,\sigma_i^2)\,\|\,N(0,1)\right)
=\frac12\left(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2\right).$$

Po zsumowaniu po $d_z$ wymiarach

$$D_{\mathrm{KL}}
=-\frac12\sum_{i=1}^{d_z}
\left(1+\text{logvar}_i-\mu_i^2-e^{\text{logvar}_i}\right).$$

Dla wartości z poprzedniego przykładu wkłady wynoszą $0{,}1344$ i $0{,}0909$,
czyli razem $0{,}2253$. Wynik jest zgodny z
`torch.distributions.kl_divergence` przy `atol=1e-6`. Nieujemność KL została
też sprawdzona na 1000 losowych parach parametrów.

## Skalowanie składników straty

Rekonstrukcja jest agregowana po pikselach, KL po wymiarach latentnych, a oba
człony po batchu. Niejawna zmiana `sum` na `mean` zmienia ich względną wagę.
W notebooku obowiązuje suma po cechach i średnia po batchu:

```python
recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / x.shape[0]
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
```

Przy $d_z=8$ średnia zamiast sumy po wymiarach latentnych podzieliła efektywną
wagę KL przez osiem. W zmierzonym przebiegu KL spadło z $14{,}4$ do $3{,}76$,
a strata rekonstrukcji z $230{,}7$ do $221{,}6$. Niższa rekonstrukcja nie
oznaczała lepszego VAE, lecz słabszą regularyzację przestrzeni latentnej.

### Model obserwacji i strata rekonstrukcji

Postać rekonstrukcji zależy od $p_\theta(x\mid\mathbf z)$:

- rozkład Bernoulliego prowadzi do binary cross-entropy na piksel,
- rozkład Gaussa o ustalonej wariancji prowadzi, z dokładnością do stałej, do MSE.

Bernoulli jest przybliżeniem dla obrazów w skali szarości, ponieważ piksele nie
są naprawdę binarne ani niezależne. W tym kursie wystarcza dla KMNIST i
Fashion-MNIST. Funkcja `binary_cross_entropy_with_logits` łączy sigmoid i BCE
w jednej stabilnej numerycznie operacji.

## Beta-VAE i posterior collapse

Beta-VAE nadaje członowi KL wagę $\beta$:

$$L_{\beta\text{-VAE}}=
-\mathbb E_{q_\phi}[\log p_\theta(x\mid\mathbf z)]
+\beta D_{\mathrm{KL}}
\left(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\right).$$

Dla $\beta=1$ jest to standardowe ujemne ELBO. Dla $\beta\neq1$ powstaje
zmodyfikowany cel treningowy, a nie ELBO tego samego modelu probabilistycznego.
Większe $\beta$ zwykle silniej zbliża posterior do priora, ale nie gwarantuje
interpretowalnych lub rozplecionych czynników.

Przy zbyt dużej wadze KL może dojść do posterior collapse. Wtedy
$q_\phi(\mathbf z\mid x)\approx p(\mathbf z)$ niezależnie od wejścia. Kod
przestaje przenosić informację o $x$, a rekonstrukcja się pogarsza. Niskie KL
nie jest więc samodzielną miarą jakości modelu.

## Trzy operacje w przestrzeni latentnej

Ten sam dekoder może otrzymać kod z trzech różnych źródeł:

1. rekonstrukcja korzysta z $\boldsymbol\mu(x)$ lub próbki z
   $q_\phi(\mathbf z\mid x)$ dla konkretnego wejścia,
2. interpolacja używa
   $\mathbf z=(1-\alpha)\boldsymbol\mu(x_a)+\alpha\boldsymbol\mu(x_b)$,
3. generowanie losuje $\mathbf z\sim N(0,I)$ bez powiązania z konkretnym
   przykładem.

Nie są to trzy nazwy tej samej czynności. W eksperymencie C średnia różnica
pikselowa między rekonstrukcją a próbką z priora wyniosła $0{,}0881$.

Interpolacja także nie musi być semantycznie gładka. Zależy od treningu,
pojemności modelu i położenia obu końców w przestrzeni latentnej. Model z
$d_z=2$ ułatwia wizualizację, ale ma znacznie mniejszą pojemność niż wersja
$d_z=16$. Wykres dwuwymiarowy nie opisuje wszystkich własności większego
modelu.

## Detekcja anomalii przez rekonstrukcję

AE trenowany wyłącznie na danych normalnych może odtwarzać je lepiej niż dane
nietypowe. Wysoki błąd rekonstrukcji staje się wtedy wynikiem anomalii.

Jest to sygnał niedoskonały. Nietypowy przykład podobny strukturalnie do klasy
normalnej może zostać zrekonstruowany dobrze. Z kolei rzadki przykład normalny
może dać wysoki błąd. Próg musi powstać na danych walidacyjnych, nigdy na
zbiorze testowym, a jego wybór powinien uwzględniać koszt fałszywych alarmów i
przeoczonych anomalii.

VAE nie jest z definicji lepszym detektorem niż AE. Optymalizuje ELBO, a nie
separację klas według błędu rekonstrukcji. Porównanie obu modeli jest pytaniem
empirycznym dla konkretnego zbioru.

## Powiązania z wcześniejszymi modułami

- W module 8 reprezentacja wynikała z podobieństwa lub klasyfikacji. Tutaj
  wynika z rekonstrukcji.
- Model z modułu 11 generował token po tokenie. VAE dekoduje próbkę z jawnej,
  ciągłej przestrzeni latentnej.

Są to różne cele treningowe i różne paradygmaty generowania, choć wszystkie
prowadzą do wektorowych reprezentacji danych.

## Wyniki eksperymentów

Poniższe liczby pochodzą z wykonania `examples.ipynb` na CPU. Użyto podzbioru
KMNIST z 8000 obrazów treningowych i 2000 testowych. Nie jest to uniwersalny
benchmark.

- **Eksperyment A.** Po 20 epokach i dla 3 ziaren testowe MSE wyniosło około
  $0{,}083$ dla $d_e=2$, $0{,}063$ dla $d_e=8$ i $0{,}052$ dla $d_e=32$.
  Większy bottleneck pomógł w tym przebiegu, ale nie stanowi to ogólnego prawa.
- **Eksperyment B.** Dla VAE z $d_z=16$ rekonstrukcja spadła z około $384$ do
  $220$, natomiast KL wzrosło z około $1{,}6$ do $16{,}8$. Lepsza
  rekonstrukcja wymagała większego odejścia posteriora od priora.
- **Eksperyment C.** Dla $d_z=2$ interpolacja między klasami 2 i 9 miała
  kolejne różnice pikselowe $[0{,}0293, 0{,}0402, 0{,}0536, 0{,}0566,
  0{,}0501, 0{,}0411]$. Przejście było lokalnie gładkie, bez nagłych skoków.
- **Eksperyment D.** Dla $d_z=8$ wartość $\beta=0$ dała KL około $207$.
  Przy $\beta=1$ uzyskano KL około $15$ i rekonstrukcję około $223$. Dla
  $\beta=5$ KL spadło do $5{,}6$, a dla $\beta=25$ do $0{,}67$, lecz
  rekonstrukcja pogorszyła się do około $322$.

Ostatni eksperyment pokazuje koszt zarówno zbyt słabej, jak i zbyt silnej
regularyzacji.

## Implikacje inżynierskie

- Wymiar latentny tworzy kompromis między kosztem, jakością rekonstrukcji i
  łatwością interpretacji.
- Kompresja AE jest stratna i zależna od danych treningowych. Nie zastępuje
  ogólnego algorytmu kompresji.
- Generowanie z VAE wymaga priora i regularyzacji KL, które porządkują
  przestrzeń latentną.
- Detektor anomalii wymaga osobnej walidacji progu i analizy błędów.
- Dryft danych może zmienić rozkład normalnych przykładów, dlatego model i próg
  wymagają okresowej ponownej oceny.
- Koszt zależy przede wszystkim od architektury i wymiaru danych. Małe MLP z
  tego modułu nie odzwierciedla skali produkcyjnych modeli obrazu.

## Najważniejsze wnioski

- AE kompresuje tylko wtedy, gdy architektura lub regularyzacja naprawdę to
  wymusza.
- VAE uczy rozkładu nad kodem latentnym, a enkoder zwraca `mu` i `logvar`.
- ELBO jest maksymalizowane, a strata $L_{\text{VAE}}=-\mathcal L$
  minimalizowana.
- Reparametryzacja zachowuje losowość i jednocześnie umożliwia propagację
  gradientu.
- Konwencje `sum` i `mean` zmieniają rzeczywistą wagę składników straty.
- Rekonstrukcja, interpolacja i generowanie korzystają z innych kodów
  wejściowych.
- Błąd rekonstrukcji nie jest pełną definicją anomalii.

## Pytania kontrolne

1. Jakiego zachowania można oczekiwać po overcomplete AE bez regularyzacji?
2. Dlaczego ujemne ELBO ma postać rekonstrukcja plus KL? Co zmieniłby
   gaussowski zamiast bernoullowskiego model obserwacji?
3. Jak różnią się kody przekazywane do dekodera przy rekonstrukcji,
   interpolacji i generowaniu?
4. Jak na wykresach rekonstrukcji i KL rozpoznać posterior collapse?
5. Jak zbadać niski błąd rekonstrukcji dla klasy anomalii i opisać go jako
   ograniczenie metody, a nie błąd implementacji?

## Bibliografia

- Kingma, D. P., Welling, M. (2013). *Auto-Encoding Variational Bayes*.
  arXiv:1312.6114.
- Baldi, P., Hornik, K. (1989). *Neural networks and principal component
  analysis: Learning from examples without local minima*. Neural Networks,
  2(1), 53-58.
- Higgins, I. i in. (2017). *beta-VAE: Learning Basic Visual Concepts with a
  Constrained Variational Framework*. ICLR.
- Clanuwat, T. i in. (2018). *Deep Learning for Classical Japanese
  Literature*. arXiv:1812.01718.
- An, J., Cho, S. (2015). *Variational Autoencoder based Anomaly Detection
  using Reconstruction Probability*. SNU Data Mining Center.
- Dokumentacja PyTorch: `torch.distributions.kl_divergence` oraz
  `torch.nn.functional.binary_cross_entropy_with_logits`.
