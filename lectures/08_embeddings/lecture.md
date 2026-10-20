Język: [English](lecture.en.md) | **Polski**

# Wykład 08 — Od reprezentacji do podobieństwa: embeddingi i geometria przestrzeni cech

## Cele wykładu

1. Zrozumieć, czym jest embedding — wektorowa reprezentacja obiektu — i czym różni
   się od ręcznie projektowanych cech oraz od wyjścia klasyfikatora (logits,
   prawdopodobieństwa).
2. Poprawnie liczyć i interpretować odległość euklidesową oraz podobieństwo
   cosinusowe, wiedzieć kiedy dają różny wynik i dlaczego.
3. Rozumieć rolę normalizacji L2 i wiedzieć, że nie gwarantuje ona lepszych wyników.
4. Umieć zweryfikować empirycznie, czy wyuczone reprezentacje faktycznie dają bardziej
   użyteczne sąsiedztwo niż surowe dane — nie zakładać tego z góry.
5. Rozumieć ograniczenia wizualizacji wysokowymiarowych reprezentacji (PCA, krótko
   t-SNE/UMAP) — co pokazują, a czego nie dowodzą.
6. Poznać `nn.Embedding` jako uczoną tablicę wektorów indeksowaną identyfikatorami —
   przygotowanie pod moduły 09-10 (sekwencje, uwaga, Transformer).

## Dlaczego to jest ważne

W module 7 backbone CNN posłużył jako ekstraktor cech — wzięliśmy jego wyjście i
podaliśmy do nowej głowicy klasyfikacyjnej, nie pytając właściwie, *czym* jest ten
wektor pośredni ani *dlaczego* w ogóle działa. Ten wykład odpowiada na to pytanie
wprost: czym jest reprezentacja wektorowa, kiedy odległość między dwoma takimi
wektorami coś znaczy, i jak tę własność wykorzystać (wyszukiwanie podobnych obiektów,
rekomendacje, wyszukiwanie semantyczne) — oraz kiedy *nie* znaczy ona tego, czego
byśmy się intuicyjnie spodziewali.

## Intuicja

**Reprezentacja** to sposób zapisania obiektu (obrazu, słowa, użytkownika) jako coś, na
czym można obliczać. Surowe piksele obrazu też są reprezentacją — wektorem liczb — ale
reprezentacją **ręcznie narzuconą** przez sposób zapisu danych (siatka pikseli, kolejność
kanałów), nie przez to, co w obrazie istotne. **Embedding** to reprezentacja
**wyuczona**: sieć neuronowa, trenowana na jakimś zadaniu, sama decyduje, jakie
wymiary wektora są przydatne, w oparciu o to, co pomagało minimalizować stratę.

❌ "Skoro zapisałem dane jako wektor liczb, odległość między wektorami automatycznie
coś znaczy."

✅ Odległość w przestrzeni liczbowej ma sens geometryczny zawsze (to matematyka), ale
**sens semantyczny** (czy bliskie wektory odpowiadają "podobnym" obiektom w sensie,
który nas interesuje) zależy całkowicie od tego, **jak** wektor powstał. Wektor
surowych pikseli i wektor cech z wytrenowanej CNN mają tę samą postać matematyczną
(ciąg liczb), ale zupełnie inną użyteczność geometryczną — zweryfikujemy to liczbowo
w eksperymencie B, nie założymy.

## Formalny model

Embedding to odwzorowanie obiektu wejściowego na wektor o ustalonej liczbie wymiarów:

$$\mathbf e = f_\theta(x), \qquad x \in \mathcal X,\quad \mathbf e \in \mathbb R^d$$

gdzie $\mathcal X$ to **przestrzeń wejściowa** (np. obrazy $3\times224\times224$,
zdania, identyfikatory produktów), $\mathbb R^d$ to **przestrzeń reprezentacji** o
wymiarze $d$ (np. $d=512$ dla wyjścia `avgpool` ResNet18), a $\theta$ to parametry
odwzorowania (wagi sieci).

**Uwaga o notacji.** Używamy pogrubionego $\mathbf e$ dla wektora embeddingu — **nowy
symbol**, nie $z$. W tym kursie $z$ oznacza od modułu 2 **logity** (wyjście warstwy
*przed* sigmoid/softmax, np. `BCEWithLogitsLoss` operuje na $z$) — embedding **nie
jest** logitem (zobacz niżej, dlaczego to dwa różne obiekty). $\mathbf e$ nie ma nic
wspólnego z liczbą Eulera $e$ (pojawiała się tylko przy $e^{-z}$ w module 2, nie w tym
wykładzie) ani z indeksem epoki $e$ z modułu 5 — w tym dokumencie $\mathbf e$ zawsze
pogrubione oznacza wektor, kontekst jest jednoznaczny.

**Embedding z warstw ukrytych CNN.** Backbone CNN (moduł 6-7) to złożenie warstw
konwolucyjnych kończące się (w ResNet18) globalnym poolingiem — wyjście tego poolingu,
**przed** finalną warstwą `fc`, to właśnie $\mathbf e$. Trzy różne obiekty, które łatwo
pomylić:

| Obiekt | Skąd | Wymiar (ResNet18, 10 klas) | Do czego służy |
|---|---|---|---|
| Embedding $\mathbf e$ | wyjście `avgpool` | $512$ | porównania podobieństwa, wyszukiwanie |
| Logity $z$ | wyjście `fc` (przed softmax) | $10$ (liczba klas) | wejście do funkcji straty |
| Prawdopodobieństwa $\hat y=\text{softmax}(z)$ | `softmax(fc(...))` | $10$ | interpretacja "pewności" modelu |

❌ "Mogę użyć logitów (wyjścia `fc`) zamiast embeddingu z `avgpool` — to w sumie to
samo, tylko inny krok."

✅ Logity są **specyficzne dla konkretnego zadania klasyfikacji** (10 albo 1000
konkretnych klas) — ich wymiarowość i znaczenie są przywiązane do etykiet, na których
model był trenowany. Embedding z `avgpool` to reprezentacja **ogólniejsza**,
niezależna od konkretnej głowicy — to dlatego w module 7 podmienialiśmy `fc`, a nie
`avgpool`, przy przenoszeniu modelu na nowe zadanie. Dla porównań podobieństwa między
obrazami **embedding, nie logity**, jest właściwym wyborem — logity dwóch obrazów
różnych, nieznanych dla oryginalnego zadania klas mogą być bliskie z przypadkowych
powodów (obie klasy "nie pasują" do żadnej ze znanych kategorii w podobny sposób), nie
dlatego, że same obrazy są podobne.

## Matematyka: podobieństwo i odległość

**Odległość euklidesowa:**

$$d(\mathbf a,\mathbf b) = \|\mathbf a - \mathbf b\|_2 = \sqrt{\textstyle\sum_i (a_i-b_i)^2}$$

**Iloczyn skalarny** i **podobieństwo cosinusowe:**

$$\mathbf a \cdot \mathbf b = \textstyle\sum_i a_i b_i, \qquad \cos(\mathbf a,\mathbf b) = \frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|_2\,\|\mathbf b\|_2} \in [-1,1]$$

Cosine similarity mierzy **kąt** między wektorami, ignorując ich długość. Odległość
euklidesowa mierzy **różnicę zarówno kierunku, jak i długości**. To dwa różne pytania
("czy wskazują podobny kierunek?" vs "jak daleko od siebie leżą ich końce?") i mogą dać
**różne rankingi najbliższych sąsiadów** — pokazujemy to niżej liczbowo, nie
deklaratywnie.

**Normalizacja L2:** $\hat{\mathbf e} = \mathbf e / \|\mathbf e\|_2$, tak że
$\|\hat{\mathbf e}\|_2 = 1$. Dla wektorów znormalizowanych do długości jednostkowej
zachodzi:

$$\|\hat{\mathbf a}-\hat{\mathbf b}\|_2^2 = 2 - 2\cos(\hat{\mathbf a},\hat{\mathbf b})$$

(zweryfikowane numerycznie na losowych wektorach w notebooku — różnica rzędu $10^{-6}$,
w granicach błędu zmiennoprzecinkowego). Oznacza to, że **dla wektorów znormalizowanych
L2, ranking najbliższych sąsiadów wg odległości euklidesowej i wg cosine similarity
jest identyczny** (monotoniczna transformacja tej samej wielkości) — ale **tylko po
normalizacji**. Przed normalizacją obie metryki mogą się różnić.

❌ "Normalizacja L2 zawsze poprawia jakość wyszukiwania."

✅ Normalizacja **zmienia geometrię** (usuwa wpływ długości wektora na odległość), ale
czy to pomaga, zależy od tego, czy długość wektora w Twojej reprezentacji niesie
użyteczną informację, czy tylko szum. Sprawdzamy to eksperymentalnie w laboratorium.

**Przypadek brzegowy: wektor zerowy.** $\mathbf e = \mathbf 0 \Rightarrow
\|\mathbf e\|_2 = 0 \Rightarrow \mathbf e/\|\mathbf e\|_2 = \mathbf 0/0$ — **niezdefiniowane**
(w PyTorch: `nan`, zweryfikowane w notebooku). W praktyce dodaje się małe $\varepsilon$
do mianownika (`e / (e.norm() + eps)`) — nie jest to kosmetyka, to jedyny sposób, by
funkcja nie wybuchała na zdegenerowanym wejściu.

**k najbliższych sąsiadów (k-NN).** Dla zapytania $\mathbf q$ i zbioru wektorów
$\{\mathbf e_1,\ldots,\mathbf e_N\}$, $k$-NN to $k$ wektorów o najmniejszej
$d(\mathbf q,\mathbf e_i)$ (albo największym $\cos(\mathbf q,\mathbf e_i)$). **Uwaga o
notacji:** $k$ tutaj to liczba sąsiadów — **inne znaczenie** niż $k$ w module 6
(rozmiar filtra konwolucyjnego $k\times k$). Ten sam symbol, inny kontekst — rozróżnia
go temat wykładu, nie ma ryzyka pomyłki w praktyce, ale warto to nazwać wprost.

## Przykład rozwiązany: różne rankingi dla różnych metryk

Zapytanie $\mathbf q=(1,0)$. Dwa kandydaci: $\mathbf a=(10,0)$, $\mathbf b=(0{,}9,
0{,}9)$.

$$d(\mathbf q,\mathbf a) = \|(1,0)-(10,0)\|_2 = 9{,}0 \qquad d(\mathbf q,\mathbf b) = \|(1,0)-(0{,}9,0{,}9)\|_2 \approx 0{,}906$$
$$\cos(\mathbf q,\mathbf a) = \frac{10}{1\cdot 10} = 1{,}0 \qquad \cos(\mathbf q,\mathbf b) = \frac{0{,}9}{1\cdot\sqrt{1{,}62}} \approx 0{,}707$$

**Odległość euklidesowa wskazuje $\mathbf b$ jako bliższego sąsiada** (mniejsza
odległość: $0{,}906 < 9{,}0$), **cosine similarity wskazuje $\mathbf a$** (większe
podobieństwo: $1{,}0 > 0{,}707$) — bo $\mathbf a$ wskazuje *dokładnie* ten sam kierunek
co zapytanie, mimo dużej normy. Zweryfikowane programowo w notebooku. To nie jest
sztuczny przykład patologiczny — to dokładnie sytuacja, w której **skala wektora**
(norma $\mathbf a$ jest $10\times$ większa niż $\mathbf q$) **dominuje odległość
euklidesową**, podczas gdy cosine similarity jest na nią odporne z definicji.

## Redukcja wymiarowości: intuicja PCA

Embeddingi mają zwykle dziesiątki-setki wymiarów (ResNet18: $512$) — nie da się ich
"zobaczyć" wprost. PCA (*Principal Component Analysis*) znajduje kierunki
$\mathbf u_1,\mathbf u_2,\ldots$ (wektory jednostkowe, wzajemnie ortogonalne) w
przestrzeni embeddingów, wzdłuż których dane mają **największą wariancję** — $\mathbf
u_1$ to kierunek największego rozrzutu danych, $\mathbf u_2$ to kierunek największego
rozrzutu **wśród pozostałych, ortogonalnych do $\mathbf u_1$**, i tak dalej.
Projekcja punktu na dwa pierwsze kierunki:

$$\mathbf e' = (\mathbf u_1^\top \mathbf e,\ \mathbf u_2^\top \mathbf e) \in \mathbb R^2$$

daje coś, co można narysować na płaszczyźnie. Implementacyjnie (notebook): liczymy
macierz kowariancji embeddingów i jej rozkład własny (`torch.linalg.eigh`) —
$\mathbf u_1,\mathbf u_2$ to wektory własne odpowiadające dwóm największym wartościom
własnym. Nie wymagamy pełnego wyprowadzenia SVD — wystarczy rozumieć, **co PCA
optymalizuje** (wariancję projekcji) i **co to oznacza dla zachowanej informacji**:
kierunki o **małej** wariancji (np. $\mathbf u_{500}, \mathbf u_{501}$) są odrzucane, a
wraz z nimi — cokolwiek w nich było zakodowane.

❌ "PCA w 2D pokazuje, czy klasy są dobrze rozróżnialne w pełnej przestrzeni."

✅ PCA pokazuje kierunki **największej wariancji**, niekoniecznie kierunki **najlepiej
rozdzielające klasy** (to różne kryteria — PCA nie wie nic o etykietach). Dobra
separacja klas w 2D jest **wystarczającym** dowodem, że *jakaś* struktura istnieje, ale
zła separacja w 2D **nie jest** dowodem złej separacji w pełnych $512$ wymiarach —
dwa pierwsze kierunki mogą po prostu nie być tymi, które rozdzielają klasy.

**t-SNE / UMAP (krótko).** To inne, nieliniowe metody wizualizacji, które często dają
*wizualnie* czytelniejsze klastry niż PCA — ale kosztem zniekształcenia globalnej
geometrii.

❌ "t-SNE pokazuje prawdziwe odległości i rozmiary klastrów z oryginalnej przestrzeni."

✅ t-SNE celowo zniekształca odległości, żeby uwydatnić lokalną strukturę sąsiedztwa —
**rozmiar klastra na wykresie t-SNE nie odpowiada** rozmiarowi/gęstości w oryginalnej
przestrzeni, a odległość między dwoma odrębnymi klastrami **nie jest** miarodajną
miarą tego, jak bardzo są "różne". Nie rozwijamy tu porównania algorytmów — to temat
na osobne studium, nie trzon tego wykładu.

## Uczenie reprezentacji z relacji między przykładami — krótka zapowiedź

Jak embedding "wie", że dwa obrazy mają być blisko siebie? W naszym eksperymencie B
embedding pochodzi z sieci trenowanej do **klasyfikacji** na ImageNet — bliskość w tej
przestrzeni to efekt uboczny tego zadania, nie jego bezpośredni cel. Istnieje rodzina
metod, które **wprost** optymalizują geometrię: *metric learning* / *contrastive
learning* — zamiast "przewiduj klasę $y$", cel treningowy to "dla par/trójek
przykładów, ściągnij bliżej te, które powinny być podobne, odepchnij te, które nie
powinny" (np. ten sam obiekt z dwóch zdjęć vs różne obiekty). To temat, do którego
wrócimy przy okazji bardziej zaawansowanych architektur — tu tylko intuicja: cel
treningowy **bezpośrednio kształtuje** geometrię wyuczonej przestrzeni, nie tylko to,
czy model dobrze klasyfikuje.

## `nn.Embedding`: uczona tablica wektorów dla identyfikatorów dyskretnych

Wszystko powyżej dotyczyło embeddingów **obliczonych** z danych ciągłych (obraz →
sieć → wektor). Dla danych **dyskretnych** (słowo, token, identyfikator użytkownika)
nie ma naturalnego wejścia ciągłego — zamiast tego używa się **tablicy** wektorów,
jednego na każdy możliwy identyfikator, odczytywanej przez indeksowanie:

```python
import torch.nn as nn

V, d = 6, 4  # V = rozmiar słownika (liczba możliwych tokenów), d = wymiar embeddingu
embedding = nn.Embedding(V, d)  # tablica V x d, losowo zainicjalizowana, trenowalna
token_ids = torch.tensor([0, 3, 1])
vectors = embedding(token_ids)  # (3, 4) -- po prostu odczyt odpowiednich wierszy
```

`nn.Embedding(V, d)` to macierz $V\times d$ przechowywana jako zwykły, trenowalny
parametr (`embedding.weight`) — `forward` to **indeksowanie** (odczyt wierszy), nie
mnożenie macierzowe. Gradient płynący wstecz aktualizuje **tylko te wiersze**, które
wystąpiły w danym batchu (zweryfikowane w notebooku: wiersze nieużyte w ogóle nie
dostają gradientu, więc się nie zmieniają — identycznie jak zamrożone warstwy w module
7, tylko że tu "zamrożenie" wynika z nieobecności w danych, nie z `requires_grad`).

❌ "Embeddingi tokenów od razu mają sensowną strukturę semantyczną — podobne słowa są
blisko siebie, bo `nn.Embedding` tak działa."

✅ **Świeżo zainicjalizowany** `nn.Embedding` to po prostu losowe wektory — żadnej
semantycznej relacji między nimi nie ma, dopóki trening (na jakimś celu — przewidywanie
sąsiedniego słowa, tłumaczenie, cokolwiek) nie zacznie je modyfikować tak, by pomagały
minimalizować stratę. To dokładnie ten sam mechanizm co embedding z CNN wyżej:
struktura przestrzeni jest **efektem treningu na konkretnym celu**, nie właściwością
samego faktu "to jest embedding". Pełne omówienie, jak taki cel trenuje się dla
sekwencji (RNN, uwaga, Transformer) — moduły 9 i 10.

## Implikacje inżynierskie

- **Koszt ekstrakcji** to zwykle jeden forward pass przez backbone na obraz — policzony
  raz, może być **cache'owany** (moduł 7) i użyty wielokrotnie, o ile backbone i
  preprocessing się nie zmieniają.
- **Pamięć**: $N$ obrazów $\times$ $d$ wymiarów $\times$ 4 bajty (float32) — dla
  $N=10^6$, $d=512$: ok. $2$GB. Wymiarowość embeddingu ma więc bezpośredni wpływ na
  koszt przechowywania całej bazy.
- **Koszt wyszukiwania**: porównanie jednego zapytania z $N$ wektorami to $O(N\cdot d)$
  — liniowe w rozmiarze bazy. Dla dużych $N$ (miliony+) używa się **przybliżonego**
  wyszukiwania najbliższych sąsiadów (ANN — *approximate nearest neighbors*, np. FAISS,
  wektorowe bazy danych) zamiast dokładnego porównania ze wszystkim — kompromis
  dokładność/szybkość, nie temat tego wykładu, ale warto wiedzieć, że istnieje.
- **Spójność preprocessingu** jest krytyczna: embedding obliczony innym
  preprocessingiem (inna normalizacja, inny rozmiar wejścia) **nie jest porównywalny**
  z resztą bazy, nawet jeśli pochodzi z tego samego modelu.
- **Degradacja jakości w czasie**: jeśli rozkład nowych zapytań/dokumentów przesuwa się
  względem danych, na których model był trenowany (podobnie do *domain shift* z modułu
  7), jakość wyszukiwania może się pogarszać bez żadnej zmiany w kodzie.

## Co warto zapamiętać

- Embedding $\mathbf e=f_\theta(x)$ to reprezentacja **wyuczona**, różna od logitów $z$
  i od surowych danych wejściowych — wszystkie trzy to wektory, ale różnej użyteczności.
- Odległość euklidesowa i cosine similarity mierzą różne rzeczy i mogą dać różny
  ranking — cosine ignoruje długość wektora, euklidesowa nie.
- Normalizacja L2 zmienia geometrię (po normalizacji euclidean i cosine rankingi się
  pokrywają), ale nie gwarantuje automatycznie lepszych wyników.
- PCA pokazuje kierunki maksymalnej wariancji, nie maksymalnej separowalności klas —
  dobra/zła separacja w 2D nie dowodzi niczego wprost o pełnej przestrzeni.
- `nn.Embedding` to uczona tablica wektorów — struktura semantyczna jest efektem
  treningu, nie domyślną właściwością.

## Pytania dla studentów

1. Masz dwa wektory o takim samym kierunku, ale różnej długości. Jaka jest ich cosine
   similarity? Jaka odległość euklidesowa? Czy to zależy od długości?
2. Dlaczego do porównań podobieństwa między obrazami lepiej użyć wyjścia `avgpool` niż
   wyjścia `fc` modelu klasyfikacyjnego?
3. Masz embedding $\mathbf e$ z normą bliską zeru. Co się stanie, jeśli znormalizujesz
   go przez podzielenie przez normę bez zabezpieczenia $\varepsilon$? Jak to naprawić?
4. Dwa punkty są blisko siebie na wykresie PCA 2D. Jakie dodatkowe sprawdzenie
   wykonałbyś/wykonałabyś, zanim uznasz, że są "podobne" w pełnej przestrzeni cech?

## Bibliografia

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image
  Recognition*. CVPR. (backbone użyty do ekstrakcji embeddingów — ResNet18, moduł 7)
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. Journal of
  Machine Learning Research, 9, 2579-2605.
- Jolliffe, I. T., & Cadima, J. (2016). *Principal component analysis: a review and
  recent developments*. Philosophical Transactions of the Royal Society A, 374(2065),
  20150202.
- PyTorch documentation: *torch.nn.Embedding*,
  `https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html`.
- PyTorch documentation: *torch.linalg.eigh*,
  `https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html`.
