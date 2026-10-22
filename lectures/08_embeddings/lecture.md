Język: [English](lecture.en.md) | **Polski**

# Wykład 08. Embeddingi i geometria przestrzeni cech

## Cele wykładu

1. wyjaśnić, czym embedding różni się od surowych danych, ręcznie projektowanych cech i logitów,
2. obliczać i interpretować odległość euklidesową oraz podobieństwo cosinusowe,
3. opisać wpływ normalizacji L2 na geometrię reprezentacji,
4. sprawdzić empirycznie użyteczność sąsiedztwa w wyuczonej przestrzeni,
5. rozpoznać ograniczenia wizualizacji PCA, t-SNE i UMAP,
6. przedstawić `nn.Embedding` jako trenowalną tablicę wektorów dla identyfikatorów.

## Dlaczego to jest ważne

W module 7 backbone CNN służył jako ekstraktor cech. Jego wyjście trafiało do nowej głowicy, ale sama geometria tego wektora nie była analizowana. Teraz istotne staje się to, kiedy odległość między reprezentacjami odpowiada podobieństwu obiektów.

Użyteczna geometria umożliwia wyszukiwanie podobnych obrazów, rekomendacje i wyszukiwanie semantyczne. Bliskość liczbowa nie zawsze ma jednak oczekiwane znaczenie, dlatego wymaga sprawdzenia na danych.

## Intuicja

Reprezentacja zapisuje obiekt, na przykład obraz, słowo lub użytkownika, w postaci odpowiedniej do obliczeń. Surowe piksele także tworzą wektor, ale jego strukturę narzuca format danych. Embedding jest reprezentacją wyuczoną. Sieć dostosowuje jego wymiary do celu minimalizowanego podczas treningu.

❌ Każdy wektor liczbowy ma automatycznie użyteczną interpretację podobieństwa.

✅ Odległość zawsze ma znaczenie geometryczne, ale jej sens semantyczny zależy od sposobu utworzenia wektora. Surowe piksele i cechy CNN są ciągami liczb, lecz mogą tworzyć zupełnie inne sąsiedztwa. Eksperyment porównuje je bez przyjmowania przewagi embeddingu z góry.

## Model formalny

Embedding jest odwzorowaniem obiektu na wektor o ustalonej liczbie wymiarów.

$$\mathbf e = f_\theta(x), \qquad x \in \mathcal X,\quad \mathbf e \in \mathbb R^d$$

$\mathcal X$ oznacza przestrzeń wejściową, na przykład obrazy $3\times224\times224$, zdania lub identyfikatory produktów. $\mathbb R^d$ jest przestrzenią reprezentacji, a $\theta$ oznacza parametry odwzorowania. Wyjście `avgpool` w ResNet18 ma $d=512$.

Pogrubiony symbol $\mathbf e$ oznacza wektor embeddingu. Symbol $z$ pozostaje zarezerwowany dla logitów, a zwykłe $e$ może oznaczać liczbę Eulera lub indeks epoki. Kontekst i krój symbolu rozdzielają te znaczenia.

### Embedding z CNN

W ResNet18 embeddingiem jest wyjście globalnego poolingu przed warstwą `fc`.

| Obiekt | Źródło | Wymiar dla 10 klas | Zastosowanie |
|---|---|---:|---|
| Embedding $\mathbf e$ | wyjście `avgpool` | $512$ | podobieństwo i wyszukiwanie |
| Logity $z$ | wyjście `fc` przed softmax | $10$ | wejście funkcji straty |
| Prawdopodobieństwa $\hat y=\text{softmax}(z)$ | `softmax(fc(...))` | $10$ | interpretacja predykcji |

❌ Logity mogą zastąpić embedding, ponieważ są tylko późniejszym etapem tego samego modelu.

✅ Logity są powiązane z konkretnymi klasami zadania. Ich wymiar i znaczenie zmieniają się razem z głowicą. Embedding z `avgpool` jest bardziej ogólny i pozostaje niezależny od konkretnej warstwy klasyfikacyjnej. Dlatego w module 7 wymieniana była warstwa `fc`, a nie `avgpool`.

## Podobieństwo i odległość

### Odległość euklidesowa

$$d(\mathbf a,\mathbf b) = \|\mathbf a - \mathbf b\|_2 = \sqrt{\textstyle\sum_i (a_i-b_i)^2}$$

### Iloczyn skalarny i podobieństwo cosinusowe

$$\mathbf a \cdot \mathbf b = \textstyle\sum_i a_i b_i, \qquad \cos(\mathbf a,\mathbf b) = \frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|_2\,\|\mathbf b\|_2} \in [-1,1]$$

Cosine similarity mierzy kąt i ignoruje długość wektorów. Odległość euklidesowa uwzględnia zarówno kierunek, jak i długość. Metryki odpowiadają więc na inne pytania i mogą tworzyć różne rankingi sąsiadów.

### Normalizacja L2

Znormalizowany embedding ma postać $\hat{\mathbf e} = \mathbf e / \|\mathbf e\|_2$ i normę równą 1. Dla wektorów jednostkowych zachodzi

$$\|\hat{\mathbf a}-\hat{\mathbf b}\|_2^2 = 2 - 2\cos(\hat{\mathbf a},\hat{\mathbf b}).$$

Notebook potwierdza tę zależność z błędem rzędu $10^{-6}$. Po normalizacji ranking według odległości euklidesowej jest identyczny z rankingiem według cosine similarity, ponieważ obie miary są monotonicznymi przekształceniami tej samej wartości. Bez normalizacji rankingi mogą się różnić.

❌ Normalizacja L2 zawsze poprawia jakość wyszukiwania.

✅ Normalizacja usuwa wpływ długości wektora. Pomaga tylko wtedy, gdy norma nie niesie użytecznej informacji. Wpływ tej decyzji należy zmierzyć.

### Wektor zerowy

Dla $\mathbf e=\mathbf 0$ norma wynosi zero, więc $\mathbf e/\|\mathbf e\|_2$ jest niezdefiniowane. PyTorch zwraca w takim przypadku `nan`. Stabilna implementacja dodaje małe $\varepsilon$ do mianownika, na przykład `e / (e.norm() + eps)`.

### k najbliższych sąsiadów

Dla zapytania $\mathbf q$ i wektorów $\{\mathbf e_1,\ldots,\mathbf e_N\}$ algorytm k-NN wybiera $k$ najmniejszych odległości lub największych podobieństw. Symbol $k$ oznacza tutaj liczbę sąsiadów, a nie rozmiar filtra konwolucyjnego z modułu 6.

## Przykład różnych rankingów

Dane są zapytanie $\mathbf q=(1,0)$ oraz kandydaci $\mathbf a=(10,0)$ i $\mathbf b=(0{,}9,0{,}9)$.

$$d(\mathbf q,\mathbf a) = 9{,}0 \qquad d(\mathbf q,\mathbf b) \approx 0{,}906$$
$$\cos(\mathbf q,\mathbf a) = 1{,}0 \qquad \cos(\mathbf q,\mathbf b) \approx 0{,}707$$

Odległość euklidesowa wskazuje $\mathbf b$, natomiast cosine similarity wskazuje $\mathbf a$. Wektor $\mathbf a$ ma ten sam kierunek co zapytanie, ale jego norma jest dziesięć razy większa. Skala dominuje więc odległość euklidesową i nie wpływa na cosine similarity.

## Redukcja wymiarowości i PCA

Embeddingi ResNet18 mają 512 wymiarów, dlatego nie można ich bezpośrednio narysować. PCA znajduje wzajemnie ortogonalne kierunki $\mathbf u_1,\mathbf u_2,\ldots$, wzdłuż których dane mają największą wariancję. Pierwszy kierunek opisuje największy rozrzut, a kolejne największy rozrzut pozostały po uwzględnieniu wcześniejszych kierunków.

Projekcja na dwa pierwsze kierunki ma postać

$$\mathbf e' = (\mathbf u_1^\top \mathbf e,\ \mathbf u_2^\top \mathbf e) \in \mathbb R^2.$$

Notebook oblicza macierz kowariancji i jej rozkład własny za pomocą `torch.linalg.eigh`. Dwa wektory własne o największych wartościach własnych wyznaczają wykres. Kierunki o małej wariancji zostają odrzucone wraz z zakodowaną w nich informacją.

❌ PCA w 2D pokazuje separowalność klas w pełnej przestrzeni.

✅ PCA maksymalizuje wariancję i nie korzysta z etykiet. Dobra separacja w 2D potwierdza istnienie pewnej struktury. Słaba separacja nie dowodzi jednak, że klasy są nierozdzielne w 512 wymiarach.

t-SNE i UMAP są nieliniowymi metodami wizualizacji, które często tworzą czytelniejsze klastry, ale zniekształcają globalną geometrię.

❌ Rozmiary klastrów i odległości na wykresie t-SNE odpowiadają oryginalnej przestrzeni.

✅ t-SNE zachowuje przede wszystkim lokalne sąsiedztwo. Wielkość klastra i odległość między odrębnymi klastrami nie są wiarygodnymi miarami ich gęstości ani różnicy w przestrzeni wejściowej.

## Reprezentacje uczone z relacji

Embedding z eksperymentu pochodzi z modelu klasyfikującego ImageNet. Użyteczne sąsiedztwo jest efektem ubocznym tego celu. *Metric learning* i *contrastive learning* optymalizują geometrię bezpośrednio. Podobne pary lub trójki są przyciągane, a różne odpychane. Cel treningowy określa więc nie tylko jakość klasyfikacji, lecz także układ wyuczonej przestrzeni.

## `nn.Embedding` dla identyfikatorów dyskretnych

Obraz jest przekształcany w embedding przez sieć. Dla słowa, tokenu lub identyfikatora użytkownika stosuje się tablicę zawierającą osobny wektor dla każdego identyfikatora.

```python
import torch.nn as nn

V, d = 6, 4
embedding = nn.Embedding(V, d)
token_ids = torch.tensor([0, 3, 1])
vectors = embedding(token_ids)  # kształt (3, 4)
```

`nn.Embedding(V, d)` przechowuje trenowalną macierz $V\times d$ w `embedding.weight`. Forward polega na odczycie wybranych wierszy, a nie na mnożeniu macierzowym. Gradient aktualizuje tylko wiersze użyte w danym batchu. Notebook potwierdza, że pozostałe wiersze się nie zmieniają.

❌ Świeżo utworzone embeddingi tokenów od razu mają strukturę semantyczną.

✅ Początkowe wektory są losowe. Relacje między nimi powstają dopiero podczas treningu na konkretnym celu, na przykład przewidywaniu sąsiedniego słowa lub tłumaczeniu. Moduły 9 i 10 rozwijają ten temat dla sekwencji.

## Implikacje inżynierskie

- **Ekstrakcja.** Jeden forward pass przez backbone wystarcza do obliczenia embeddingu. Wynik może być cache'owany, jeśli model i preprocessing pozostają bez zmian.
- **Pamięć.** Baza $N$ embeddingów o wymiarze $d$ w float32 zajmuje około $N\cdot d\cdot4$ bajty. Dla $N=10^6$ i $d=512$ jest to około 2 GB.
- **Wyszukiwanie.** Dokładne porównanie zapytania z całą bazą kosztuje $O(N\cdot d)$. Duże zbiory korzystają z przybliżonego wyszukiwania ANN, na przykład FAISS lub baz wektorowych.
- **Preprocessing.** Wektory obliczone z inną normalizacją lub rozmiarem wejścia nie są bezpośrednio porównywalne.
- **Zmiana rozkładu.** Domain shift nowych zapytań może pogorszyć jakość wyszukiwania mimo braku zmian w kodzie.

## Co warto zapamiętać

- Embedding $\mathbf e=f_\theta(x)$ jest wyuczoną reprezentacją inną od logitów i surowych danych.
- Odległość euklidesowa uwzględnia normę, a cosine similarity mierzy kierunek.
- Normalizacja L2 zrównuje rankingi obu metryk, ale nie gwarantuje lepszych wyników.
- PCA pokazuje kierunki największej wariancji, a nie najlepszej separacji klas.
- `nn.Embedding` jest trenowalną tablicą, której struktura powstaje podczas uczenia.

## Pytania dla studentów

1. Jakie wartości przyjmują cosine similarity i odległość euklidesowa dla wektorów o tym samym kierunku i różnych długościach?
2. Dlaczego do porównywania obrazów lepiej nadaje się wyjście `avgpool` niż `fc`?
3. Co dzieje się podczas normalizacji embeddingu o normie bliskiej zeru bez zabezpieczenia $\varepsilon$?
4. Jak można zweryfikować podobieństwo dwóch punktów widocznych blisko siebie na wykresie PCA 2D w pełnej przestrzeni cech?

## Bibliografia

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. Journal of Machine Learning Research, 9, 2579-2605.
- Jolliffe, I. T., & Cadima, J. (2016). *Principal component analysis: a review and recent developments*. Philosophical Transactions of the Royal Society A, 374(2065), 20150202.
- PyTorch documentation. *torch.nn.Embedding*. `https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html`.
- PyTorch documentation. *torch.linalg.eigh*. `https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html`.
