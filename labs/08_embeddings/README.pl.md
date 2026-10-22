Język: [English](README.md) | **Polski**

# Laboratorium 08. System wyszukiwania podobnych obrazów

Laboratorium trwa około 2 godzin i rozwija materiał z
[`lectures/08_embeddings/lecture.md`](../../lectures/08_embeddings/lecture.md).
Obejmuje embeddingi, odległość euklidesową, cosine similarity, normalizację L2 i
k-NN. Zadaniem jest zbudowanie małego systemu wyszukiwania obrazów EuroSAT. Obraz
zapytania ma prowadzić do rankingu najbardziej podobnych elementów bazy.

## 0. Konfiguracja (gotowe)

Import bibliotek, wybór urządzenia i ustawienie ziarna.

## 1. Gallery i queries (częściowo gotowe)

Dane EuroSAT są dzielone na dwa rozłączne zbiory.

- **gallery** zawiera 40 obrazów z każdej klasy, czyli 400 łącznie,
- **queries** zawiera 10 innych obrazów z każdej klasy, czyli 100 łącznie.

**Zadanie.** Należy programowo potwierdzić, że `gallery_indices` i `query_indices`
nie mają wspólnych elementów. Jeśli obraz zapytania znajduje się również w gallery,
wyszukiwarka odnajdzie go z podobieństwem $1{,}0$ i sztucznie zawyży metryki.

## 2. Ekstrakcja embeddingów

Należy wczytać `resnet18` z wagami `ResNet18_Weights.IMAGENET1K_V1` i wyodrębnić
wyjście `avgpool` dla `gallery_imgs` oraz `query_imgs`. Logity z `fc` nie są używane.

Model powinien działać w trybie, który nie aktualizuje statystyk BatchNorm. Osobny
kontekst ma wyłączyć budowanie grafu gradientów. Poprawne użycie obu mechanizmów jest
częścią zadania. Należy również zmierzyć czas ekstrakcji.

### Ważność cache'u

Embeddingi 500 obrazów są obliczane raz i przechowywane w pamięci na potrzeby
pozostałych sekcji.

**Zadanie w Markdown, 2-3 zdania.** Dla bazy 1000 razy większej warto byłoby zapisać
embeddingi na dysku. Należy wskazać cztery konkretne zmiany, które unieważniłyby taki
cache i wymagały ponownej ekstrakcji.

## 3. Normalizacja L2

Funkcja `l2_normalize(X)` powinna dzielić każdy embedding przez jego normę i chronić
przed dzieleniem przez zero. Gotowa asercja sprawdza, czy normy wynikowych wektorów są
bliskie $1$.

## 4. Wyszukiwanie top-k

Należy zaimplementować `search(query_embedding, gallery_embeddings, k)`, która
zwraca indeksy $k$ obrazów o największym cosine similarity, uporządkowane od
najlepszego wyniku. Implementacja powinna korzystać z mnożenia macierzowego i
sortowania, bez gotowej funkcji wyszukiwania sąsiadów.

Mały przykład zawiera sześć wektorów 2D z dwóch klas i znany ranking. Funkcja musi
zwrócić dokładnie oczekiwaną kolejność.

## 5. Precision@k

Wynik jest uznawany za trafny, gdy jego klasa odpowiada klasie zapytania. Jest to
uproszczenie podobieństwa semantycznego, które należy uwzględnić we wnioskach.

Funkcja `precision_at_k(query_embeddings, query_labels, gallery_embeddings,
gallery_labels, k)` powinna zwracać średnie Precision@k dla wszystkich zapytań.
Najpierw trzeba zweryfikować ją na ręcznym przykładzie, a następnie obliczyć
Precision@1, Precision@5 i Precision@10 dla znormalizowanych embeddingów ResNet18.

## 6. Baseline oparty na pikselach

Te same metryki należy obliczyć dla surowych, spłaszczonych pikseli. Porównanie musi
korzystać z tych samych zbiorów gallery i queries oraz z wcześniej przygotowanych
funkcji `search` i `precision_at_k`.

## 7. Wpływ normalizacji i metryki

Precision@5 należy porównać dla trzech wariantów.

1. Cosine similarity na embeddingach znormalizowanych L2.
2. Iloczyn skalarny na embeddingach nieznormalizowanych.
3. Odległość euklidesowa na embeddingach nieznormalizowanych.

Wszystkie warianty muszą używać identycznych gallery i queries.

**Zadanie w Markdown, 3-4 zdania.** Analiza powinna wskazać najlepszy wariant,
wielkość różnicy oraz związek wyniku z rolą normy opisaną na wykładzie.

## 8. Wizualizacja i analiza błędów

Należy wybrać dokładnie dwa zapytania, jedno z wysokim i jedno z niskim Precision@5.
Każde zapytanie powinno zostać pokazane wraz z pięcioma najlepszymi wynikami.

**Zadanie w Markdown, 3-5 zdań.** Dla słabszego zapytania należy ocenić, czy błędne
wyniki mają podobną teksturę lub klasę terenu, czy wyglądają przypadkowo. Analiza ma
wymienić konkretne klasy widoczne w przykładzie.

## 9. Wnioski

Odpowiedzi powinny odwoływać się do wyników z sekcji 5-8.

1. O ile embeddingi ResNet18 przewyższają baseline pikselowy według Precision@k? Czy
   wynik jest zaskakujący dla domeny odległej od ImageNet?
2. Jak duży wpływ miały metryka i normalizacja? Czy system budowany od zera powinien
   normalizować embeddingi?
3. Czy analiza zawiera wynik o innej etykiecie, ale sensownym podobieństwie, albo
   wynik o tej samej etykiecie i wyraźnie innym wyglądzie? Co pokazuje to o granicach
   Precision@k opartego na klasach?
4. Jakie ograniczenie eksperymentu utrudnia uogólnienie wyniku na dowolny system
   wyszukiwania obrazów?

## Orientacyjny podział czasu

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, gallery/queries) | 10 min |
| Sekcja 2 (ekstrakcja + pytanie o cache) | 15 min |
| Sekcja 3 (normalizacja L2) | 10 min |
| Sekcja 4 (top-k + weryfikacja) | 20 min |
| Sekcja 5 (Precision@k + weryfikacja) | 20 min |
| Sekcja 6 (baseline pikselowy) | 10 min |
| Sekcja 7 (metryka i normalizacja) | 15 min |
| Sekcja 8 (wizualizacja i błędy) | 10 min |
| Sekcja 9 (wnioski) | 10 min |

Sekcje 0-9 tworzą obowiązkowy rdzeń. Ekstrakcja bazy trwa na CPU kilka sekund, więc
większość czasu można przeznaczyć na implementację rankingu, metryk i analizę.

**Rozszerzenia opcjonalne.** Można policzyć najczęstsze błędne pary klas, utworzyć
cache na dysku z jawnym tagiem ważności albo sprawdzić wpływ rozmiaru gallery.

## Kryteria oceny

- Gallery i queries są rozłączne, a asercja z sekcji 1 przechodzi.
- Ekstrakcja używa poprawnego trybu modelu i kontekstu bez gradientów.
- `l2_normalize`, `search` i `precision_at_k` są zaimplementowane samodzielnie i
  przechodzą test na ręcznym przykładzie.
- Precision@k ma jawny mianownik i jawną definicję trafności opartą na klasie.
- Wszystkie porównania korzystają z tych samych gallery i queries.
- Wnioski odwołują się do konkretnych wyników przebiegu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego w
`decisions/solutions/08_embeddings/` i nie jest częścią repozytorium.
