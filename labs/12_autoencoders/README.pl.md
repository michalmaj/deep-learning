Język: [English](README.md) | **Polski**

# Laboratorium 12. Detekcja anomalii za pomocą autoenkodera

Laboratorium trwa około 120 minut i korzysta z pojęć opisanych w
[`lecture.md`](../../lectures/12_autoencoders/lecture.md). Tym razem
rekonstrukcja służy do wykrywania anomalii, a nie do generowania obrazów.
Analiza obejmuje zarówno skuteczność metody, jak i przypadki, w których zawodzi.

Używany jest Fashion-MNIST na licencji MIT, zarejestrowany w
[`data/registry.yml`](../../data/registry.yml). Klasą normalną jest *Sneaker*
(7), a pozostałe dziewięć klas tworzy anomalie. Odpowiada to sytuacji, w której
normalność jest wąsko zdefiniowana, a anomalią jest wszystko spoza tej grupy.

Nie jest to klasyfikacja dziesięciu klas. Autoenkoder podczas treningu nie
otrzymuje etykiet ani przykładów anomalii. Uczy się wyłącznie rekonstrukcji
obrazów klasy normalnej.

## 0. Konfiguracja

Gotowa część obejmuje importy, wybór urządzenia, ziarno losowości i wczytanie
Fashion-MNIST przez `torchvision.datasets.FashionMNIST`.

## 1. Dane i podział

Należy przygotować rozłączne podzbiory:

1. **`ae_train`** zawiera tylko obrazy klasy Sneaker z oryginalnego zbioru
   treningowego. Wyłącznie ten podzbiór trafia do treningu autoenkodera.
2. **`val_normal`** zawiera inne obrazy klasy Sneaker z oryginalnego zbioru
   treningowego. Można użyć `torch.randperm` i podziału indeksów klasy 7.
3. **`val_anomaly`** zawiera próbkę pozostałych klas z oryginalnego zbioru
   treningowego, najlepiej z równą liczbą obrazów na klasę.
4. **`test_normal`** i **`test_anomaly`** pochodzą z oryginalnego zbioru
   testowego. Pierwszy zawiera klasę 7, a drugi próbkę pozostałych klas.

Podzbiory `test_*` służą wyłącznie do jednorazowej oceny końcowej. Nie mogą
wpływać na architekturę, hiperparametry ani próg detekcji. W notebooku trzeba
wyświetlić rozmiary wszystkich podzbiorów i programowo potwierdzić rozłączność
ich indeksów.

## 2. Autoenkoder

Należy zaimplementować klasę `AE(nn.Module)` z enkoderem MLP, bottleneckiem i
dekoderem do obrazu $28\times28$. Funkcja treningowa używa MSE i wyłącznie
`ae_train`. Główny model dla sekcji 3-7 ma $d_e=16$.

Trening trwa od ułamka sekundy do kilku sekund na CPU. Większość czasu jest
przeznaczona na przygotowanie danych i analizę wyników.

## 3. Błąd rekonstrukcji

Funkcja `per_image_error(model, X)` ma zwracać osobny błąd dla każdego obrazu:

```python
((x_hat - x) ** 2).flatten(1).mean(dim=1)
```

Błąd należy policzyć dla `val_normal` i `val_anomaly`. Oba rozkłady powinny
znaleźć się na wspólnym histogramie lub wykresie gęstości. Dla każdego trzeba
również podać średnią, odchylenie standardowe i medianę.

Częściowe nakładanie rozkładów jest oczekiwane. O użyteczności metody decyduje
separacja przy wybranym progu oraz charakter przykładów w obszarze nakładania.

## 4. Wybór progu

Próg powstaje **wyłącznie** z rozkładu `val_normal`. Można użyć średniej
powiększonej o $k$ odchyleń standardowych, pod warunkiem uzasadnienia wartości
$k$. `val_anomaly` nie może wpływać na konkretną wartość progu.

Procedura nie jest w pełni nienadzorowana. Etykiety służą do utworzenia zbioru
normalnego oraz ogólnych decyzji walidacyjnych. Sam próg opiera się jednak tylko
na przykładach normalnych, co odpowiada sytuacji bez dostępnych anomalii podczas
konfiguracji systemu.

## 5. Metryki testowe

Implementację metryk trzeba najpierw sprawdzić na znanym przykładzie:

```text
scores = [0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.6, 0.7, 0.8, 1.0]
labels = [0,   0,   0,   0,   0,   1,   1,   1,   1,   1]
threshold = 0.45
```

Najpierw należy ręcznie wyznaczyć TP, FP, TN, FN, precision, recall i F1, a
potem potwierdzić wynik kodem. Oczekiwane wartości to:

- TP = 5, FP = 1, TN = 4, FN = 0,
- precision $\approx0{,}833$,
- recall $=1{,}0$,
- F1 $\approx0{,}909$.

Repozytorium nie zawiera `scikit-learn`, dlatego ROC-AUC i PR-AUC trzeba
zaimplementować samodzielnie.

- **ROC-AUC** można policzyć jako
  $P(\text{score}_{\text{anomalia}}>\text{score}_{\text{normalny}})$ dla par
  anomalia-normalny. W małym teście dopuszczalne są dwie pętle.
- **PR-AUC**, tutaj average precision, wymaga sortowania wyników malejąco i
  uśrednienia precision w pozycjach zajętych przez prawdziwie dodatnie przykłady.

Na zbiorze testowym należy podać precision, recall, F1, macierz pomyłek $2\times2$,
ROC-AUC i PR-AUC. Metryki AUC oceniają ranking niezależnie od progu. Metryki
precision, recall i F1 opisują działanie przy progu wybranym w sekcji 4. Obie
grupy są potrzebne.

## 6. Baseline i analiza błędów

Baseline bez treningu używa odległości od średniego obrazu klasy normalnej:

```python
mean_normal_image = ae_train.mean(dim=0)
((x - mean_normal_image) ** 2).mean()
```

Dla baseline'u należy policzyć te same metryki testowe i porównać je z AE.
Słaby wynik autoenkodera względem prostego punktu odniesienia jest wartościową
informacją. Może oznaczać, że klasy różnią się głównie prostymi własnościami
obrazu, które nie wymagają uczonej reprezentacji.

Analiza błędów musi zawierać dwa konkretne przypadki:

1. anomalię z niskim błędem rekonstrukcji, czyli false negative, wraz z klasą i
   wizualnym komentarzem,
2. obraz klasy Sneaker z wysokim błędem, czyli false positive, również pokazany
   i opisany.

Autoenkoder może dobrze odtworzyć anomalię podobną do klasy normalnej, na
przykład inny typ obuwia. Jest to ograniczenie sygnału rekonstrukcji, a nie
automatycznie błąd implementacji.

## 7. Diagnostyka przecieku danych

Dostarczony kod przeszukuje progi i wybiera najwyższe F1 na etykietach ze
zbioru testowego. Działa technicznie, ale łamie metodologię oceny.

Należy:

1. uruchomić przeszukiwanie na `test_normal` i `test_anomaly`,
2. porównać znaleziony próg z progiem opartym na `val_normal`,
3. porównać F1 po przecieku z uczciwym wynikiem testowym,
4. wyjaśnić, dlaczego użycie testu do podjęcia decyzji jest błędne niezależnie
   od wielkości różnicy w tym przebiegu,
5. wskazać poprawkę zastosowaną w sekcji 4.

Niewielka różnica wyników nie unieważnia problemu. Ten sam przeciek na innym
zbiorze lub modelu może silnie zawyżyć rezultat. Zbiór testowy musi pozostać
niezależny od wszystkich decyzji modelowania.

## 8. Wnioski

Odpowiedzi powinny odwoływać się do liczb i przykładów z sekcji 3-7:

1. Jakie precision, recall, F1 i ROC-AUC osiągnął autoenkoder?
2. Jak wynik wypada na tle baseline'u i co mówi o źródle separacji klas?
3. Który konkretny false negative lub false positive najlepiej pokazuje
   ograniczenie metody?
4. Na czym polegał przeciek danych i jaka ogólna zasada z niego wynika?
5. Czego brakuje do praktycznego wdrożenia, w tym do określenia akceptowalnych
   kosztów FP i FN?

## Rozszerzenie opcjonalne

Można wybrać jedno z dwóch zadań:

- powtórzyć sekcje 2-5 dla $d_e\in\{4,32\}$ i porównać wpływ bottlenecku na
  separację klas,
- zbudować analogiczny VAE i użyć błędu rekonstrukcji lub $-$ELBO jako wyniku
  anomalii, bez zakładania z góry przewagi nad AE.

## Podział czasu

| Blok | Czas |
|---|---:|
| Konfiguracja | 5 min |
| Dane i podział | 15 min |
| Architektura i trening AE | 15 min |
| Błąd rekonstrukcji i rozkłady | 10 min |
| Próg walidacyjny | 10 min |
| Metryki | 15 min |
| Analiza błędów i baseline | 15 min |
| Diagnostyka przecieku | 20 min |
| Wnioski | 10 min |

Rdzeń zajmuje 115 minut, a pozostałe 5 minut stanowi bufor. Rozszerzenie nie
wchodzi do tego limitu.

## Kryteria oceny

- Rozłączność podzbiorów jest potwierdzona programowo.
- Autoenkoder jest trenowany wyłącznie na danych normalnych.
- Próg zależy wyłącznie od `val_normal`.
- Metryki przechodzą test na znanym przykładzie przed użyciem na danych.
- Porównanie z baseline'em zostało wykonane niezależnie od wyniku.
- Analiza pokazuje i omawia konkretne false negative oraz false positive.
- Diagnostyka poprawnie wskazuje przeciek i wyjaśnia jego konsekwencje.
- Wnioski opierają się na wynikach danego przebiegu.

Pełne rozwiązanie wzorcowe znajduje się lokalnie u prowadzącego w
`decisions/solutions/12_autoencoders/` i nie jest częścią repozytorium.
