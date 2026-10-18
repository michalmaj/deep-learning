# Laboratorium 13: samodzielna implementacja i analiza niewielkiego modelu dyfuzyjnego (DDPM)

Czas: ok. 120 minut. Bazuje na
[`lectures/13_diffusion/lecture.md`](../../lectures/13_diffusion/lecture.md)
(forward/reverse diffusion, DDPM), ale na **innym problemie** niż wykład
(tam: mieszanina 8 Gaussów na okręgu). Tu: zbiór **dwóch półksiężyców**
(*two moons*) — geometria wklęsła, z "dziurą" między ramionami, którą
proste metody (np. jeden dopasowany rozkład Gaussa) nie potrafią odtworzyć.
To **nie jest** kopia eksperymentu wykładowego ze zmienionym ziarnem —
inny kształt danych wymaga tych samych wzorów, ale innej interpretacji
wyników.

**Cel**: samodzielnie zaimplementować matematyczny rdzeń DDPM (harmonogram,
zamkniętą postać forward diffusion, funkcję celu, odwrotne próbkowanie),
zweryfikować go, i krytycznie ocenić jakość wygenerowanych próbek —
**nie** tylko sprawdzić, czy wykres "wygląda podobnie".

## 0. Konfiguracja i dane (gotowe)

Import bibliotek, ziarno losowości, funkcja generująca zbiór two-moons
(czysty NumPy/PyTorch, bez `sklearn`).

## 1. Wizualizacja danych — Twoje zadanie

Wygeneruj `X_train` (two-moons, $N=2000$) i narysuj wykres rozrzutu. Opisz
(komentarz albo markdown): czym ten kształt różni się od mieszaniny Gaussów
z wykładu — w szczególności, dlaczego pojedynczy rozkład Gaussa (jedna
średnia, jedna macierz kowariancji) nie mógłby go dobrze opisać.

## 2. Harmonogram $\beta_t$ i obliczanie $\bar\alpha_t$ — Twoje zadanie

Zaimplementuj harmonogram liniowy: `betas = torch.linspace(beta_start,
beta_end, T)`, `alphas = 1 - betas`, `alphas_bar = torch.cumprod(alphas,
dim=0)`. Użyj $T=500$ (patrz wykład: wystarczające, by $\bar\alpha_T$ było
bliskie 0 dla tego zakresu $\beta$, bez niepotrzebnie wysokiego kosztu
generowania).

**Zweryfikuj**: wypisz $\bar\alpha_1$, $\bar\alpha_{T/2}$, $\bar\alpha_T$ —
czy $\bar\alpha_T$ jest wystarczająco bliskie 0 (porównaj z wykładową
dyskusją o roli końcowej wartości harmonogramu)?

## 3. Zamknięta postać forward diffusion — Twoje zadanie

Zaimplementuj `q_sample(x0, t, eps, alphas_bar)`:
$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$.

**Zweryfikuj** (wymagane, nie opcjonalne): dla ustalonego $x_0$ i $t$,
porównaj **statystyki** (średnia/wariancja na wielu próbach $\varepsilon$)
próbkowania zamkniętą postacią z symulacją krok-po-kroku (`for` po krokach
$1,\ldots,t$, z $q(x_s\mid x_{s-1})$). **Nie** oczekuj identyczności
pojedynczych próbek — to dwa różne, niezależne losowania; porównujesz
**rozkłady**, nie konkretne liczby.

## 4. Sieć przewidująca szum i warunkowanie czasem — Twoje zadanie

Zaimplementuj klasę `NoisePredictor(nn.Module)`:
- `forward(x, t)` zwraca przewidywany szum (2D wektor na przykład).
- **Musi** wykorzystywać $t$ — nie tylko przyjmować je jako argument.
  Użyj podanej funkcji `sinusoidal_embedding(t, dim)` (gotowa, poniżej) i
  **skonkatenuj** jej wynik z $x$ przed podaniem do warstw liniowych.

```python
def sinusoidal_embedding(t, dim=16):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)
```

**Zadanie (komentarz/markdown)**: wyjaśnij własnymi słowami, dlaczego $x_t$
**samo** nie wystarcza modelowi do poprawnego przewidzenia szumu — czego mu
brakuje bez znajomości $t$?

## 5. Funkcja celu i pętla treningowa — Twoje zadanie

Zaimplementuj jeden krok treningowy wewnątrz podanej pętli epok (szkielet
pętli — `for epoch in range(epochs): ...` z optymalizatorem — jest gotowy,
**zawartość** kroku jest Twoim zadaniem):

1. Wylosuj batch $x_0$ z `X_train`.
2. Wylosuj $t\sim\text{Uniform}\{0,\ldots,T-1\}$ — **inny $t$ dla każdego
   przykładu w batchu**.
3. Wylosuj $\varepsilon\sim N(0,I)$.
4. Skonstruuj $x_t$ (`q_sample` z sekcji 3).
5. Policz $\varepsilon_\theta(x_t,t)$ i stratę MSE względem $\varepsilon$.
6. Backward + krok optymalizatora.

Zbieraj i wypisz/zwizualizuj stratę per epoka (monitorowanie — wykres
gotowy do użycia w komórce wizualizacji).

## 6. Odwrotne próbkowanie — Twoje zadanie

Zaimplementuj jeden krok odwrotny `ddpm_reverse_step(model, x, t_idx,
alphas, betas, alphas_bar)`:

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\right)$$

Dla `t_idx > 0`: $x_{t-1}=\mu_\theta+\sqrt{\beta_t}z$, $z\sim N(0,I)$. Dla
`t_idx == 0` (ostatni krok): $x_0=\mu_\theta$ — **bez** dodanego szumu
(patrz wykład: dlaczego ostatni krok jest inny).

Następnie: funkcja `ddpm_sample(model, n_samples, T, ...)` zaczynająca od
$x_T\sim N(0,I)$ i iterująca krok po kroku do $t=0$.

**Zweryfikuj indeksowanie**: wypisz, ile razy pętla faktycznie wykonuje
krok z dodanym szumem, a ile bez — powinno być $T-1$ razy z szumem, $1$ raz
bez.

## 7. Generowanie nowych próbek i porównanie rozkładów — Twoje zadanie

Wygeneruj $1000$ nowych próbek. Narysuj: (a) dane treningowe, (b)
wygenerowane próbki, na tej samej skali osi. **Jawnie podpisz**, które
502 punkty to rekonstrukcje (nie ma ich tu — nie dajemy modelowi żadnego
realnego $x_0$ do odtworzenia), a które to **nowe** próbki od czystego
szumu.

Policz **przynajmniej jedną** metrykę kwantytatywną porównującą rozkłady
(zobacz sekcję "Ocena generowania" niżej) na **niezależnym** zbiorze
testowym (nowa próbka two-moons, nie `X_train`).

## 8. Sprawdzenie jednej zmiany konfiguracji — Twoje zadanie

Wybierz **jedną** zmianę (nie więcej — brief wymaga izolowania efektu
jednego czynnika):
- inna liczba kroków próbkowania (użyj **poprawnego** respacing, patrz
  wykład Eksperyment D.2 — nie przeskakuj kroków formułą jednostkowego
  kroku), **albo**
- inny harmonogram (np. cosinusowy, patrz wykład), **albo**
- inna liczba epok treningu (niedotrenowany vs dotrenowany model).

Zmierz wpływ na jakość generowania (ta sama metryka, co w sekcji 7).

## 9. Baseline — Twoje zadanie

Dopasuj **jeden** rozkład Gaussa do `X_train`
(`torch.distributions.MultivariateNormal` z empirycznym `mean`/`cov`).
Wygeneruj z niego $1000$ próbek. Policz te same metryki (sekcja 7) na tym
samym niezależnym zbiorze testowym.

**Nie zakładaj z góry**, że model dyfuzyjny wygra na **każdej** metryce —
zweryfikuj. Jeśli jedna metryka wypada niejednoznacznie, sprawdź **drugą**,
bardziej czułą na kształt rozkładu (np. średnia odległość do najbliższego
sąsiada w niezależnym zbiorze referencyjnym) — wyjaśnij, czemu te dwie
metryki mogą dawać różne wnioski.

## 10. Zadanie diagnostyczne

Poniższy, **oddzielnie dostarczony** fragment kodu (nie Twój z sekcji 4-6)
implementuje `NoisePredictorColleague`, trenuje go i generuje próbki — nie
rzuca wyjątku, generuje **coś**, ale ma błąd: sieć przyjmuje `t` jako
argument, ale **nigdy go nie używa** w `forward`.

**Zadanie:**
1. Wytrenuj `NoisePredictorColleague` tak samo, jak swój model (sekcja 5).
2. Wygeneruj próbki tą samą procedurą odwrotną (sekcja 6).
3. Porównaj metryki (sekcja 7) z Twoim poprawnym modelem — **nie** oceniaj
   tylko "na oko", policz liczby.
4. Zdiagnozuj: dlaczego model bez informacji o $t$ radzi sobie **wyraźnie
   gorzej**, mimo że wciąż coś generuje (nie crashuje)? Odwołaj się do tego,
   co napisałeś/aś w zadaniu z sekcji 4.
5. Napraw (dodaj warunkowanie czasem z powrotem) i zweryfikuj poprawę
   kwantytatywnie.

❌ "Jeśli kod nie rzuca wyjątku i coś generuje, implementacja jest
prawdopodobnie poprawna."

✅ Brak błędu wykonania **nie** jest dowodem poprawności modelu
generatywnego — trzeba **zmierzyć** jakość, nie tylko sprawdzić, że kod
"działa" w sensie "nie crashuje".

## 11. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami/przykładami z sekcji 2-10:

1. Czy $\bar\alpha_T$ dla Twojego harmonogramu było wystarczająco bliskie
   $0$? Co by się stało, gdyby nie było?
2. Jak dobrze (konkretne metryki) Twój model dyfuzyjny odtwarza rozkład
   two-moons na niezależnym zbiorze testowym?
3. Jak wypada w porównaniu z baseline'em (sekcja 9)? Czy obie metryki dały
   tę samą odpowiedź "który model lepszy"?
4. Jaki efekt miała zmiana konfiguracji z sekcji 8?
5. Jaki błąd wprowadził brak warunkowania czasem (sekcja 10), i jaka jest
   ogólna zasada, którą z tego wyciągasz?
6. Czy są obszary/regiony przestrzeni danych, których Twój model **nie**
   generuje dobrze (np. sama "dziura" między półksiężycami, albo końce
   ramion)? Jak to sprawdziłeś/aś?

## Ocena generowania — wymagania

Nie oceniaj modelu jedynie na podstawie tego, czy chmura punktów "wygląda
podobnie". Wymagane:

- **Poprawnie zdefiniowany rozkład referencyjny**: `X_train` do treningu,
  **niezależny** nowy zbiór two-moons do oceny (nie te same punkty).
- **Porównanie z baseline'em** (sekcja 9) — nie tylko "model dyfuzyjny sam
  w sobie wygląda nieźle".
- **Przynajmniej jedna metryka kwantytatywna** — zalecane: MMD (Maximum
  Mean Discrepancy, kernel RBF) — **zweryfikuj ją na znanym przypadku**
  (np. porównanie dwóch próbek z tego samego rozkładu powinno dać wynik
  bliski $0$) **przed** użyciem na wygenerowanych danych.
- **Analiza niewygenerowanych/słabo pokrytych obszarów** — sprawdź, czy są
  regiony danych treningowych, które generator rzadko/nigdy nie trafia.
- **Ograniczenia wynikające z małej liczby próbek** — nazwij je explicit
  (np. "przy 1000 próbkach, różnica X vs Y może być częściowo szumem
  próbkowania, nie realną różnicą modeli").

❌ "Większy rozrzut wygenerowanych punktów oznacza, że model lepiej odwzorował
rozkład."

✅ Rozrzut (np. wariancja) jest **jedną** własnością rozkładu, nie dowodem
poprawnego odwzorowania **kształtu** — model mógłby mieć podobną wariancję
co dane, a wciąż kompletnie nie odtwarzać ich geometrii (np. wypełniać
"dziurę" między półksiężycami, której w rzeczywistych danych nie ma).

## Podział czasu (realistyczny, zweryfikowany względem rzeczywistego kosztu CPU)

| Blok | Czas |
|---|---|
| Sekcja 0 (konfiguracja, dane) | 5 min |
| Sekcja 1 (wizualizacja danych) | 5 min |
| Sekcja 2 (harmonogram, ᾱ_t) | 8 min |
| Sekcja 3 (zamknięta postać + weryfikacja) | 10 min |
| Sekcja 4 (sieć + warunkowanie czasem) | 8 min |
| Sekcja 5 (funkcja celu + trening) | 13 min |
| Sekcja 6 (odwrotne próbkowanie) | 13 min |
| Sekcja 7 (generowanie + porównanie) | 8 min |
| Sekcja 8 (zmiana konfiguracji) | 6 min |
| Sekcja 9 (baseline) | 5 min |
| Sekcja 10 (zadanie diagnostyczne) | 10 min |
| Sekcja 11 (wnioski) | 7 min |

Suma: **98 min**, z **22-minutowym** buforem (razem 120 min) — więcej niż
wymagane minimum 15 minut. Trening (300 epok na 2000 próbkach, mały MLP)
trwa **ułamek sekundy do ~1,5s** na CPU; generowanie $1000$ próbek, $T=500$
kroków, trwa **ułamek sekundy**. Budżet czasowy idzie wyłącznie w
implementację, debugging i analizę — nie w czekanie na model.

## Kryteria oceny

- Harmonogram i $\bar\alpha_t$ poprawne, $\bar\alpha_T$ sprawdzone jako
  wystarczająco bliskie $0$.
- `q_sample` zweryfikowane statystycznie względem symulacji krok-po-kroku.
- Sieć wykorzystuje $t$ (nie tylko przyjmuje je jako argument) — zadanie z
  sekcji 4 odpowiedziane konkretnie, nie ogólnikiem.
- Odwrotne próbkowanie poprawnie obsługuje ostatni krok (brak dodanego
  szumu), zweryfikowane programowo.
- Generowanie porównane z **niezależnym** zbiorem testowym i z baseline'em,
  **przynajmniej jedną** zweryfikowaną metryką kwantytatywną.
- Zadanie diagnostyczne poprawnie identyfikuje brak warunkowania czasem
  jako przyczynę, z konkretnymi liczbami przed/po poprawce.
- Wnioski odwołują się do konkretnych liczb/przykładów z Twojego przebiegu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/13_diffusion/`) i nie jest częścią tego repozytorium.
