# Wykład 11 — Od modelu Transformer do działającego systemu generowania tekstu

## Cele wykładu

1. Rozumieć autoregresyjne modelowanie języka jako faktoryzację rozkładu całej
   sekwencji na iloczyn rozkładów warunkowych "następny token".
2. Umieć ręcznie policzyć, jak logity zamieniają się w rozkład prawdopodobieństwa
   (softmax), i jak temperature/top-k/top-p modyfikują ten rozkład **przed**
   wyborem tokenu.
3. Rozumieć tokenizację subword (BPE) jako kompromis między słownikiem znaków i
   słownikiem słów — i wiedzieć, że liczba tokenów **nie** jest proporcjonalna
   do liczby słów, zwłaszcza między językami.
4. Znać różnicę między prefill i decode, rolę KV cache, i umieć zmierzyć
   rzeczywisty koszt generowania (nie tylko go zadeklarować).
5. Umieć poprawnie użyć `tokenizer.apply_chat_template()` i `model.generate()`
   przez Hugging Face Transformers, rozumiejąc, co dokładnie robią.
6. Krytycznie oceniać wygenerowany tekst: odróżniać poprawność wykonania
   instrukcji od poprawności faktograficznej, i rozumieć ograniczenia małych
   modeli językowych.

## Dlaczego to jest ważne

Moduły 8-10 zbudowały wszystkie elementy: embeddingi (moduł 8), modelowanie
sekwencji (moduł 9), self-attention i blok Transformer (moduł 10). Ten wykład
**nie dodaje nowej matematyki Transformera** — pyta o coś innego: mając już
wytrenowany model, jak zamienia się jego surowe wyjście (logity) w tekst, który
faktycznie widzimy na ekranie? To pytanie inżynierskie, nie architektoniczne —
i ma własne, nietrywialne decyzje (strategia dekodowania, formatowanie
promptu, budżet pamięci, koszt czasowy), które wpływają na wynik dokładnie
tak samo, jak architektura modelu.

## Intuicja

Model językowy, w każdym kroku, nie "wie" z góry, jaką całą odpowiedź da — on
oblicza **jedną** liczbę na token w słowniku: "jak prawdopodobne jest, że *ten*
token będzie następny, biorąc pod uwagę wszystko, co było dotychczas?".
Generowanie tekstu to **powtarzanie** tego kroku: policz rozkład, wybierz
token (jakoś), dołącz go do kontekstu, policz rozkład na nowo, powtórz.

❌ "LLM generuje całą odpowiedź na raz, tak jak klasyfikator generuje jedną
etykietę."

✅ Typowy autoregresyjny model generuje **jeden token na raz**, każdy kolejny
token zależny od wszystkich poprzednich (włącznie z tymi, które sam model
wygenerował chwilę wcześniej). Zobaczymy to dosłownie w Eksperymencie B —
ręczna pętla generowania krok po kroku, porównana z `model.generate()`.

## Formalny model: autoregresyjna faktoryzacja

Niech $x_1,\ldots,x_T$ będzie sekwencją tokenów (identyfikatorów ze słownika
o rozmiarze $V$ — ten sam $V$, co w module 8, `nn.Embedding(V, d)`; $T$ =
długość sekwencji, jak w modułach 9-10). Reguła łańcuchowa prawdopodobieństwa
(dokładna, bez żadnego założenia o modelu) daje:

$$P(x_1,\ldots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_1,\ldots,x_{t-1})$$

(dla $t=1$: $P(x_1 \mid x_1,\ldots,x_0) := P(x_1)$, czyli rozkład bez żadnego
kontekstu — pusty warunek). To **nie jest założenie modelu** — to tożsamość
matematyczna prawdziwa dla **każdego** rozkładu łącznego. Założenie modelu
jest gdzie indziej: że $P(x_t \mid x_1,\ldots,x_{t-1})$ można **dobrze
przybliżyć** siecią neuronową $\theta$ — stąd zapis $P_\theta(x_t \mid
x_{<t})$ ($x_{<t} := x_1,\ldots,x_{t-1}$, standardowy skrót, używany dalej).

**Uwaga o notacji.** $x_t$ oznacza tutaj **identyfikator tokenu** (liczba
całkowita $\in\{1,\ldots,V\}$), nie wektor — inaczej niż $x_t\in\mathbb R^d$ w
module 9 (tam: embedding/cecha ciągła). To nie jest nowe znaczenie zastępujące
stare — to pierwszy krok pipeline'u: $x_t$ (id) → $\mathbf e_t$ (embedding,
pogrubione, moduł 8) → … → $z_t$ (logity, niżej). $V$ = rozmiar słownika,
rozszerzenie notacji z modułu 8 (tam: parametr konstruktora `nn.Embedding`, tu:
formalny wymiar logitów i rozkładu wyjściowego).

## Jak model przekształca kontekst w rozkład: logity i softmax

Dla kontekstu $x_{<t}$, model (encoder/decoder-only Transformer z modułu 10,
**nie wyprowadzamy go tu ponownie**) produkuje wektor **logitów** dla
pozycji $t$:

$$z_t = f_\theta(x_{<t}) \in \mathbb R^V$$

($z$ = logity, moduł 2 i dalej — tu rozszerzone indeksem czasowym $t$, ten sam
wzorzec co $h_t$, $c_t$, $\delta_t$ w module 9). $z_t$ ma **jeden wpis na
token w słowniku** — nieinterpretowalny bezpośrednio jako prawdopodobieństwo
(może być ujemny, nie sumuje się do 1). Softmax zamienia go w rozkład:

$$\hat y_t = \text{softmax}(z_t), \qquad \hat y_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^V e^{z_{t,j}}}$$

(rozszerzenie $\hat y=\text{softmax}(z)$ z modułu 8, tam bez indeksu czasowego
— klasyfikacja jednorazowa; tu $\hat y_t$ to rozkład nad **całym słownikiem**
dla **tej konkretnej pozycji generowania**, nie nad zbiorem klas zadania).
$\hat y_{t,i} = P_\theta(x_t = i \mid x_{<t})$ — składowa $i$ wektora $\hat
y_t$ to prawdopodobieństwo, że następny token to konkretnie token $i$.

❌ "Najwyższe prawdopodobieństwo jednego tokenu gwarantuje najlepszą całą
odpowiedź."

✅ $\hat y_t$ mówi tylko o **jednym** kroku. Wybranie najbardziej
prawdopodobnego tokenu na każdym kroku (greedy decoding, niżej) **nie**
gwarantuje najbardziej prawdopodobnej **całej** sekwencji — to dwa różne
zadania optymalizacyjne. Zweryfikowany kontrprzykład liczbowy w sekcji
"Greedy decoding" niżej.

## Temperature: transformacja rozkładu, nie wybór tokenu

**Temperature** skaluje logity **przed** softmaxem:

$$\hat y_t^{(\tau)} = \text{softmax}\!\left(\frac{z_t}{\tau}\right), \qquad \tau > 0$$

**Uwaga o notacji.** Temperaturę oznaczamy $\tau$, **nie** $T$ — $T$ w tym
kursie od modułu 9 oznacza **długość sekwencji**, a oba pojęcia pojawiają się
w tym samym akapicie (temperatura przy generowaniu sekwencji długości $T$) —
kolizja byłaby realna, nie tylko techniczna, więc tu naprawdę wprowadzamy
nowy symbol, zamiast dodawać notkę disambiguacyjną do istniejącego. $\tau$ jest
nieużywane wcześniej w kursie.

- $\tau < 1$: dzielenie przez liczbę $<1$ **zwiększa** różnice między
  logitami przed softmaxem → rozkład **ostrzejszy** (bardziej skoncentrowany
  na już-najbardziej-prawdopodobnych tokenach).
- $\tau > 1$: logity się "spłaszczają" → rozkład **bardziej płaski**
  (bliższy jednorodnemu).
- $\tau \to 0^+$: $\hat y_t^{(\tau)} \to$ rozkład skoncentrowany w 100% na
  $\arg\max_i z_{t,i}$ — formalnie równoważne greedy decoding w granicy.

❌ "Temperature zwiększa kreatywność modelu."

✅ To nieprecyzyjne antropomorfizowanie. Temperature jest **czysto
matematyczną transformacją rozkładu prawdopodobieństwa** — nie zmienia tego,
*czego* model "wie", tylko to, jak **skoncentrowany** albo **płaski** jest
rozkład, z którego **później** coś wybieramy. Sama temperature **nie wybiera
tokenu** — potrzebuje do tego procedury wyboru (sampling, niżej). Co więcej:
**jeśli `do_sample=False` (greedy), temperature nie ma żadnego wpływu na
wynik** — greedy bierze $\arg\max$ niezależnie od tego, jak rozkład został
przeskalowany (monotoniczna transformacja logitów nie zmienia, który z nich
jest największy). Przypisywanie temperature efektu w trybie, który jej nie
używa, jest błędem konfiguracji, nie subtelnością.

## Procedury wyboru tokenu: deterministyczne vs losowe

**Greedy decoding**: $x_t = \arg\max_i \hat y_{t,i}$ — deterministyczne,
zawsze ten sam wynik dla tego samego kontekstu i modelu.

❌ "Greedy decoding wybiera globalnie najbardziej prawdopodobną całą
sekwencję."

✅ Greedy optymalizuje **lokalnie, krok po kroku**, nie globalnie. **Zweryfikowany
kontrprzykład** (2 kroki, słownik $\{A,B\}\to\{X,Y\}$): krok 1,
$P(A)=0{,}5498 > P(B)=0{,}4502$ — greedy wybiera $A$. Krok 2 pod warunkiem
$A$: $P(X|A)=P(Y|A)=0{,}5$. Krok 2 pod warunkiem $B$: $P(X|B)=0{,}99$. Greedy
daje sekwencję $(A,X)$ z $P=0{,}2749$. **Globalnie najlepsza** sekwencja to
$(B,X)$ z $P=0{,}4457$ — większa, ale greedy jej nie znajduje, bo nigdy nie
"patrzy w przyszłość" przy wyborze kroku 1. Zweryfikowane enumeracją
wszystkich $4$ sekwencji w notebooku, nie zadeklarowane.

**Sampling**: losujemy $x_t \sim \hat y_t^{(\tau)}$ — niedeterministyczne (bez
ustalonego ziarna, różny wynik na różnych przebiegach; **z** ustalonym
ziarnem PyTorcha, powtarzalne — zweryfikowane w notebooku).

### Top-k

Ograniczamy losowanie do $k$ tokenów o najwyższym $z_t$, zerujemy
prawdopodobieństwo pozostałych, **renormalizujemy** tak, by zachowane
prawdopodobieństwa sumowały się do $1$.

**Uwaga o notacji.** $k$ tutaj to liczba zachowanych tokenów — nazwa
techniki ("top-k") jest ustandaryzowana w literaturze; symbol $k$ koliduje z
rozmiarem filtra w module 6 i liczbą sąsiadów w module 8 k-NN — kontekst
(dekodowanie vs konwolucja vs wyszukiwanie) jednoznacznie rozróżnia, zgodnie
z tym samym wzorcem co poprzednie kolizje $k$ w tym kursie.

### Top-p (nucleus sampling)

Sortujemy tokeny po prawdopodobieństwie malejąco, bierzemy **najmniejszy**
przedni zbiór, którego **skumulowane** prawdopodobieństwo przekracza próg
$p$, odrzucamy resztę, renormalizujemy.

**Uwaga o notacji.** Próg top-p oznaczamy $p$ — inne znaczenie niż
prawdopodobieństwo dropoutu $p$ w module 5 (tam: prawdopodobieństwo
wyzerowania neuronu). Różne tematy (dekodowanie vs regularizacja), ten sam
symbol, zgodnie z ustaloną w kursie zasadą: kontekst rozróżnia, nota to
nazywa explicite.

**Kluczowa różnica top-k vs top-p**: top-k zachowuje **stałą liczbę**
tokenów niezależnie od tego, jak rozkład wygląda (nawet jeśli jeden token ma
99% masy, i tak zachowujemy $k$ tokenów); top-p zachowuje **zmienną liczbę**
tokenów, dopasowaną do faktycznej koncentracji rozkładu (przy rozkładzie
bardzo skoncentrowanym, top-p może zachować tylko $1$-$2$ tokeny; przy
płaskim — znacznie więcej).

## Przykład rozwiązany: 5-tokenowy słownik

Słownik: `kot, pies, dom, morze, xyz123`. Logity: $z=[2{,}0,\ 1{,}5,\ 0{,}5,\
0{,}1,\ -1{,}0]$.

**Softmax, $\tau=1$:** $\hat y = [0{,}4928,\ 0{,}2989,\ 0{,}1100,\ 0{,}0737,\
0{,}0245]$ (suma $=1$, zweryfikowane).

**Wpływ temperatury** (ten sam $z$, inne $\tau$):

| $\tau$ | kot | pies | dom | morze | xyz123 |
|---|---|---|---|---|---|
| $0{,}5$ (ostrzejszy) | $0{,}6932$ | $0{,}2550$ | $0{,}0345$ | $0{,}0155$ | $0{,}0017$ |
| $1{,}0$ | $0{,}4928$ | $0{,}2989$ | $0{,}1100$ | $0{,}0737$ | $0{,}0245$ |
| $2{,}0$ (płaski) | $0{,}3495$ | $0{,}2722$ | $0{,}1651$ | $0{,}1352$ | $0{,}0780$ |

Mniejsze $\tau$ → rozkład bardziej skoncentrowany na `kot`; większe $\tau$ →
bardziej jednorodny — **ten sam** wyjściowy $z$, tylko inna transformacja.

**Top-k=3** (na $\tau=1$ rozkładzie): zachowuje $\{$`kot`,`pies`,`dom`$\}$,
renormalizacja → $[0{,}5465,\ 0{,}3315,\ 0{,}1220,\ 0,\ 0]$.

**Top-p**: skumulowane prawdopodobieństwa (malejąco) to $0{,}4928,\
0{,}7918,\ 0{,}9017,\ 0{,}9755,\ 1{,}0$. Przy $p=0{,}7$: zachowujemy
$\{$`kot`,`pies`$\}$ (skumulowane $0{,}7918 \geq 0{,}7$, `kot` samo nie
wystarcza: $0{,}4928 < 0{,}7$). Przy $p=0{,}9$: zachowujemy
$\{$`kot`,`pies`,`dom`$\}$ (skumulowane $0{,}9017 \geq 0{,}9$).

Wszystkie liczby zweryfikowane programowo w notebooku (Eksperyment B/C), nie
tylko przeliczone na papierze.

## Tokenizacja: od znaków do podsłów

Trzy poziomy granulacji wejścia, z realnymi kompromisami:

- **Znaki**: słownik mały (dla alfabetu łacińskiego: dziesiątki symboli), ale
  sekwencje **bardzo długie** (każdy znak to krok) — drogie obliczeniowo,
  trudne do wyuczenia długozasięgowych zależności na poziomie znaków.
- **Słowa**: sekwencje krótsze, ale słownik musiałby być **ogromny** (setki
  tysięcy++ unikalnych form) i **nigdy kompletny** — każde nowe słowo,
  nazwa własna, czy literówka to token "nieznany".
- **Subword (BPE i podobne)**: kompromis — częste słowa to **jeden** token,
  rzadkie/nieznane słowa rozbijają się na **znane fragmenty** (podsłowa),
  bez potrzeby jawnego tokenu "`<UNK>`" dla większości przypadków.

**BPE (Byte-Pair Encoding), intuicja bez implementacji od zera**: zaczynamy od
pojedynczych znaków (bajtów), iteracyjnie **łączymy najczęstszą sąsiadującą
parę** w jeden nowy token, powtarzamy do ustalonego rozmiaru słownika. Częste
ciągi (np. "ing", "the") stają się jednym tokenem wcześnie; rzadkie
pozostają rozbite dłużej. Nie implementujemy tego algorytmu — używamy
gotowego tokenizera wytrenowanego razem z modelem (Eksperyment A).

**Token ID i tokeny specjalne**: każdy token ma unikalny identyfikator
całkowity (wiersz w tablicy embeddingów, moduł 8). Modele dodają tokeny
**specjalne**, poza "zwykłym" słownikiem tekstu:

- **BOS** (*beginning of sequence*) — oznacza początek sekwencji.
- **EOS** (*end of sequence*) — oznacza, że model "uznał" generowanie za
  zakończone; `generate()` **zatrzymuje się**, gdy go wygeneruje (zweryfikowane
  w Eksperymencie B).
- **PAD** — wypełnienie do wspólnej długości w batchu (ta sama zasada co w
  module 9: padding to nie treść).

❌ "Każdy model ma inny, osobny token na BOS/EOS/PAD."

✅ Prawda dla niektórych, ale **nie zawsze są to trzy różne tokeny** — w
modelu użytym w tym module `pad_token` i `eos_token` to **ten sam** token
(zweryfikowane: `tokenizer.pad_token_id == tokenizer.eos_token_id`).
Konsekwencja: nie można zakładać, że sprawdzenie "czy to pad" i "czy to eos"
to zawsze dwa różne warunki — zależy od konkretnego tokenizera, trzeba
zweryfikować, nie zgadywać.

❌ "Token to synonim słowa."

✅ Zweryfikowane w Eksperymencie A: angielskie zdanie 9 słów → 10 tokenów
(blisko 1:1), ale polskie zdanie 7 słów → **25** tokenów (ok. 3,6 tokenów na
słowo) — ten sam tokenizer, dramatycznie różna granulacja, bo był trenowany
głównie na angielskim tekście. **Nie można zakładać stałego stosunku
słowa:tokeny, zwłaszcza między językami** — to zmierzony fakt, nie
przypuszczenie.

## Prefill i decode: dwa różne etapy inferencji

Generowanie odpowiedzi na prompt to **dwa jakościowo różne** etapy:

1. **Prefill**: model przetwarza **cały prompt naraz** (jeden forward pass,
   równoległy po pozycjach — ta sama własność co w module 10: self-attention
   nie ma sekwencyjnej zależności między pozycjami **wewnątrz** jednego
   forward passu). Wynik: logity dla *następnego* tokenu po promptcie, plus
   (jeśli `use_cache=True`) zapisane reprezentacje K/V dla każdej pozycji
   promptu.
2. **Decode**: model generuje **jeden token na raz**, każdy krok to **osobny**
   forward pass (tym razem tylko dla **jednej nowej** pozycji, korzystający z
   zapisanego K/V z poprzednich pozycji — stąd KV cache, niżej). Decode jest
   **z definicji sekwencyjny** — krok $t+1$ potrzebuje tokenu wygenerowanego w
   kroku $t$.

Konsekwencja inżynierska: prefill jest (przy tej samej liczbie tokenów)
zwykle **znacznie szybszy na token** niż decode, bo jest **jednym**
równoległym przejściem, nie $T$ **osobnymi** przejściami.

## KV cache: czym jest, a czym nie jest

Bez cache, generowanie tokenu $t{+}1$ wymagałoby **ponownego** przeliczenia
self-attention dla **wszystkich** pozycji $1,\ldots,t$ od zera — mimo że ich
reprezentacje K/V **się nie zmieniają** między krokami (self-attention jest
przyczynowe, moduł 10: pozycja $i$ nie zależy od $j>i$). KV cache **zapamiętuje**
policzone $K,V$ dla już przetworzonych pozycji, żeby nie liczyć ich od nowa.

❌ "KV cache to trwała pamięć wiedzy modelu."

✅ KV cache przechowuje **wyłącznie** policzone reprezentacje $K,V$ dla
pozycji **tego konkretnego, trwającego przebiegu inferencji** — zerowany na
starcie nowej rozmowy, nie zawiera żadnej wiedzy wyuczonej podczas treningu
(ta jest w parametrach $\theta$, nie w cache). Nie zwiększa też zdolności
rozumowania modelu — to wyłącznie **optymalizacja obliczeniowa**, matematycznie
równoważna liczeniu bez cache (różne tylko pod względem szybkości).

**Koszt pamięci cache** (dokładny wzór, nie przybliżenie):

$$\text{bajty} = 2 \times N \times H_{kv} \times d_{\text{head}} \times T \times \text{bytes}_{\text{dtype}}$$

gdzie $N$ = liczba warstw (bloków Transformer) — ten sam $N$, co "$N$ bloków
ułożonych jeden po drugim" w module 10, nie uniwersalny symbol liczby
warstw z modułu 2/6 ($K$), bo $K$ (macierz kluczy) i $V$ (macierz wartości)
pojawiają się chwilę niżej w **tym samym** wzorze/akapicie — użycie $K$ na
"liczbę warstw" tutaj kolidowałoby **lokalnie**, nie tylko w skali kursu.
$H_{kv}$ = liczba **głów KV** (patrz niżej — **nie** zawsze równa liczbie
głów zapytań!), $d_{\text{head}}$ = wymiar jednej głowy, $T$ = długość
sekwencji (ten sam $T$ co zawsze), czynnik $2$ = osobno $K$ i $V$.

❌ "Każdy model ma identyczną liczbę głów zapytań (Q) i głów KV."

✅ **Nieprawda, i to nie teoretyczna uwaga** — model użyty w tym module ma
**9 głów Q, ale tylko 3 głowy KV** (*grouped-query attention*, GQA: kilka
głów zapytań **współdzieli** tę samą głowę KV). Zweryfikowane wprost z
konfiguracji modelu i z rzeczywistego kształtu tensora cache: `keys.shape ==
(batch, 3, T, 64)`, nie `(batch, 9, T, 64)`. GQA to **świadoma** decyzja
projektowa, która **zmniejsza** koszt pamięci cache (mniej głów do
przechowania), nie błąd czy przeoczenie — i oznacza, że wzór wyżej **musi**
używać $H_{kv}$, nie "liczby głów" w ogólności.

Zweryfikowane programowo (Eksperyment E): wzór dał $737\,280$ bajtów dla
$T=16$ pozycji tego konkretnego modelu — **bit-exact** zgodne z rzeczywistym
rozmiarem tensorów `past_key_values` (nie oszacowanie).

## Chat templates i role wiadomości

Modele **instruct** są trenowane na rozmowach o konkretnym, ustalonym
formacie tekstowym — nie na "zwykłym" tekście. Ten format (role `system`/
`user`/`assistant`, specjalne tokeny rozgraniczające wiadomości) jest zapisany
w `tokenizer.chat_template` i aplikowany przez `apply_chat_template()`.

❌ "Chat template to kosmetyczne formatowanie tekstu, nie wpływa na wynik."

✅ Model **nie widział** podczas treningu tekstu w innym formacie niż ten,
na którym był fine-tunowany (instruction tuning, niżej) — podanie promptu
**bez** właściwego formatu to podanie wejścia z **rozkładu, którego model
nigdy nie widział**, co może dramatycznie obniżyć jakość odpowiedzi, nie
tylko "wyglądać nieelegancko". To jest część **kontraktu** z modelem, nie
stylistyka.

Konkretny przykład (model użyty w tym module, format ChatML):
```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```
`add_generation_prompt=True` dodaje **ostatnią** linię (`<|im_start|>assistant\n`)
— sygnał dla modelu "teraz Twoja kolej", bez tego model "nie wie", że powinien
zacząć generować odpowiedź, nie kontynuować wiadomość użytkownika.

❌ "Trzeba samemu dopisać tokeny specjalne (BOS/EOS) do promptu."

✅ `apply_chat_template()` **już** je wstawia (zweryfikowane: tekst po
aplikacji template zaczyna się od `<|im_start|>`, nie od treści). Dopisanie
ich ręcznie **na dodatek** do tego, co template już wstawił, daje
zduplikowane/błędnie sformatowane tokeny — klasyczny, realny błąd
konfiguracji (temat zadania diagnostycznego w laboratorium).

### Historia rozmowy nie jest pamięcią modelu

❌ "Model 'pamięta' poprzednie wiadomości rozmowy."

✅ Model jest **bezstanowy** między wywołaniami — każde wywołanie `generate()`
dostaje **tylko to, co jest w jego wejściu** w tym konkretnym wywołaniu. Jeśli
chcemy, żeby model "pamiętał" wcześniejsze wiadomości, musimy **sami** dołączyć
całą dotychczasową historię rozmowy (wszystkie wiadomości `user`/`assistant`)
do wejścia kolejnego wywołania — co oznacza, że **koszt tokenowy rośnie z
długością rozmowy** (każda kolejna tura płaci za **całą** dotychczasową
historię, nie tylko za nową wiadomość), aż do limitu okna kontekstowego.

## Okno kontekstowe i obcinanie

Model ma **maksymalną** liczbę pozycji, jakie może przyjąć na wejściu
(`max_position_embeddings` w konfiguracji — $8192$ dla modelu użytego tu).
To jest **twardy** limit architektury (positional encoding/cache nie są
zdefiniowane poza nim), nie parametr generowania.

❌ "`max_length` i `max_new_tokens` to to samo."

✅ **Nie** — `max_length` to limit na **całkowitą** długość (prompt +
wygenerowane), `max_new_tokens` to limit **tylko** na nowo wygenerowaną
część. Pomylenie ich jest realnym, częstym błędem: ustawienie `max_length`
zbyt małym względem **już długiego** promptu może **od razu** przyciąć
wejście albo nie wygenerować nic nowego — temat zadania diagnostycznego w
laboratorium.

Długi prompt **zabiera budżet** z okna kontekstowego — jeśli prompt +
pożądana długość odpowiedzi przekracza limit, trzeba **obciąć** coś (zwykle
najstarszą część historii rozmowy) — decyzja inżynierska, nie automatyczna
"magia".

## Base vs instruct, i skąd się biorą (krótko)

- **Model bazowy (base)**: trenowany na **pretrainingu** — przewidywanie
  następnego tokenu na ogromnym korpusie tekstu, bez specjalnego formatu
  rozmowy. Dobry w kontynuowaniu tekstu, nie w "wykonywaniu poleceń".
- **Supervised fine-tuning (SFT)**: dodatkowy trening na parach
  (instrukcja, pożądana odpowiedź) w ustalonym formacie (chat template,
  wyżej) — model uczy się **formatu** rozmowy i **wykonywania** poleceń.
- **Dostrajanie preferencji** (np. DPO — *Direct Preference Optimization*,
  użyte przy modelu z tego modułu): dodatkowy krok, w którym model uczy się
  **preferować** jedną z dwóch odpowiedzi nad drugą (oceniane przez ludzi
  albo inny model) — dalsze dopasowanie stylu/jakości, nie nowa architektura.

❌ "Model instruct to inna architektura niż model bazowy, zaprojektowana,
żeby lepiej wykonywać polecenia."

✅ **Ta sama architektura** (ten sam Transformer) — różnica jest **wyłącznie**
w tym, na czym i jak był dalej trenowany (SFT + preferencje) **po**
pretrainingu. "SmolLM2-135M" i "SmolLM2-135M-Instruct" różnią się wagami
(efektem dodatkowego treningu), nie liczbą warstw czy mechanizmem attention.

Nie rozwijamy tu procesu trenowania (ani SFT, ani DPO) jako głównego tematu —
to świadomie poza zakresem tego modułu (moduły 12+ wracają do trenowania
w innych kontekstach).

## Workflow Hugging Face Transformers

Trzy klasy, które pokrywają prawie wszystko, czego tu potrzeba:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION)

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
```

`AutoTokenizer`/`AutoModelForCausalLM` wybierają właściwą klasę tokenizera/
architektury na podstawie konfiguracji zapisanej razem z modelem — nie
trzeba znać typu architektury z góry. `revision=` **przypina** dokładny
commit wag (zweryfikowana, powtarzalna wersja — nie "cokolwiek jest aktualnie
pod tą nazwą"). `pipeline("text-generation", ...)` istnieje jako wygodna
abstrakcja wysokopoziomowa — pokazujemy ją krótko w notebooku, ale główny
eksperyment **celowo** używa niższego poziomu (`generate`, nie `pipeline`),
żeby etapy inferencji były widoczne, nie schowane.

## Latency, throughput i koszt

- **Latency** (opóźnienie): czas do otrzymania (pierwszego/całego) wyniku.
- **Throughput**: tokenów na sekundę — ile "pracy" system wykonuje w czasie.
- Te dwie miary **nie są** zamienne: wysoki throughput (dużo tokenów/s
  łącznie, np. w batchu wielu zapytań naraz) nie oznacza niskiej latency dla
  **jednego, konkretnego** zapytania — i odwrotnie.

❌ "Krótszy czas inferencji automatycznie oznacza lepszy model."

✅ Czas inferencji zależy od rozmiaru modelu, długości promptu/odpowiedzi,
sprzętu i konfiguracji (cache, precyzja) — **nie** od jakości odpowiedzi.
Mniejszy, szybszy model może dawać gorsze odpowiedzi; większy, wolniejszy —
lepsze. To dwie **niezależne** osie oceny, nie jedna.

Zmierzone wprost (Eksperyment E, CPU, ten komputer — **nie** uniwersalny
benchmark): `use_cache=True` dramatycznie szybsze niż `use_cache=False`,
z przewagą **rosnącą** z długością generowania (bo bez cache koszt **każdego**
kroku rośnie z pozycją — $O(T)$ na krok, $O(T^2)$ łącznie; z cache — $O(1)$
na krok, $O(T)$ łącznie, dla ustalonego rozmiaru modelu).

## Ocena jakości i ograniczenia

❌ "Ta odpowiedź wygląda lepiej, więc jest lepsza."

✅ Subiektywne "wygląda lepiej" nie jest procedurą oceny. Rozdzielamy
**przynajmniej**: (1) poprawność wykonania instrukcji (czy model zrobił to,
o co prosiliśmy — sprawdzalne często automatycznie), (2) poprawność
faktograficzną (tam, gdzie da się zweryfikować względem źródła), (3)
kompletność, (4) zgodność z wymaganym formatem (też często automatycznie
sprawdzalne), (5) powtarzalność (czy wynik jest stabilny, czy przypadkowy
pod sampling). Rozwijamy to w laboratorium jako systematyczną procedurę, nie
wrażenie.

❌ "Mały model lokalny oferuje jakość dużego modelu komercyjnego."

✅ Model użyty w tym module ma $135$ milionów parametrów — rząd wielkości (albo
więcej) mniej niż współczesne duże modele komercyjne. Oczekujemy spójnych,
gramatycznych odpowiedzi na proste zadania (zweryfikowane), **nie**
rozumowania wieloetapowego, głębokiej wiedzy specjalistycznej, czy
niezawodnej poprawności faktograficznej na złożonych pytaniach.

## Halucynacje i podstawowe ryzyka (krótko — nie osobny wykład)

- **Halucynacje**: model generuje tekst **brzmiący** wiarygodnie i gramatycznie,
  **niezależnie** od tego, czy jest faktycznie prawdziwy — model nie ma
  wbudowanego mechanizmu odróżniania "wiem to na pewno" od "to brzmi
  prawdopodobnie". Poprawna składnia i gramatyka **nie gwarantują**
  poprawności merytorycznej — to dwa zupełnie inne kryteria.
- **Niestabilność przy sampling**: ten sam prompt, inny wynik na innym
  przebiegu (bez ustalonego ziarna) — istotne przy ocenie ("czy to był
  pech, czy systematyczny problem?").
- **Wrażliwość na formę promptu**: inne sformułowanie tego samego pytania
  może dać inną jakość odpowiedzi — nie tylko inny tekst.
- **Prompt injection**: jeśli model przetwarza tekst z **nieufnego** źródła
  (np. treść strony internetowej, e-mail, dokument od użytkownika) **razem**
  z instrukcją systemową, ten nieufny tekst może zawierać sformułowania
  **wyglądające jak instrukcje** ("zignoruj powyższe i zrób X") — traktowanie
  treści danych jako polecenia jest ryzykiem bezpieczeństwa, nie tylko
  teoretyczną ciekawostką.
- **Dane poufne**: wysyłanie promptów do zewnętrznej usługi (API modelu
  komercyjnego) oznacza przekazanie treści promptu stronie trzeciej —
  konsekwencja do rozważenia **przed**, nie po.
- **Licencje**: wagi modelu i dane treningowe mają własne licencje/warunki
  (ten moduł: Apache 2.0, zweryfikowane — zobacz `data/registry.yml`) — nie
  każdy model jest "wolny do użycia w każdym celu" tylko dlatego, że jest
  publicznie dostępny.
- **Wygenerowana treść jako "zweryfikowany fakt"**: traktowanie odpowiedzi
  modelu jako automatycznie prawdziwej (bez weryfikacji) jest błędem
  użytkownika/systemu, nie modelu — model nigdy nie obiecał poprawności.

**Trzy różne, niezależne kryteria — żadne nie gwarantuje pozostałych:**

1. **Poprawne wykonanie kodu** — jeśli model generuje kod, czy ten kod
   **faktycznie się wykonuje** bez błędu (kwestia składniowa języka
   programowania, nie języka naturalnego).
2. **Poprawność składni wygenerowanego tekstu** — czy tekst jest
   gramatycznie/strukturalnie dobrze sformułowany (np. w tym module: czy
   odpowiedź jest płynnym, gramatycznym angielskim).
3. **Poprawność merytoryczna** — czy treść odpowiedzi jest **faktycznie
   prawdziwa/trafna**.

Kod może się wykonać bez błędu i dać **złą** odpowiedź (błąd logiczny, nie
składniowy). Tekst może być gramatycznie bezbłędny i **merytorycznie
fałszywy** (halucynacja, wyżej). Żadne z tych trzech kryteriów automatycznie
nie implikuje pozostałych — oceniając wygenerowaną treść (sekcja "Ocena
jakości" wyżej, i laboratorium), trzeba sprawdzić **każde z nich osobno**.

To jest **wprowadzenie** do tych tematów (świadomość, nie głęboka analiza) —
pełne bezpieczeństwo LLM to osobny obszar, poza zakresem tego modułu.

## Co obserwujemy

Wszystkie liczby niżej pochodzą z rzeczywistego wykonania `examples.ipynb`
(CPU, ten komputer — nie uniwersalny benchmark) — zobacz notebook po pełne
wyjście.

- **Eksperyment A (tokenizacja).** Zgodnie z tabelą wyżej: angielskie zdanie
  $0{,}90$ słowa/token, polskie $0{,}28$ (ok. $3{,}6\times$ więcej tokenów na
  słowo), liczby dzielone na pojedyncze cyfry (`1234.56` →
  `1`,`2`,`3`,`4`,`.`,`5`,`6`), znak `§` (poza ASCII) daje bajtowy artefakt BPE
  (`ĠÂ§`), rzadkie/długie słowa rozbite na sensowne podsłowa, nigdy na jeden
  token `<UNK>`.
- **Eksperyment B.** Dla promptu "The capital of France is" (po chat
  template, $35$ tokenów): top-5 następnych tokenów to `The` ($p=0{,}3113$),
  `B` ($0{,}0674$), `"` ($0{,}0616$), `Bon` ($0{,}0353$), `Paris`
  ($0{,}0344$) — model **nie** jest od razu pewny jednego tokenu. Temperature
  na tym samym rozkładzie: $\tau=0{,}5$ → `The` skacze do $0{,}866$;
  $\tau=2{,}0$ → `The` spada do $0{,}026$ (ten sam $z_t$, inna koncentracja).
  Ręczna pętla generowania (8 kroków, zatrzymana przez EOS) dała token po
  tokenie **identyczne** jak `model.generate(do_sample=False)`: `The capital
  of France is Paris.` — zweryfikowane programowo (`==` na liście id, nie
  deklaracja).
- **Eksperyment C.** Na otwartym, kreatywnym prompcie (robot odkrywa muzykę):
  greedy, sampling ($\tau=0{,}7$/$1{,}5$), top-k=10, top-p=0,9 dały **pięć
  wyraźnie różnych** kontynuacji. Na skoncentrowanym, faktograficznym
  prompcie ("capital of France"): wszystkie pięć strategii dało **tę samą
  odpowiedź** ("Paris") z drobnymi różnicami tylko w kontynuacji **po** niej
  — ta sama konfiguracja, radykalnie różna widoczna różnorodność, bo zależy
  od **kształtu rozkładu** dla danego promptu, nie tylko od parametrów
  samplingu. Z ustalonym ziarnem (`torch.manual_seed(42)`): dwa przebiegi
  dały **identyczny** tekst.
- **Eksperyment D.** Zwykły tekst (bez chat template, 7 tokenów) dał **pusty**
  wygenerowany tekst (`''`) — model natychmiast wygenerował EOS, bo nigdy nie
  widział tego formatu podczas fine-tuningu; z chat template (37 tokenów, z
  automatycznie dodanym system promptem) dał poprawną odpowiedź "The capital
  of France is Paris." Historia rozmowy: po pierwszej turze $36$ tokenów, po
  drugiej (z pełną historią dołączoną) $72$ tokeny — model odpowiedział
  poprawnie na "What is my name?" **tylko** dlatego, że pierwsza tura została
  jawnie dołączona do wejścia.
- **Eksperyment E.** `use_cache=True` wyraźnie szybsze niż `False`, z
  przewagą rosnącą z długością: przy $16$ nowych tokenach $0{,}18$s vs
  $0{,}44$s ($2{,}5\times$); przy $128$ tokenach $1{,}29$s vs $5{,}64$s
  ($4{,}4\times$) — rosnąca przewaga, zgodna z teorią ($O(T)$ na krok bez
  cache vs $O(1)$ na krok z cache). Dłuższy prompt ($437$ vs $31$ tokenów,
  to samo generowanie=$32$ tokeny): $0{,}47$s vs $0{,}32$s — dłuższy prefill,
  ale decode wciąż dominuje przy tej skali. Wzór pamięci cache
  ($2NH_{kv}d_{\text{head}}T\cdot\text{bytes}$) dał $737\,280$ bajtów dla
  $T=16$ — **dokładnie** (identyczność bitowa, nie przybliżenie) zgodne z
  rzeczywistym rozmiarem tensorów `past_key_values.layers[i].keys/values`.
  Model: $9$ głów Q, **$3$ głowy KV** — GQA, nie błąd.

## Częste błędne przekonania

(Zbiorcze podsumowanie — każde ma pełny ❌/✅ blok wyżej, w odpowiedniej
sekcji merytorycznej)

1. LLM nie generuje całej odpowiedzi jednocześnie w typowym autoregresyjnym
   dekodowaniu.
2. Token nie jest synonimem słowa.
3. Większa temperatura nie gwarantuje lepszej odpowiedzi (i nie ma żadnego
   wpływu, gdy `do_sample=False`).
4. Greedy decoding nie musi wybierać globalnie najbardziej prawdopodobnej
   kompletnej sekwencji.
5. Dłuższy prompt nie musi poprawiać jakości.
6. Kontekst modelu nie jest trwałą pamięcią — historia rozmowy musi być
   jawnie dołączona do każdego wywołania.
7. KV cache nie przechowuje całej wiedzy wyuczonej przez model — tylko K/V
   bieżącego przebiegu.
8. Model instruct nie jest inną architekturą wyłącznie dlatego, że lepiej
   wykonuje polecenia — różni się dalszym treningiem (SFT + preferencje).
9. Chat template nie jest wyłącznie kosmetycznym formatowaniem tekstu.
10. Mniejszy czas inferencji nie oznacza automatycznie lepszego modelu.
11. Poprawnie wygenerowany (gramatycznie) tekst może zawierać błędy
    faktograficzne.
12. Model lokalny nie musi oferować jakości dużego modelu udostępnianego
    komercyjnie.

## Implikacje inżynierskie

- **Koszt rośnie z długością generowania, nie tylko z rozmiarem modelu.**
  Decode jest sekwencyjny z definicji — podwojenie `max_new_tokens` z grubsza
  podwaja czas decode (zmierzone: throughput w tok/s względnie stabilny z
  cache, ale **całkowity** czas rośnie liniowo z liczbą generowanych tokenów).
- **KV cache to kompromis pamięć-za-czas**, nie "darmowe" przyspieszenie —
  rozmiar cache rośnie **liniowo** z długością sekwencji i **liniowo** z
  liczbą warstw/głów KV (wzór wyżej) — dla bardzo długich kontekstów/wielu
  równoległych rozmów, pamięć cache może stać się realnym ograniczeniem.
- **GQA (mniej głów KV niż głów Q) to świadoma optymalizacja** zmniejszająca
  koszt pamięci cache — nie każdy model z tej samej "rodziny" architektur ma
  identyczną konfigurację, trzeba sprawdzić `config.json`, nie zgadywać.
- **Prefill vs decode mają różną charakterystykę kosztu** — długi prompt
  kosztuje głównie w prefill (jeden, równoległy przebieg), długa odpowiedź
  kosztuje głównie w decode (sekwencyjne kroki) — optymalizacja jednego nie
  automatycznie optymalizuje drugiego.
- **Format promptu (chat template) jest częścią kontraktu z modelem**, nie
  szczegółem stylistycznym — błędny format to cichy spadek jakości, nie
  zawsze wyjątek/błąd.
- **Pomylenie `max_length` z `max_new_tokens`** jest realnym, częstym błędem
  konfiguracji z konkretnymi, złymi konsekwencjami (przycięte wejście albo
  brak nowej treści) — nie teoretyczną pomyłką.

## Co warto zapamiętać

- $P(x_1,\ldots,x_T) = \prod_{t=1}^T P_\theta(x_t\mid x_{<t})$ — dokładna
  tożsamość, nie założenie; założenie jest w tym, że sieć neuronowa dobrze
  przybliża każdy czynnik.
- Logity → softmax → rozkład; temperature skaluje logity **przed**
  softmaxem i **nie ma wpływu** przy `do_sample=False`.
- Top-k zachowuje stałą liczbę tokenów, top-p zachowuje zmienną liczbę
  dopasowaną do koncentracji rozkładu.
- Tokenizacja subword to kompromis znaki/słowa — liczba tokenów na słowo
  **nie** jest stała, zwłaszcza między językami (zmierzone: 1,1 dla
  angielskiego, 3,6 dla polskiego, ten sam tokenizer).
- Prefill (równoległy, cały prompt) i decode (sekwencyjny, token po tokenie)
  to różne etapy z różną charakterystyką kosztu; KV cache przyspiesza decode,
  unikając ponownego liczenia niezmiennych K/V.
- Chat template, BOS/EOS/PAD są częścią kontraktu z modelem, nie
  kosmetyką — i nie zawsze są to trzy różne tokeny.
- Model instruct = ten sam Transformer + SFT + dostrajanie preferencji, nie
  inna architektura.

## Pytania dla studentów

1. Masz rozkład $\hat y_t$ bardzo skoncentrowany (jeden token ma $99\%$ masy).
   Czy top-k=$50$ i top-p=$0{,}9$ dadzą ten sam efektywny zbiór tokenów do
   wylosowania? Uzasadnij.
2. Dlaczego `do_sample=False` i `temperature=5.0` dają identyczny wynik co
   `do_sample=False` i `temperature=1.0`?
3. Masz model z $40$ warstwami, $8$ głowami KV, wymiarem głowy $128$,
   precyzją fp16 ($2$ bajty). Jaki jest koszt pamięci KV cache dla sekwencji
   długości $4096$? Jak zmienia się ten koszt, jeśli liczba głów KV spadnie
   do $2$ (GQA)?
4. Masz rozmowę wieloturową z modelem. Dlaczego koszt (w tokenach) **każdej
   kolejnej** tury rośnie, nawet jeśli nowa wiadomość użytkownika jest
   bardzo krótka?

## Bibliografia

- Ben Allal, L., Lozhkov, A., Bakouch, E., i in. (2025). *SmolLM2: When Smol
  Goes Big — Data-Centric Training of a Small Language Model*.
  arXiv:2502.02737.
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation
  of Rare Words with Subword Units*. ACL (wprowadzenie BPE do NLP).
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The
  Curious Case of Neural Text Degeneration*. ICLR (nucleus sampling / top-p).
- Vaswani, A., i in. (2017). *Attention Is All You Need*. NeurIPS 30 (zobacz
  też moduł 10 — architektura Transformer, nie wyprowadzana ponownie tutaj).
- Pope, R., i in. (2022). *Efficiently Scaling Transformer Inference*.
  arXiv:2211.05102 (KV cache, koszt inferencji).
- Hugging Face Transformers documentation: *Generation strategies*,
  `https://huggingface.co/docs/transformers/generation_strategies`.
- Hugging Face Transformers documentation: *Chat templates*,
  `https://huggingface.co/docs/transformers/chat_templating`.
- Model card: *HuggingFaceTB/SmolLM2-135M-Instruct*,
  `https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct`.
