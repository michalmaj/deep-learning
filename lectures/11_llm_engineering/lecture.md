Język: [English](lecture.en.md) | **Polski**

# Wykład 11. Od Transformera do systemu generowania tekstu

## Cele wykładu

1. opisać autoregresyjne modelowanie języka za pomocą rozkładów następnego tokenu,
2. obliczyć softmax oraz wpływ temperature, top-k i top-p,
3. wyjaśnić tokenizację subword i jej zależność od języka,
4. rozróżnić etapy prefill i decode oraz określić rolę KV cache,
5. poprawnie użyć chat template i `model.generate()`,
6. oddzielić wykonanie instrukcji, płynność tekstu i poprawność faktograficzną.

## Dlaczego to jest ważne

Moduły 8-10 wprowadziły embeddingi, sekwencje, self-attention i blok Transformer. Ten moduł nie dodaje nowej matematyki architektury. Pokazuje, jak logity wytrenowanego modelu zmieniają się w tekst.

Strategia dekodowania, format promptu, pamięć i budżet czasowy wpływają na działanie systemu. Są to decyzje inżynierskie niezależne od samej architektury.

## Intuicja

Model językowy oblicza po jednym wyniku dla każdego tokenu słownika. Wyniki opisują możliwy następny token, a nie całą gotową odpowiedź.

Generowanie powtarza ten sam cykl. Powstaje rozkład, wybierany jest token, token trafia do kontekstu i model wykonuje kolejny krok.

❌ Model autoregresyjny generuje całą odpowiedź jednocześnie.

✅ Typowe dekodowanie tworzy jeden token na raz. Każdy krok zależy także od tokenów wygenerowanych wcześniej przez model.

## Autoregresyjna faktoryzacja

Dla identyfikatorów tokenów $x_1,\ldots,x_T$ ze słownika o rozmiarze $V$ reguła łańcuchowa daje

$$P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1}).$$

Jest to tożsamość dla dowolnego rozkładu łącznego. Założenie modelu dotyczy możliwości przybliżenia każdego czynnika przez sieć z parametrami $\theta$.

$$P_\theta(x_t\mid x_{<t}),\qquad x_{<t}:=x_1,\ldots,x_{t-1}$$

Tutaj $x_t$ oznacza identyfikator całkowity, a nie wektor z modułu 9. Pipeline ma postać $x_t$ jako identyfikator, $\mathbf e_t$ jako embedding i $z_t$ jako logity. Symbol $V$ nadal oznacza rozmiar słownika.

## Logity i softmax

Dla kontekstu $x_{<t}$ model produkuje jeden logit na token.

$$z_t=f_\theta(x_{<t})\in\mathbb R^V$$

Logity mogą być ujemne i nie sumują się do 1. Softmax zamienia je w rozkład.

$$\hat y_t=\operatorname{softmax}(z_t),\qquad \hat y_{t,i}=\frac{e^{z_{t,i}}}{\sum_{j=1}^{V}e^{z_{t,j}}}$$

Składowa $\hat y_{t,i}$ przybliża $P_\theta(x_t=i\mid x_{<t})$.

❌ Najbardziej prawdopodobny token w każdym kroku gwarantuje najlepszą całą odpowiedź.

✅ Rozkład dotyczy jednego kroku. Lokalnie najlepsze decyzje nie muszą tworzyć globalnie najbardziej prawdopodobnej sekwencji.

## Temperature

Temperature skaluje logity przed softmaxem.

$$\hat y_t^{(\tau)}=\operatorname{softmax}\!\left(\frac{z_t}{\tau}\right),\qquad \tau>0$$

Symbol $\tau$ odróżnia temperature od długości sekwencji $T$.

- Dla $\tau<1$ różnice między logitami rosną, więc rozkład jest ostrzejszy.
- Dla $\tau>1$ rozkład staje się bardziej płaski.
- Dla $\tau\to0^+$ cała masa skupia się na $\arg\max_i z_{t,i}$.

❌ Temperature zwiększa kreatywność modelu.

✅ Jest to matematyczna transformacja rozkładu, a nie zmiana wiedzy modelu. Temperature nie wybiera tokenu. Przy `do_sample=False` nie wpływa na wynik, ponieważ skalowanie nie zmienia argmaxu.

## Wybór tokenu

### Greedy decoding

Greedy decoding wybiera $x_t=\arg\max_i\hat y_{t,i}$ i jest deterministyczny.

Nie gwarantuje globalnie najlepszej sekwencji. W kontrprzykładzie z notebooka $P(A)=0{,}5498$ i $P(B)=0{,}4502$, więc pierwszy krok wybiera $A$. Następnie $P(X\mid A)=P(Y\mid A)=0{,}5$, a $P(X\mid B)=0{,}99$.

Greedy tworzy $(A,X)$ z prawdopodobieństwem $0{,}2749$. Sekwencja $(B,X)$ ma większe prawdopodobieństwo $0{,}4457$, ale lokalna decyzja odrzuca ją w pierwszym kroku. Notebook potwierdza wynik przez enumerację czterech sekwencji.

### Sampling

Sampling losuje $x_t\sim\hat y_t^{(\tau)}$. Bez ustalonego ziarna wynik może się zmieniać. Ustawienie ziarna PyTorch zapewnia powtarzalność sprawdzoną w notebooku.

### Top-k

Top-k pozostawia $k$ tokenów o najwyższych wynikach, zeruje pozostałe prawdopodobieństwa i renormalizuje rozkład. Liczba zachowanych tokenów jest stała.

### Top-p

Top-p sortuje tokeny malejąco i zachowuje najmniejszy początkowy zbiór, którego skumulowane prawdopodobieństwo przekracza $p$. Liczba tokenów zależy od koncentracji rozkładu.

Przy rozkładzie skupionym top-p może pozostawić jeden lub dwa tokeny. Przy płaskim rozkładzie zachowa ich więcej. Top-k nie dostosowuje liczności zbioru w ten sposób.

## Przykład z pięcioma tokenami

Dla słownika `kot, pies, dom, morze, xyz123` i logitów $z=[2{,}0,1{,}5,0{,}5,0{,}1,-1{,}0]$ softmax przy $\tau=1$ daje

$$\hat y=[0{,}4928,0{,}2989,0{,}1100,0{,}0737,0{,}0245].$$

| $\tau$ | kot | pies | dom | morze | xyz123 |
|---|---:|---:|---:|---:|---:|
| $0{,}5$ | $0{,}6932$ | $0{,}2550$ | $0{,}0345$ | $0{,}0155$ | $0{,}0017$ |
| $1{,}0$ | $0{,}4928$ | $0{,}2989$ | $0{,}1100$ | $0{,}0737$ | $0{,}0245$ |
| $2{,}0$ | $0{,}3495$ | $0{,}2722$ | $0{,}1651$ | $0{,}1352$ | $0{,}0780$ |

Top-k z $k=3$ zachowuje `kot`, `pies` i `dom`, a po renormalizacji daje $[0{,}5465,0{,}3315,0{,}1220,0,0]$.

Skumulowane prawdopodobieństwa wynoszą $0{,}4928$, $0{,}7918$, $0{,}9017$, $0{,}9755$ i $1{,}0$. Top-p dla $p=0{,}7$ zachowuje `kot` i `pies`, a dla $p=0{,}9$ także `dom`.

## Tokenizacja subword

Trzy poziomy granulacji mają różne koszty.

- Znaki wymagają małego słownika, ale tworzą bardzo długie sekwencje.
- Słowa skracają sekwencje, lecz wymagają ogromnego i zawsze niepełnego słownika.
- Podsłowa pozwalają zapisać częste słowa jednym tokenem, a rzadkie rozłożyć na znane fragmenty.

BPE zaczyna od znaków lub bajtów i wielokrotnie łączy najczęstsze sąsiadujące pary. W module używany jest gotowy tokenizer wytrenowany dla konkretnego modelu.

Każdy token ma identyfikator wskazujący wiersz tablicy embeddingów. Tokeny specjalne mogą obejmować BOS, EOS i PAD. `generate()` kończy pracę po wygenerowaniu EOS.

❌ BOS, EOS i PAD zawsze są trzema różnymi tokenami.

✅ W używanym modelu `pad_token_id == eos_token_id`. Znaczenie identyfikatorów trzeba sprawdzić w konkretnym tokenizerze.

❌ Jeden token zawsze odpowiada jednemu słowu.

✅ Angielskie zdanie z 9 słów ma 10 tokenów, a polskie zdanie z 7 słów ma 25 tokenów. Ten sam tokenizer tworzy około 3,6 tokenu na polskie słowo, ponieważ był trenowany głównie na angielskim tekście.

## Prefill i decode

Generowanie składa się z dwóch etapów.

1. **Prefill** przetwarza cały prompt w jednym przejściu równoległym po pozycjach. Powstają logity następnego tokenu oraz, przy `use_cache=True`, wartości K i V dla promptu.
2. **Decode** generuje po jednym tokenie. Każdy nowy token wymaga osobnego przejścia i zależy od poprzedniego kroku.

Prefill jest zwykle szybszy w przeliczeniu na token, ponieważ wykorzystuje równoległość pozycji. Decode pozostaje sekwencyjny.

## KV cache

Bez cache każdy krok ponownie obliczałby K i V dla wszystkich wcześniejszych pozycji. Przy attention przyczynowym te reprezentacje nie zmieniają się, więc mogą zostać zachowane.

❌ KV cache jest trwałą pamięcią wiedzy modelu.

✅ Cache przechowuje K i V tylko dla bieżącego przebiegu inferencji. Wiedza modelu znajduje się w parametrach $\theta$. Cache jest optymalizacją obliczeniową i nie zmienia wyniku matematycznego ani zdolności rozumowania.

Dokładny koszt pamięci wynosi

$$\text{bajty}=2\times N\times H_{kv}\times d_{\text{head}}\times T\times\text{bytes}_{\text{dtype}}.$$

$N$ oznacza liczbę bloków, $H_{kv}$ liczbę głów K/V, $d_{\text{head}}$ wymiar głowy, a czynnik 2 odpowiada osobnym K i V.

❌ Liczba głów zapytań zawsze jest równa liczbie głów K/V.

✅ Używany model ma 9 głów Q i 3 głowy K/V. W grouped-query attention kilka głów zapytań współdzieli jedną głowę K/V, co zmniejsza pamięć cache. Tensor ma kształt `(batch, 3, T, 64)`.

Dla $T=16$ wzór daje dokładnie 737 280 bajtów, zgodnie z rzeczywistym rozmiarem tensorów `past_key_values`.

## Chat templates

Model instruct jest trenowany na rozmowach w określonym formacie. Role `system`, `user` i `assistant` oraz tokeny rozdzielające są zapisane w `tokenizer.chat_template`.

❌ Chat template jest tylko kosmetycznym formatowaniem.

✅ Format jest częścią kontraktu z modelem. Prompt spoza rozkładu używanego podczas dostrajania może znacznie obniżyć jakość bez zgłaszania błędu.

Przykład formatu ChatML:

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

`add_generation_prompt=True` dodaje początek wiadomości asystenta. `apply_chat_template()` sam wstawia wymagane tokeny specjalne. Ręczne dodanie ich drugi raz powoduje błędne formatowanie.

### Historia rozmowy

Model jest bezstanowy między wywołaniami. Poprzednie wiadomości muszą zostać jawnie dołączone do kolejnego wejścia. Koszt każdej tury rośnie wraz z całą historią aż do limitu kontekstu.

## Okno kontekstowe

`max_position_embeddings` określa maksymalną liczbę pozycji. Używany model ma limit 8192.

❌ `max_length` i `max_new_tokens` oznaczają to samo.

✅ `max_length` ogranicza łączną długość promptu i odpowiedzi. `max_new_tokens` ogranicza tylko nową część. Długi prompt zmniejsza miejsce dostępne na odpowiedź, a przekroczenie okna wymaga świadomego obcięcia treści.

## Model base i instruct

- Model bazowy uczy się przewidywania następnego tokenu podczas pretrainingu.
- SFT dostraja go na parach instrukcja i odpowiedź w formacie rozmowy.
- Dostrajanie preferencji, na przykład DPO, uczy preferować jedną odpowiedź nad inną.

❌ Model instruct ma inną architekturę niż model bazowy.

✅ Architektura pozostaje taka sama. Dodatkowy trening zmienia wagi i zachowanie, a nie liczbę warstw lub mechanizm attention.

## Workflow Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION)

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
```

Klasy `AutoTokenizer` i `AutoModelForCausalLM` wybierają implementację z konfiguracji modelu. `revision` przypina konkretny commit wag. `pipeline("text-generation")` jest wygodną abstrakcją, ale niższy poziom `generate()` lepiej pokazuje etapy inferencji.

## Latency, throughput i koszt

Latency oznacza czas oczekiwania na wynik, a throughput liczbę tokenów przetwarzanych na sekundę. Wysoki łączny throughput nie gwarantuje niskiego opóźnienia pojedynczego żądania.

❌ Krótszy czas inferencji oznacza lepszy model.

✅ Czas zależy od modelu, długości, sprzętu, cache i precyzji. Jakość odpowiedzi jest osobną osią oceny.

Bez cache koszt kolejnego kroku rośnie wraz z pozycją. Dla ustalonego modelu daje to w uproszczeniu $O(T^2)$ dla całego decode. Cache ogranicza ponowne obliczenia i prowadzi do kosztu $O(T)$, płacąc dodatkową pamięcią.

## Ocena jakości i ryzyka

Ocena powinna rozdzielać wykonanie instrukcji, poprawność faktograficzną, kompletność, wymagany format i powtarzalność. Subiektywne wrażenie nie wystarcza.

Model z modułu ma 135 milionów parametrów. Może tworzyć płynne odpowiedzi na proste zadania, ale nie należy oczekiwać niezawodnego rozumowania wieloetapowego ani specjalistycznej wiedzy.

Podstawowe ryzyka obejmują:

- halucynacje, czyli wiarygodnie brzmiący, lecz fałszywy tekst,
- niestabilność przy samplingu,
- wrażliwość na sformułowanie promptu,
- prompt injection w nieufnych danych,
- przekazywanie poufnych treści zewnętrznej usłudze,
- ograniczenia licencji modelu i danych,
- traktowanie wygenerowanej treści jako zweryfikowanego faktu.

Poprawne wykonanie kodu, gramatyczna poprawność tekstu i poprawność merytoryczna są niezależnymi kryteriami. Każde wymaga osobnego sprawdzenia.

## Eksperymenty i obserwacje

Wyniki pochodzą z wykonanego [`examples.ipynb`](examples.ipynb) na CPU i nie stanowią uniwersalnego benchmarku.

### A. Tokenizacja

Angielskie zdanie osiąga 0,90 słowa na token, a polskie 0,28. Liczba `1234.56` dzieli się na pojedyncze cyfry i znak kropki. Znak `§` tworzy bajtowy artefakt `ĠÂ§`. Rzadkie słowa są rozbijane na podsłowa zamiast `<UNK>`.

### B. Rozkład i generowanie greedy

Prompt `The capital of France is` ma po chat template 35 tokenów. Pięć najwyższych następnych tokenów to `The` z $0{,}3113$, `B` z $0{,}0674$, `"` z $0{,}0616$, `Bon` z $0{,}0353$ oraz `Paris` z $0{,}0344$.

Dla $\tau=0{,}5$ prawdopodobieństwo `The` rośnie do $0{,}866$, a dla $\tau=2{,}0$ spada do $0{,}026$. Ręczna pętla ośmiu kroków, zakończona przez EOS, daje te same identyfikatory co `model.generate(do_sample=False)` i tekst `The capital of France is Paris.`.

### C. Strategie dekodowania

Dla otwartego promptu o robocie odkrywającym muzykę greedy, dwa poziomy temperature, top-k i top-p tworzą pięć różnych kontynuacji. Dla pytania o stolicę Francji wszystkie strategie odpowiadają `Paris`, ponieważ rozkład jest bardziej skoncentrowany. Dwa przebiegi z `torch.manual_seed(42)` dają identyczny tekst.

### D. Format i historia

Zwykły tekst bez chat template ma 7 tokenów i prowadzi do natychmiastowego EOS oraz pustej odpowiedzi. Chat template tworzy 37 tokenów wraz z promptem systemowym i prowadzi do odpowiedzi `The capital of France is Paris.`.

Historia ma 36 tokenów po pierwszej turze i 72 po drugiej. Model odpowiada na pytanie o imię tylko wtedy, gdy wcześniejsza wiadomość jest dołączona do wejścia.

### E. Cache i długość

Przy 16 nowych tokenach czas z cache wynosi 0,18 s, a bez cache 0,44 s. Przy 128 tokenach wyniki to 1,29 s i 5,64 s. Przewaga rośnie z około $2{,}5\times$ do $4{,}4\times$.

Dla promptów długości 31 i 437 tokenów oraz odpowiedzi długości 32 czas wynosi odpowiednio 0,32 s i 0,47 s. Dłuższy prefill zwiększa koszt, lecz w tym eksperymencie decode nadal dominuje.

## Implikacje inżynierskie

- Czas rośnie z liczbą generowanych tokenów, ponieważ decode jest sekwencyjny.
- KV cache wymienia dodatkową pamięć na krótszy czas i rośnie liniowo z długością.
- GQA zmniejsza pamięć przez mniejszą liczbę głów K/V niż Q.
- Prefill i decode mają różne profile kosztu.
- Chat template jest częścią interfejsu modelu.
- Pomylenie `max_length` z `max_new_tokens` może uciąć wejście lub zablokować generowanie.

## Co warto zapamiętać

- Reguła łańcuchowa rozkłada prawdopodobieństwo sekwencji na rozkłady następnego tokenu.
- Temperature zmienia koncentrację rozkładu i nie wpływa na greedy decoding.
- Top-k zachowuje stałą, a top-p zmienną liczbę tokenów.
- Liczba tokenów na słowo zależy od tokenizera i języka.
- Prefill jest równoległy po pozycjach, a decode sekwencyjny.
- KV cache zachowuje K i V bieżącego przebiegu, nie wiedzę modelu.
- Chat template i tokeny specjalne należą do kontraktu z modelem.
- Model instruct różni się treningiem, nie architekturą.

## Pytania

1. Czy top-k z $k=50$ i top-p z $p=0{,}9$ zachowają ten sam zbiór, jeśli jeden token ma 99% masy?
2. Dlaczego temperature nie zmienia wyniku przy `do_sample=False`?
3. Jaki jest koszt KV cache dla 40 warstw, 8 głów K/V, wymiaru 128, fp16 i $T=4096$? Jak zmieni się dla 2 głów K/V?
4. Dlaczego koszt tokenowy kolejnej tury rozmowy rośnie mimo krótkiej nowej wiadomości?

## Bibliografia

- Ben Allal, L. et al. (2025). *SmolLM2: When Smol Goes Big. Data-Centric Training of a Small Language Model*. arXiv:2502.02737.
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL.
- Holtzman, A. et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 30.
- Pope, R. et al. (2022). *Efficiently Scaling Transformer Inference*. arXiv:2211.05102.
- Hugging Face Transformers documentation. *Generation strategies*. `https://huggingface.co/docs/transformers/generation_strategies`.
- Hugging Face Transformers documentation. *Chat templates*. `https://huggingface.co/docs/transformers/chat_templating`.
- Model card. *HuggingFaceTB/SmolLM2-135M-Instruct*. `https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct`.
