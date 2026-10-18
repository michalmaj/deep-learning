# Laboratorium 11: projektowanie i ocena niewielkiego pipeline'u inferencji LLM

Czas: ok. 2 godziny (**105 minut rdzenia obowiązkowego + 15 minut bufora**).
Bazuje na
[`lectures/11_llm_engineering/lecture.md`](../../lectures/11_llm_engineering/lecture.md)
(tokenizacja, next-token prediction, strategie dekodowania, chat templates,
KV cache), ale to **inżynierski eksperyment**, nie kolejne zadanie
klasyfikacyjne: budujesz i oceniasz mały, powtarzalny pipeline inferencji.

**Sekcje 0-11 to rdzeń obowiązkowy. Sekcje R1-R2 (na końcu notebooka) są
opcjonalnymi rozszerzeniami** — wykonaj je tylko, jeśli zostanie czas po
zakończeniu rdzenia. Żadne kryterium oceny rdzenia nie wymaga ukończenia
rozszerzeń.

**Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (ten sam co w wykładzie,
Apache 2.0 — zobacz [`data/registry.yml`](../../data/registry.yml)). To
model na **135 milionów** parametrów — oczekuj spójnych, gramatycznych
odpowiedzi na proste zadania, **nie** rozumowania wieloetapowego ani
niezawodnej poprawności faktograficznej. Prompty w tym laboratorium są po
angielsku (model działa wyraźnie lepiej po angielsku — zweryfikowane w
wykładzie), ale wszystkie instrukcje i Twoje odpowiedzi — po polsku.

**Pipeline, który budujesz**: prompt → tokenizacja → model → generowanie →
dekodowanie → pomiar → ocena.

## 0. Konfiguracja (gotowe)

Import bibliotek, urządzenie, ziarno losowości, `torch.set_num_threads(4)`.

## 1. Model i tokenizer (gotowe + krótkie zadanie)

Ładujemy model i tokenizer (`revision=` przypięty do konkretnego commitu —
powtarzalność, nie "najnowsza wersja").

**Zadanie:** wypisz `model.config.num_attention_heads` i
`model.config.num_key_value_heads`. Czy są równe? Co to oznacza dla kosztu
pamięci KV cache (zobacz wykład)?

## 2. Przygotowanie promptów — Twoje zadanie

Wybierz **dokładnie 3 obowiązkowe** prompty, pokrywające:

1. **jeden** prompt jednoznaczny typu **ekstrakcja informacji** z krótkiego
   tekstu (ma jednoznaczną oczekiwaną odpowiedź — automatycznie sprawdzalna,
   np. przez `in`),
2. **jeden drugi** prompt jednoznaczny, innego typu (np. **przekształcenie
   formatu** — data w innym formacie),
3. **jeden** prompt **jakościowy** (parafraza albo kontynuacja — `expected=None`,
   NIE ma jednej "poprawnej" odpowiedzi).

Dla **każdego** promptu z jednoznaczną odpowiedzią, zapisz tę odpowiedź **z
góry**, zanim zobaczysz, co model wygeneruje — inaczej ocena nie jest
wiarygodna (łatwo "dopasować" kryteria po fakcie).

Struktura danych: lista słowników `{"id", "typ", "prompt", "expected"}`
(`expected=None` dla zadania jakościowego).

**Rozszerzenie R1 (opcjonalne):** dodaj 2 kolejne prompty (np. pytanie z
kontekstem, drugi styl kontynuacji) z pełnym traktowaniem jak w rdzeniu.

## 3. Chat template — Twoje zadanie

Zaimplementuj `format_prompt(prompt_text)`, która zwraca `inputs` gotowe do
`model.generate(**inputs, ...)`, używając `tokenizer.apply_chat_template()`
z `add_generation_prompt=True`. **Nie** dodawaj ręcznie żadnych tokenów
specjalnych — `apply_chat_template` już je wstawia (zobacz wykład, sekcja
"Chat templates").

**Uwaga API (zweryfikowana w wykładzie):** `apply_chat_template(...,
tokenize=True, return_tensors="pt")` **bez** `return_dict=True` może zwrócić
inny typ niż oczekujesz w Twojej wersji `transformers` — użyj
`return_dict=True` jawnie i odwołuj się przez `inputs["input_ids"]`, nie
zakładaj gołego tensora.

**Zadanie (kod):** zweryfikuj, że Twoja funkcja faktycznie dodaje tokeny
specjalne (sprawdź, że zdekodowany prompt zaczyna się od tokenu BOS/
`<|im_start|>`, nie od treści promptu).

## 4. Generowanie i poprawne dekodowanie — Twoje zadanie

Zaimplementuj `generate_response(prompt_text, **gen_kwargs)`:
1. sformatuj prompt (sekcja 3),
2. wywołaj `model.generate(**inputs, **gen_kwargs, pad_token_id=...)`,
3. **wyciągnij TYLKO nowo wygenerowane tokeny** (nie cały `output`, który
   zawiera też prompt — zobacz wykład, Eksperyment B/E),
4. zdekoduj (`skip_special_tokens=True`),
5. zwróć tekst **oraz** liczbę nowo wygenerowanych tokenów (potrzebne do
   pomiaru przepustowości w sekcji 6).

**Weryfikacja (gotowa asercja czeka w notebooku):** dla `do_sample=False`,
dwa wywołania `generate_response()` z tym samym promptem muszą dać
**identyczny** wynik — jeśli nie, masz błąd w obsłudze stanu/losowości, nie
w samym modelu.

## 5. Deterministyczne generowanie i sampling — Twoje zadanie

Dla **wszystkich 3 obowiązkowych** promptów (sekcja 2), wygeneruj odpowiedź
**greedy** (`do_sample=False`) przez `generate_response()`. Dodatkowo, dla
**podzbioru** — prompt jednoznaczny, który wybierzesz (np. ekstrakcja), **i**
prompt jakościowy — uruchom też **sampling** z dwiema temperaturami (0,7 i
1,3). **Nie** wymagamy pełnego iloczynu "3 prompty × wszystkie strategie" —
to nieproporcjonalne do dostępnego czasu.

Użyj **ustalonego ziarna** (`torch.manual_seed(...)`) przed każdym
wywołaniem samplingu, żeby wyniki były powtarzalne między Twoimi
przebiegami.

**Rozszerzenie R2 (opcjonalne):** uruchom sampling również dla drugiego
promptu jednoznacznego (i dla promptów z R1, jeśli je dodałaś/eś) — czy
sampling pomaga, szkodzi, czy nie ma znaczenia na zadaniach, którym rdzeń go
nie dał?

## 6. Pomiar czasu i przepustowości — Twoje zadanie

Dla każdej kombinacji (prompt, strategia), którą faktycznie uruchomiłaś/eś w
sekcji 5, zmierz **rzeczywisty** czas wywołania `generate_response()`
(`time.time()` przed/po) i policz **tokenów na sekundę** = liczba nowo
wygenerowanych tokenów / czas. Zbierz wyniki w jedną tabelę.

## 7. Wpływ długości promptu — Twoje zadanie

Wybierz **jeden** prompt i wygeneruj jego dłuższą wersję (np. dodaj
kontekst/historię). Zmierz czas generowania **tej samej liczby** nowych
tokenów dla krótkiej i długiej wersji promptu — użyj `min_new_tokens ==
max_new_tokens`, żeby wymusić tę samą liczbę nowych tokenów dla obu (inaczej
różnice w czasie generowania, np. wcześniejsze zatrzymanie przez EOS,
zanieczyszczą pomiar samego wpływu długości promptu — ta pułapka jest
realna, sprawdzona przy przygotowaniu tego laboratorium). Czy różnica jest
duża czy mała względem całkowitego czasu? (Zobacz wykład: prefill vs decode.)

## 8. KV cache — Twoje zadanie

Dla jednego promptu i jednej długości generowania, porównaj czas z
`use_cache=True` i `use_cache=False`. Czy przewaga cache rośnie, maleje, czy
jest stała z długością generowania? (Porównaj z wykładem — czy Twój wynik
jest zgodny?)

## 9. Ocena jakości — Twoje zadanie

1. **Automatycznie**, dla **obu** promptów z jednoznaczną odpowiedzią
   (sekcja 2), na ich odpowiedzi **greedy**: sprawdź
   `p["expected"].lower() in response.lower()`.
2. **Jakościowo**, dla promptu **jakościowego**: oceń **każdą** uruchomioną
   w sekcji 5 strategię (greedy + ewentualny sampling) pod względem:
   - **poprawności wykonania instrukcji** — czy model zrobił to, o co
     prosiliśmy?
   - **kompletności** — czy odpowiedź jest ucięta w połowie myśli?
   - **zgodności z wymaganym formatem** (jeśli prompt określał format),
   - **naturalności/spójności** odpowiedzi,
   - **powtarzalności** (dla promptów z `do_sample=True`, czy wielokrotne
     wywołanie z różnymi ziarnami dałoby podobny wynik?).

**Dla oceny jakościowej**: napisz **krótkie uzasadnienie oparte na
konkretnym fragmencie odpowiedzi** — nie "wygląda nieźle". Np. "parafraza
zachowała znaczenie ('fox'/'dog' nadal obecne), ale użyła dokładnie tych
samych 3 słów co oryginał w kluczowym miejscu — słaba parafraza" jest
konkretną oceną; "brzmi dobrze" nie jest.

❌ "Ta odpowiedź wygląda lepiej, więc jest lepsza."

✅ Rozdzielamy powyższe kryteria właśnie, żeby **nie** oceniać jednym,
subiektywnym wrażeniem (zobacz wykład).

**Rozszerzenie R1/R2 (opcjonalne):** jeśli wykonałaś/eś R1 lub R2, oceń je
tym samym schematem.

## 10. Zadanie diagnostyczne

Poniższy kod (w notebooku) **wygląda** poprawnie — nie rzuca wyjątku (dla
krótkiego promptu), generuje jakiś tekst — ale ma błąd: pomylenie
`max_length` z `max_new_tokens`.

**Zadanie:**
1. Uruchom kod z krótkim promptem i zaobserwuj: ile **nowych** tokenów
   faktycznie powstało, względem tego, ile **chciał** autor kodu (zmienna
   `DESIRED_NEW_TOKENS`)?
2. Zdiagnozuj przyczynę: co dokładnie liczy `max_length`, a co
   `max_new_tokens`? (Zobacz wykład, sekcja "Okno kontekstowe i
   obcinanie".)
3. Uruchom **ten sam** kod z **dłuższym** promptem (wystarczająco długim,
   żeby `prompt_len >= max_length`) — co się teraz dzieje? (Podpowiedź:
   inne, głośniejsze zachowanie niż przy krótkim prompcie — konkretnie,
   `transformers` rzuca `ValueError` wprost, zamiast po cichu zwrócić zbyt
   mało tokenów.)
4. Popraw kod (`max_new_tokens=` zamiast `max_length=`).
5. Zweryfikuj poprawkę: dla **obu** długości promptu, liczba nowo
   wygenerowanych tokenów musi być zgodna z zamierzoną.

## 11. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami/przykładami z sekcji 5-10:

1. Która strategia dekodowania dała najbardziej **użyteczne** wyniki dla
   Twoich dwóch zadań z jednoznaczną odpowiedzią? Czy sampling (tam, gdzie
   go przetestowałaś/eś) pomógł, zaszkodził, czy nie miał znaczenia?
2. Dla zadania jakościowego — czy wyższa temperatura dała **subiektywnie**
   lepsze wyniki? Czy różnorodność miała znaczenie dla **tego konkretnego**
   zadania?
3. Jaki jest zmierzony koszt (czas, tokeny/s) Twojego pipeline'u? Jak
   zmienia się z długością promptu i z `use_cache`?
4. Jakie konkretne ograniczenia modelu/pipeline'u zaobserwowałaś/eś (sekcja
   9)? Podaj **konkretny** przykład błędnej/niekompletnej/niezgodnej z
   formatem odpowiedzi, nie ogólnik.
5. Jaki jest Twój wniosek o kompromisie jakość/koszt/konfiguracja — czy
   "lepsza" konfiguracja (np. wyższa temperatura, większy budżet tokenów)
   zawsze oznacza lepszy wynik dla **każdego** typu zadania z sekcji 2?

## Rozszerzenia (opcjonalne — R1-R2, jeśli zostanie czas)

- **R1.** Dodaj 2 kolejne prompty (np. pytanie z kontekstem, inny styl
  kontynuacji) do listy z sekcji 2 i przeprowadź dla nich to samo
  traktowanie (sekcje 5, 6, 9) co dla promptów obowiązkowych.
- **R2.** Uruchom sampling (obie temperatury) również dla drugiego promptu
  jednoznacznego (i dla promptów z R1) — czy sampling pomaga, szkodzi, czy
  nie ma znaczenia dla zadań, którym rdzeń go nie dał?
- Dodaj **szósty** prompt własnego projektu, w innym stylu niż propozycje
  wyżej.
- Zmierz wpływ `repetition_penalty` na kontynuację tekstu (czy zmniejsza
  powtarzalność fraz przy dłuższym generowaniu?).

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---:|
| Sekcje 0-1 (konfiguracja, model) | 7 min |
| Sekcja 2 (przygotowanie promptów, 3 obowiązkowe) | 8 min |
| Sekcja 3 (chat template) | 10 min |
| Sekcja 4 (generate_response + weryfikacja) | 15 min |
| Sekcja 5 (greedy + sampling, zakres zredukowany) | 12 min |
| Sekcja 6 (pomiar czasu/przepustowości) | 8 min |
| Sekcje 7-8 (długość promptu, KV cache) | 10 min |
| Sekcja 9 (ocena jakości, zakres zredukowany) | 15 min |
| Sekcja 10 (zadanie diagnostyczne) | 12 min |
| Sekcja 11 (wnioski) | 8 min |
| **Rdzeń (sekcje 0-11)** | **105 min** |
| Bufor | 15 min |
| **Całość** | **ok. 120 min** |

Priorytet, jeśli czasu zabraknie nawet w rdzeniu: sekcje 2-5 (rdzeń
pipeline'u) i 9-10 (ocena i diagnostyka). Generowanie krótkich odpowiedzi
(≤40 tokenów) na tym modelu trwa pojedyncze sekundy na CPU — budżet czasowy
idzie w projektowanie i analizę, nie w czekanie na model (zweryfikowane:
pełne rozwiązanie wzorcowe, łącznie z ładowaniem modelu, 7 wywołaniami
`generate()` w sekcji 5 i całą diagnostyką, wykonuje się w kilkadziesiąt
sekund).

**Rozszerzenia R1-R2 są opcjonalne** i nie są wliczone do rdzenia.

## Kryteria oceny

Wszystkie poniższe kryteria dotyczą **rdzenia (sekcje 0-11)** — żadne nie
wymaga ukończenia rozszerzeń R1-R2.

- Dokładnie 3 obowiązkowe prompty, z jawnie zapisanym `expected` **przed**
  wygenerowaniem odpowiedzi, dla obu zadań z jednoznacznym wynikiem.
- `format_prompt`/`generate_response` poprawnie oddzielają prompt od nowo
  wygenerowanej odpowiedzi — asercja z sekcji 4 przechodzi.
- Chat template użyty poprawnie (`apply_chat_template`, bez ręcznie
  dodanych tokenów specjalnych).
- Greedy uruchomiony dla wszystkich 3 promptów; sampling (obie temperatury)
  uruchomiony dla co najmniej 2 z nich, w tym promptu jakościowego.
- Pomiar czasu rozdziela przynajmniej: krótki vs długi prompt (sekcja 7),
  `use_cache` True vs False (sekcja 8).
- Ocena jakości (sekcja 9) rozdziela automatyczne kontrole (dla zadań z
  jednoznacznym wynikiem) od jakościowej oceny z konkretnym uzasadnieniem
  — nie jedna, subiektywna etykieta na wszystko.
- Zadanie diagnostyczne wskazuje konkretną przyczynę (`max_length` liczy
  **całkowitą** długość, `max_new_tokens` liczy tylko nowe tokeny), nie
  ogólnik.
- Wnioski odwołują się do konkretnych liczb/przykładów z Twojego przebiegu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/11_llm_engineering/`) i nie jest częścią tego
repozytorium.
