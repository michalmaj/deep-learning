# Laboratorium 11: projektowanie i ocena niewielkiego pipeline'u inferencji LLM

Czas: ok. 2 godziny. Bazuje na
[`lectures/11_llm_engineering/lecture.md`](../../lectures/11_llm_engineering/lecture.md)
(tokenizacja, next-token prediction, strategie dekodowania, chat templates,
KV cache), ale to **inżynierski eksperyment**, nie kolejne zadanie
klasyfikacyjne: budujesz i oceniasz mały, powtarzalny pipeline inferencji.

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

Wybierz **3-5** promptów reprezentujących różne zadania — proponowane typy
(możesz dopasować treść, zachowaj rozróżnienie **automatycznie
sprawdzalne** vs **jakościowe**):

1. **Ekstrakcja informacji** z krótkiego tekstu (ma jednoznaczną oczekiwaną
   odpowiedź — automatycznie sprawdzalna, np. przez `in` albo regex).
2. **Przekształcenie formatu** (np. data w innym formacie — jednoznaczna,
   automatycznie sprawdzalna).
3. **Parafraza** (NIE ma jednej "poprawnej" odpowiedzi — jakościowa ocena).
4. **Pytanie wymagające odpowiedzi na podstawie podanego kontekstu**
   (jednoznaczna, automatycznie sprawdzalna, jeśli odpowiedź jest w
   kontekście).
5. **Kontynuacja krótkiego tekstu** (jakościowa — oceniasz spójność/
   gramatyczność, nie dokładne słowa).

Dla **każdego** promptu, który ma jednoznaczną odpowiedź, zapisz tę
odpowiedź **z góry**, zanim zobaczysz, co model wygeneruje — inaczej ocena
nie jest wiarygodna (łatwo "dopasować" kryteria po fakcie).

Struktura danych: lista słowników `{"id", "typ", "prompt", "expected"}`
(`expected=None` dla zadań jakościowych).

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

Dla **każdego** promptu z sekcji 2, wygeneruj odpowiedź przez
`generate_response()`:
1. greedy (`do_sample=False`),
2. sampling z umiarkowaną temperaturą,
3. sampling z wyższą temperaturą (lub top-k/top-p — wybierz **co najmniej
   jedną** dodatkową strategię poza greedy i jedną temperaturą).

Użyj **ustalonego ziarna** (`torch.manual_seed(...)`) przed każdym
wywołaniem samplingu, żeby wyniki były powtarzalne między Twoimi
przebiegami.

## 6. Pomiar czasu i przepustowości — Twoje zadanie

Dla każdej kombinacji (prompt, strategia) z sekcji 5, zmierz **rzeczywisty**
czas wywołania `generate_response()` (`time.time()` przed/po) i policz
**tokenów na sekundę** = liczba nowo wygenerowanych tokenów / czas. Zbierz
wyniki w jedną tabelę (lista słowników albo `pandas.DataFrame`, jeśli wolisz).

## 7. Wpływ długości promptu — Twoje zadanie

Wybierz **jeden** prompt i wygeneruj jego dłuższą wersję (np. dodaj
kontekst/historię). Zmierz czas generowania **tej samej liczby** nowych
tokenów dla krótkiej i długiej wersji promptu. Czy różnica jest duża czy
mała względem całkowitego czasu? (Zobacz wykład: prefill vs decode.)

## 8. KV cache — Twoje zadanie

Dla jednego promptu i jednej długości generowania, porównaj czas z
`use_cache=True` i `use_cache=False`. Czy przewaga cache rośnie, maleje, czy
jest stała z długością generowania? (Porównaj z wykładem — czy Twój wynik
jest zgodny?)

## 9. Ocena jakości — Twoje zadanie

Dla **każdej** odpowiedzi wygenerowanej w sekcji 5 (wybierz jedną strategię
dekodowania na prompt, jeśli to dużo pracy — ale uzasadnij wybór), oceń:

1. **Poprawność wykonania instrukcji** — czy model zrobił to, o co
   prosiliśmy? (Dla promptów z `expected` — sprawdzenie automatyczne, np.
   `expected.lower() in response.lower()`.)
2. **Poprawność faktograficzna** — tam, gdzie da się zweryfikować (np.
   odpowiedź na pytanie z kontekstem — czy fakt w odpowiedzi zgadza się z
   podanym kontekstem?).
3. **Kompletność** — czy odpowiedź jest ucięta w połowie myśli (częste przy
   małym `max_new_tokens`)?
4. **Zgodność z wymaganym formatem** (jeśli prompt określał format).
5. **Powtarzalność** — dla promptów z `do_sample=True`, czy **wielokrotne**
   wywołanie (różne ziarna) daje konsekwentnie podobny wynik, czy
   kompletnie inny za każdym razem?

**Dla ocen jakościowych (parafraza, kontynuacja)**: napisz **krótkie
uzasadnienie oparte na konkretnym fragmencie odpowiedzi** — nie "wygląda
nieźle". Np. "parafraza zachowała znaczenie ('fox'/'dog' nadal obecne), ale
użyła dokładnie tych samych 3 słów co oryginał w kluczowym miejscu — słaba
parafraza" jest konkretną oceną; "brzmi dobrze" nie jest.

❌ "Ta odpowiedź wygląda lepiej, więc jest lepsza."

✅ Rozdzielamy powyższe pięć kryteriów właśnie, żeby **nie** oceniać jednym,
subiektywnym wrażeniem (zobacz wykład).

## 10. Zadanie diagnostyczne

Poniższy kod (w notebooku) **wygląda** poprawnie — nie rzuca wyjątku (dla
niektórych promptów), generuje jakiś tekst — ale ma błąd: pomylenie
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
   inne, głośniejsze zachowanie niż przy krótkim prompcie.)
4. Popraw kod (`max_new_tokens=` zamiast `max_length=`).
5. Zweryfikuj poprawkę: dla **obu** długości promptu, liczba nowo
   wygenerowanych tokenów musi być zgodna z zamierzoną.

## 11. Wnioski — Twoje zadanie

Odpowiedz, uzasadniając rzeczywistymi liczbami/przykładami z sekcji 5-10:

1. Która strategia dekodowania dała najbardziej **użyteczne** wyniki dla
   zadań z jednoznaczną odpowiedzią (ekstrakcja/format/kontekst)? Czy
   sampling pomógł, zaszkodził, czy nie miał znaczenia dla **tych
   konkretnych** zadań?
2. Dla zadań jakościowych (parafraza/kontynuacja) — czy wyższa temperatura
   dała **subiektywnie** lepsze wyniki? Czy różnorodność (sekcja 5) była
   duża czy mała — i czy to zależało od **zadania**, nie tylko od
   parametrów?
3. Jaki jest zmierzony koszt (czas, tokeny/s) Twojego pipeline'u? Jak
   zmienia się z długością promptu i z `use_cache`?
4. Jakie konkretne ograniczenia modelu/pipeline'u zaobserwowałaś/eś (sekcja
   9)? Podaj **konkretny** przykład błędnej/niekompletnej/niezgodnej z
   formatem odpowiedzi, nie ogólnik.
5. Jaki jest Twój wniosek o kompromisie jakość/koszt/konfiguracja — czy
   "lepsza" konfiguracja (np. wyższa temperatura, większy budżet tokenów)
   zawsze oznacza lepszy wynik dla **każdego** typu zadania z sekcji 2?

## Podział czasu (orientacyjny, nie zweryfikowany na zajęciach)

| Blok | Czas |
|---|---|
| Sekcje 0-1 (konfiguracja, model) | 10 min |
| Sekcja 2 (przygotowanie promptów) | 10 min |
| Sekcja 3 (chat template) | 10 min |
| Sekcja 4 (generate_response + weryfikacja) | 15 min |
| Sekcja 5 (strategie dekodowania × prompty) | 15 min |
| Sekcja 6 (pomiar czasu/przepustowości) | 10 min |
| Sekcja 7 (wpływ długości promptu) | 10 min |
| Sekcja 8 (KV cache) | 10 min |
| Sekcja 9 (ocena jakości) | 20 min |
| Sekcja 10 (zadanie diagnostyczne) | 15 min |
| Sekcja 11 (wnioski) | 10 min |

Suma: 135 min — priorytet mają sekcje 2-5 i 9-10 (rdzeń pipeline'u i
diagnostyka); sekcje 7-8 mogą zostać skrócone (jeden pomiar zamiast kilku),
jeśli czasu zabraknie. Generowanie krótkich odpowiedzi (≤60 tokenów) na tym
modelu trwa sekundy na CPU — budżet czasowy idzie w projektowanie i analizę,
nie w czekanie na model.

**Rozszerzenie (opcjonalne, jeśli zostanie czas):** dodaj **szósty** prompt
własnego projektu, w innym stylu niż propozycje wyżej. Albo: zmierz wpływ
`repetition_penalty` na kontynuację tekstu (czy zmniejsza powtarzalność
fraz przy dłuższym generowaniu?).

## Kryteria oceny

- Co najmniej 3 prompty, z jawnie zapisanym `expected` **przed**
  wygenerowaniem odpowiedzi, dla zadań z jednoznacznym wynikiem.
- `format_prompt`/`generate_response` poprawnie oddzielają prompt od nowo
  wygenerowanej odpowiedzi — asercja z sekcji 4 przechodzi.
- Chat template użyty poprawnie (`apply_chat_template`, bez ręcznie
  dodanych tokenów specjalnych).
- Pomiar czasu rozdziela przynajmniej: krótki vs długi prompt (sekcja 7),
  `use_cache` True vs False (sekcja 8).
- Ocena jakości (sekcja 9) rozdziela automatyczne kontrole (dla zadań z
  jednoznacznym wynikiem) od jakościowych ocen z konkretnym uzasadnieniem
  — nie jedna, subiektywna etykieta na wszystko.
- Zadanie diagnostyczne wskazuje konkretną przyczynę (`max_length` liczy
  **całkowitą** długość, `max_new_tokens` liczy tylko nowe tokeny), nie
  ogólnik.
- Wnioski odwołują się do konkretnych liczb/przykładów z Twojego przebiegu.

Pełne rozwiązanie wzorcowe istnieje lokalnie u prowadzącego
(`decisions/solutions/11_llm_engineering/`) i nie jest częścią tego
repozytorium.
