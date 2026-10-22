Język: [English](README.md) | **Polski**

# Laboratorium 11. Pipeline inferencji małego modelu językowego

Przewidywany czas wynosi około 2 godzin. Rdzeń obowiązkowy zajmuje 105 minut, a bufor 15 minut. Laboratorium korzysta z tokenizacji, strategii dekodowania, chat template i KV cache opisanych w [`lecture.md`](../../lectures/11_llm_engineering/lecture.md).

Sekcje 0-11 stanowią obowiązkowy rdzeń. Rozszerzenia R1 i R2 są opcjonalne.

Używany model to `HuggingFaceTB/SmolLM2-135M-Instruct`, dostępny na licencji Apache 2.0 zgodnie z [`data/registry.yml`](../../data/registry.yml). Model ma 135 milionów parametrów. Powinien radzić sobie z prostymi poleceniami po angielsku, lecz nie gwarantuje rozumowania wieloetapowego ani poprawności faktograficznej. Prompty są po angielsku, a instrukcje i odpowiedzi w laboratorium po polsku.

Budowany pipeline obejmuje prompt, tokenizację, model, generowanie, dekodowanie, pomiar i ocenę.

## 0. Konfiguracja

Importy, urządzenie, ziarno losowości i `torch.set_num_threads(4)` są gotowe.

## 1. Model i tokenizer

Model i tokenizer są ładowane z `revision` przypiętym do konkretnego commita.

**Zadanie.** Należy wypisać `model.config.num_attention_heads` oraz `model.config.num_key_value_heads`, porównać je i wyjaśnić wpływ różnicy na pamięć KV cache.

## 2. Przygotowanie promptów

Należy przygotować dokładnie trzy obowiązkowe prompty.

1. Jednoznaczna ekstrakcja informacji z krótkiego tekstu.
2. Drugie zadanie z jednoznacznym wynikiem, na przykład zmiana formatu daty.
3. Zadanie jakościowe, takie jak parafraza lub kontynuacja, z `expected=None`.

Dla dwóch zadań jednoznacznych oczekiwana odpowiedź musi zostać zapisana przed uruchomieniem modelu. Zapobiega to dopasowaniu kryterium do otrzymanego wyniku.

Wymagana struktura to lista słowników `{"id", "typ", "prompt", "expected"}`.

Opcjonalne R1 dodaje dwa prompty oceniane tym samym sposobem.

## 3. Chat template

Funkcja `format_prompt(prompt_text)` powinna zwracać wejście gotowe do `model.generate`. Musi korzystać z `tokenizer.apply_chat_template()` oraz `add_generation_prompt=True`.

Tokenów specjalnych nie należy dodawać ręcznie. Wywołanie powinno jawnie ustawiać `return_dict=True`, ponieważ bez niego typ wyniku może zależeć od wersji `transformers`. Identyfikatory są dostępne przez `inputs["input_ids"]`.

**Zadanie w kodzie.** Zdekodowany prompt powinien zaczynać się od BOS lub `<|im_start|>`, a nie od treści użytkownika.

## 4. Generowanie i dekodowanie

Funkcja `generate_response(prompt_text, **gen_kwargs)` powinna:

1. sformatować prompt,
2. wywołać `model.generate(**inputs, **gen_kwargs, pad_token_id=...)`,
3. wybrać tylko nowo wygenerowane tokeny,
4. zdekodować je z `skip_special_tokens=True`,
5. zwrócić tekst i liczbę nowych tokenów.

Gotowa asercja wymaga identycznego wyniku dwóch wywołań z tym samym promptem przy `do_sample=False`.

## 5. Greedy i sampling

Dla wszystkich trzech promptów należy wygenerować odpowiedź greedy z `do_sample=False`. Sampling z temperature 0,7 i 1,3 jest wymagany dla jednego wybranego promptu jednoznacznego oraz promptu jakościowego.

Przed każdym samplingiem należy ustawić ziarno za pomocą `torch.manual_seed(...)`. Pełny iloczyn wszystkich promptów i strategii nie jest wymagany.

Opcjonalne R2 rozszerza sampling na drugi prompt jednoznaczny oraz prompty z R1.

## 6. Czas i przepustowość

Dla każdej uruchomionej pary prompt i strategia należy zmierzyć rzeczywisty czas `generate_response()` oraz obliczyć liczbę nowych tokenów na sekundę. Wyniki powinny znaleźć się w jednej tabeli.

## 7. Długość promptu

Należy przygotować krótką i długą wersję jednego promptu. Obie mają generować tę samą liczbę nowych tokenów, wymuszoną przez `min_new_tokens == max_new_tokens`. Bez tego EOS mógłby zniekształcić pomiar wpływu długości promptu.

Wynik powinien zostać odniesiony do różnicy między prefill i decode.

## 8. KV cache

Dla jednego promptu i wspólnej długości generowania należy porównać `use_cache=True` oraz `use_cache=False`. Analiza powinna określić, jak przewaga cache zmienia się wraz z długością odpowiedzi.

## 9. Ocena jakości

Dla obu promptów jednoznacznych odpowiedź greedy należy sprawdzić automatycznie:

```python
p["expected"].lower() in response.lower()
```

Dla promptu jakościowego każda uruchomiona strategia wymaga oceny:

- wykonania instrukcji,
- kompletności,
- zgodności z wymaganym formatem,
- naturalności i spójności,
- powtarzalności przy samplingu.

Uzasadnienie powinno wskazywać konkretny fragment odpowiedzi. Ogólne stwierdzenie, że tekst wygląda dobrze, nie jest wystarczającym kryterium.

## 10. Zadanie diagnostyczne

Kod w notebooku myli `max_length` z `max_new_tokens`. Dla krótkiego promptu generuje tekst, ale nie realizuje zakładanego budżetu nowych tokenów.

Należy:

1. porównać rzeczywistą liczbę nowych tokenów z `DESIRED_NEW_TOKENS`,
2. wyjaśnić zakres `max_length` i `max_new_tokens`,
3. powtórzyć test z promptem, dla którego `prompt_len >= max_length`, i zaobserwować `ValueError`,
4. zastąpić `max_length` przez `max_new_tokens`,
5. potwierdzić poprawną liczbę tokenów dla obu długości promptu.

## 11. Wnioski

Odpowiedzi powinny odwoływać się do wyników z sekcji 5-10.

1. Która strategia była najbardziej użyteczna dla zadań jednoznacznych i jaki wpływ miał sampling?
2. Czy wyższa temperature poprawiła wynik zadania jakościowego i czy różnorodność była przydatna?
3. Jaki był czas i throughput oraz ich zależność od długości promptu i cache?
4. Jakie konkretne błędy, braki lub naruszenia formatu wystąpiły?
5. Jak wygląda kompromis między jakością, kosztem i konfiguracją? Większy budżet lub temperature nie muszą pomagać każdemu zadaniu.

## Rozszerzenia opcjonalne

- **R1.** Dwa dodatkowe prompty z pełnym pomiarem i oceną.
- **R2.** Sampling obu temperatur dla drugiego promptu jednoznacznego oraz promptów z R1.
- Szósty prompt w innym stylu.
- Pomiar wpływu `repetition_penalty` na powtarzalność dłuższej odpowiedzi.

## Orientacyjny podział czasu

| Blok | Czas |
|---|---:|
| Sekcje 0-1, konfiguracja i model | 7 min |
| Sekcja 2, trzy prompty | 8 min |
| Sekcja 3, chat template | 10 min |
| Sekcja 4, `generate_response` i asercja | 15 min |
| Sekcja 5, greedy i sampling | 12 min |
| Sekcja 6, czas i throughput | 8 min |
| Sekcje 7-8, długość i cache | 10 min |
| Sekcja 9, ocena jakości | 15 min |
| Sekcja 10, diagnostyka | 12 min |
| Sekcja 11, wnioski | 8 min |
| **Rdzeń** | **105 min** |
| Bufor | 15 min |
| **Całość** | **około 120 min** |

Przy ograniczonym czasie priorytet mają sekcje 2-5 oraz 9-10. Krótkie odpowiedzi do 40 tokenów powstają na CPU w pojedyncze sekundy, a pełne rozwiązanie wykonuje się w kilkadziesiąt sekund.

## Kryteria oceny

- Istnieją dokładnie trzy obowiązkowe prompty, a `expected` dla obu zadań jednoznacznych zapisano przed generowaniem.
- `format_prompt` i `generate_response` oddzielają prompt od nowej odpowiedzi, a asercja przechodzi.
- Chat template jest używany bez ręcznego dodawania tokenów specjalnych.
- Greedy obejmuje trzy prompty, a obie wartości temperature co najmniej dwa, w tym jakościowy.
- Pomiary porównują krótki i długi prompt oraz cache włączony i wyłączony.
- Ocena oddziela automatyczne testy od jakościowego uzasadnienia opartego na fragmencie odpowiedzi.
- Diagnoza poprawnie rozróżnia całkowite `max_length` od nowych `max_new_tokens`.
- Wnioski korzystają z konkretnych wyników przebiegu.

Pełne rozwiązanie wzorcowe znajduje się lokalnie u prowadzącego w `decisions/solutions/11_llm_engineering/` i nie jest częścią repozytorium.
