Language: **English** | [Polski](README.pl.md)

# Lab 09. SMS classification with RNN, LSTM, and GRU

The estimated time is about 2 hours. This lab uses recurrence, hidden states, LSTM, GRU, and padding from [`lecture.en.md`](../../lectures/09_rnn/lecture.en.md). The experiments use the real SMS Spam Collection instead of synthetic data.

## 0. Setup

Imports, device selection, the random seed, and `torch.set_num_threads(4)` are provided.

## 1. SMS Spam Collection data

The dataset contains 5574 messages labeled `ham` or `spam`. It is downloaded as an archive of about 200 KB. Each line in the text file has the form `label\ttext`. Details are available in [`data/registry.yml`](../../data/registry.yml).

The provided cache writes downloaded and extracted data to temporary files. It verifies the SHA-256 of the archive and extracted member before atomically replacing the final files. An interrupted download therefore cannot replace a valid cache entry with a partial file.

**Task.** Count the `ham` and `spam` classes and determine whether the dataset is balanced. The result will guide metric selection in section 10.

## 2. Tokenization

Complete `tokenize(text)`. The simple tokenizer should convert text to lowercase and separate words from punctuation. The notebook contains an unfinished regular-expression pattern.

**Markdown task, 2 to 3 sentences.** Describe the limitations of this tokenization. Relevant cases include inflection, deliberately distorted words such as `free` and `freee`, punctuation attached to words, and the lack of subword tokenization introduced in module 11.

## 3. Train, validation, and test split

The provided split uses a 70/15/15 ratio and a fixed seed. Results remain comparable across experiments.

## 4. Vocabulary from training data

Build the vocabulary exclusively from `train_texts`. Including validation or test data would cause data leakage. This is the token-set equivalent of the normalization rule used in modules 05-07.

Two special tokens are required.

- `<PAD>` with identifier $0$ extends a batch to a common length and is not content.
- `<UNK>` with identifier $1$ replaces tokens outside the training vocabulary.

Tokens occurring fewer than `MIN_FREQ` times in training should be removed. This reduces vocabulary size and overfitting to rare words.

**Code task.** A token unseen during training must map to `<UNK>`. It must not raise an exception or disappear silently.

## 5. Encoding and variable-length batches

The provided `encode()` and `make_batch_tensors()` functions convert tokens to identifiers. They also create a padded batch and a tensor of real lengths called `lengths`, which is required for correct padding handling in section 7.

Check the tensor shape for two messages of different lengths and explain what happened to the shorter message.

The function `train_eval(model_cls, vocab_size, epochs, lr, seed, **model_kwargs)` is defined after this section. It tracks training and validation loss after each epoch and returns the trained model. Every later model uses the same function.

## 6. Order-insensitive baseline

Build a classifier that averages token embeddings without a recurrent layer. Padding must not contribute to the mean. A mask such as `x != PAD_ID` is required because padding would otherwise dilute shorter-message representations.

The provided `train_eval` call trains the baseline. Its result will be the reference point in section 13.

## 7. `nn.Embedding` with `nn.GRU` or `nn.LSTM`

The model should contain `nn.Embedding`, either `nn.GRU` or `nn.LSTM`, and a classification layer.

The full-sequence representation must be the state after the last real token, not the state at the last tensor position, which may contain padding. Apply `torch.nn.utils.rnn.pack_padded_sequence` before the recurrent layer. It supplies the real lengths and prevents the layer from processing padding.

A provided assertion compares the same text encoded on its own and in a batch with a longer message. Both representations must be identical. A failure usually indicates the problem examined again in section 11.

## 8. Classifier training

Train the classifier from section 7 with the provided `train_eval` call used earlier for the baseline.

## 9. Comparing at least two configurations

Compare at least two configurations. Options include GRU versus LSTM, different hidden-state dimensions, dropout, or different epoch counts. Justify the selection in 2 to 3 sentences. Every configuration must use the same data split and metrics.

## 10. Test-set evaluation

Select the better configuration using validation results. Only then should the test set be evaluated for the first time in this lab. Three results are required.

- accuracy,
- macro-F1,
- a confusion matrix.

Macro-F1 matters for an imbalanced dataset. A model that always predicts `ham` could achieve high accuracy while remaining useless.

## 11. Diagnostic task

The notebook code does not raise an exception and may train successfully, but it handles padding incorrectly.

The task requires:

1. identifying the exact bug and the line where it occurs,
2. explaining when it appears by comparing `batch_size=1` with a batch of different sequence lengths,
3. fixing the code,
4. checking different sequence lengths with the assertion from section 7.

## 12. Error analysis

For the model from section 10, print 5 to 8 misclassified messages with their true and predicted labels. Determine whether `ham` to `spam` or `spam` to `ham` errors are more common. A comparison with message length is optional.

## 13. Conclusions

Answers should use the measured values from sections 6-12.

1. By how much does GRU or LSTM outperform the order-insensitive baseline? Relate the size of the difference to the role of order in spam classification and compare it with experiment C from the lecture.
2. Which configuration from section 9 performed better on validation? Is the difference from one run large enough to support a strong conclusion?
3. Which limitation, such as dataset size, simple tokenization, or a single seed, prevents broad generalization?

## Estimated time allocation

| Block | Time |
|---|---|
| Sections 0-1, setup and exploration | 10 min |
| Section 2, tokenization and limitations | 10 min |
| Sections 3-5, vocabulary, PAD/UNK, and encoding | 10 min |
| Section 6, order-insensitive baseline | 15 min |
| Section 7, classifier and padding | 25 min |
| Section 8, training | 5 min |
| Section 9, configuration comparison | 10 min |
| Section 10, test evaluation | 10 min |
| Section 11, padding diagnostic | 10 min |
| Section 12, error analysis | 5 min |
| Section 13, conclusions | 10 min |

Sections 0-13 form the mandatory core. CPU training takes only a few seconds, so most of the time should be spent on correct sequence handling and analysis.

Optional extensions include comparing the lengths of correctly and incorrectly classified messages, using a bidirectional GRU or LSTM with `bidirectional=True`, and reporting separate accuracy for short messages below 10 tokens and long messages above 30 tokens.

## Grading criteria

- The vocabulary is built exclusively from `train_texts`, and `<UNK>` handles out-of-vocabulary tokens correctly.
- The classifier from section 7 uses `pack_padded_sequence` or an equivalent mechanism. The assertion that representation is independent of padding passes.
- The baseline from section 6 excludes padding from the average.
- Test evaluation includes accuracy, macro-F1, and a confusion matrix.
- The test set is used only after selecting a configuration from validation results.
- The section 11 diagnosis identifies a specific cause.
- Conclusions refer to values measured during the experiment.

A complete reference solution is stored locally by the instructor in `decisions/solutions/09_rnn/` and is not part of this repository.
