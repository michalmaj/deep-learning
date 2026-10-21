Language: **English (default)** | [Polski](README.pl.md)

# Lab 09: SMS message classification (spam/ham) with RNN/LSTM/GRU

Time: about 2 hours. Builds on [`lectures/09_rnn/lecture.en.md`](../../lectures/09_rnn/lecture.en.md)
(recurrence, $h_t$, LSTM/GRU, padding), but uses **a different problem**
than the lecture -- real text classification (SMS Spam Collection), not
synthetic sequences.

## 0. Setup (provided)

Library imports, device, random seed, `torch.set_num_threads(4)`.

## 1. Data: SMS Spam Collection (provided + short task)

5574 SMS messages, labeled `ham`/`spam`, downloaded directly (a text file
with `label\ttext` per line, a ~200KB zip -- see
[`data/registry.yml`](../../data/registry.yml)). The provided cache helper
writes downloads and extraction results to temporary files, verifies the
archive SHA-256 and extracted member, and only then replaces the final
files atomically. A failed or interrupted operation therefore cannot turn
a valid cache entry into a partial file.

**Task:** count and print the class distribution (`ham`/`spam`). Is the
set balanced or not? Note this down -- it matters for the choice of
metrics in section 10.

## 2. Tokenization -- your task

Implement `tokenize(text)`: a simple split into tokens (words +
punctuation as separate tokens, lowercase). A provided regular-expression
pattern is waiting to be finished.

**Task (Markdown, 2-3 sentences):** what are the limitations of such
simple tokenization? Think about word forms (e.g. "free" vs. "freee" in
deliberately distorted spam), punctuation glued to words, and what you
will only learn about subword tokenization in module 11.

## 3. Train/val/test split (provided)

70/15/15, a fixed seed -- so results are comparable across your
experiments.

## 4. Vocabulary: only from the training set -- your task

Build a vocabulary **exclusively** from `train_texts` (never from
val/test -- exactly the same principle as the normalization in modules
05-07, just applied here to a set of tokens instead of numerical
statistics). Add two special tokens:

- `<PAD>` (id $0$) -- padding to a common length within a batch, **not**
  content,
- `<UNK>` (id $1$) -- the unknown token (a word outside the training
  vocabulary, e.g. a rare word seen only in val/test).

Drop tokens that occur in training fewer than `MIN_FREQ` times (shrinks
the vocabulary, reduces overfitting to rare words).

**Task (code):** verify that no token from the validation/test set that
did not occur in training causes an error (it should be mapped to
`<UNK>`, not raise an exception or be silently dropped).

## 5. Text -> sequence of identifiers and variable-length batches (provided)

The `encode()` and `make_batch_tensors()` functions turn a list of tokens
into identifiers and build a padded batch **together with a tensor of
the real lengths** `lengths` -- this tensor is critical for correctly
handling padding in section 7.

Check: two messages of different length in the same batch -- what is the
shape of the resulting tensor? What happened to the shorter message?

Right after this section, the notebook defines `train_eval(model_cls,
vocab_size, epochs, lr, seed, **model_kwargs)` (provided, used by every
model in the rest of the lab) -- it tracks the training and validation
loss after every epoch and returns the trained model.

## 6. An order-blind baseline -- your task

Build a classifier based on **averaging token embeddings** (with no
recurrent layer at all) -- exactly as in the lecture, experiment C, but
here the averaging must skip padding (see the lecture: padding is not
content). Hint: you need a mask `x != PAD_ID` when computing the mean,
otherwise padding artificially "dilutes" the representation of shorter
messages. Train it right away (`train_eval`, a provided call) -- this
number will be your reference point in section 13.

## 7. An `nn.Embedding` + `nn.GRU`/`nn.LSTM` classifier -- your task

Build a model: `nn.Embedding` -> `nn.GRU` (or `nn.LSTM`) -> a
classification layer.

**This is the single most important design decision in this lab:** the
representation of the whole sequence for classification must be the
state **after the last real token**, not after the last position of the
tensor (which may be padding). Use
`torch.nn.utils.rnn.pack_padded_sequence` before passing to the recurrent
layer -- this is the mechanism that tells PyTorch the real lengths, so
the layer does not process the padding at all.

Verify your implementation (a provided assertion is waiting): **the same
text**, encoded on its own and batched together with a longer message
(and so padded differently), must give an **identical** representation
in both cases. If the assertion does not pass -- you have exactly the
bug that is the topic of the diagnostic section (section 11) -- but here
we expect your implementation to already be correct.

## 8. Training the classifier from section 7 (the loop from section 5 -- a provided call)

Train your classifier from section 7 via `train_eval` (the same function
as in section 6 for the baseline).

## 9. Comparing >=2 configurations -- your task

Compare **at least two** configurations -- choose (and justify in 2-3
sentences why these specifically): GRU vs. LSTM, a different hidden-state
dimension, presence/absence of dropout, a different number of epochs.
Use **the same** data split and the same metrics for each.

## 10. Evaluation on the test set -- your task

Choose the better (by validation) configuration from section 9. Evaluate
it on the test set (**the first time** in this lab) using:

- accuracy,
- **macro-F1** (the set is imbalanced -- section 1 -- accuracy alone is
  misleading: a model that always predicts `ham` would have high
  accuracy and zero usefulness),
- a confusion matrix.

## 11. Diagnostic task

The following code (in the notebook) **looks** correct -- it raises no
exception, it trains, it gives some accuracy -- but it has a bug related
to padding handling.

**Task:**
1. Diagnose the bug -- exactly what it is, on which line.
2. Explain why it matters (when it shows up, and when it does not --
   hint: check for batch_size=1 and for a batch with sequences of
   different lengths).
3. Fix the code.
4. Check the behavior on sequences of different lengths (the provided
   assertion from section 7 should now pass).

## 12. Error analysis -- your task

For the model from section 10: print 5-8 misclassified messages (text +
true/predicted label). Do the errors concentrate on one side (more
`ham`->`spam` or `spam`->`ham`)? Comparing error rates by message length
is left as an optional extension.

## 13. Conclusions -- your task

Answer, justified with actual numbers from sections 6-12:

1. By how much does the sequential model (GRU/LSTM) beat the order-blind
   baseline? Is this difference large or modest -- and what does that say
   about how much this **specific** problem (spam classification) actually
   requires understanding word order, compared to the lecture's task
   (experiment C), which was specifically constructed so that order was the
   only decisive information?
2. Which of the two compared configurations (section 9) did better on
   validation? Is the difference large enough to trust on a single run?
3. What is at least one limitation of this experiment (data size, simple
   tokenization, a single seed) that prevents you from generalizing the
   conclusions?

## Time budget (approximate, not verified in class)

| Block | Time |
|---|---|
| Sections 0-1 (setup, exploration) | 10 min |
| Section 2 (tokenization + limitations) | 10 min |
| Sections 3-5 (vocabulary, PAD/UNK, provided encoding) | 10 min |
| Section 6 (order-blind baseline) | 15 min |
| Section 7 (GRU/LSTM classifier + correct padding) | 25 min |
| Section 8 (training) | 5 min |
| Section 9 (comparing >=2 configurations) | 10 min |
| Section 10 (test evaluation) | 10 min |
| Section 11 (focused padding diagnostic) | 10 min |
| Section 12 (focused error analysis) | 5 min |
| Section 13 (conclusions) | 10 min |

Sections 0-13 are the mandatory core. Training on this dataset takes a
few seconds on CPU -- the time budget should go into correctly
implementing sequence handling and analysis, not waiting for training.

**Extensions (optional, if time remains):** compare the average length or
error rate of correctly and incorrectly classified messages; try a
bidirectional GRU/LSTM (`bidirectional=True`); or report accuracy
separately for short (<10 tokens) and long (>30 tokens) messages.

## Grading criteria

- The vocabulary is built exclusively from `train_texts`; `<UNK>`
  correctly handles out-of-vocabulary tokens without an error.
- The classifier from section 7 uses `pack_padded_sequence` (or an
  equivalent correct mechanism) -- the "same text, same representation
  regardless of padding" assertion passes.
- The baseline from section 6 correctly skips padding when averaging (not
  only the recurrent classifier).
- Test evaluation (section 10) uses accuracy **and** macro-F1 **and** a
  confusion matrix, not accuracy alone.
- The test set appears only in section 10, after a configuration choice
  based on validation.
- The diagnostic task (section 11) points to a specific cause, not a
  generality.
- The conclusions refer to specific numbers from your own run.

A full reference solution exists locally with the instructor
(`decisions/solutions/09_rnn/`) and is not part of this repository.
