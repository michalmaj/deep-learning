Language: **English** | [Polski](README.pl.md)

# Lab 10. A small Transformer for sequence classification

The estimated time is about 2 hours. The required core takes 100-105 minutes, with a 15-20 minute buffer. This lab uses scaled dot-product attention, multi-head attention, masking, positional encoding, and the Transformer block from [`lecture.en.md`](../../lectures/10_attention/lecture.en.md).

The task is harder than the lecture example. Sequences have variable lengths and contain real padding and random distractors.

Sections 0-10 form the required core. Sections R1-R4 are optional and do not affect the core grade.

## Task

Every sequence has a random length between 10 and 40. It contains tokens `A` and `B` at random positions and distractors elsewhere. The label indicates whether `A` appears before `B`.

Length is sampled independently of the label. The model should therefore not solve the task through a relationship between class and length.

## 0. Setup

Imports, device selection, the random seed, and `torch.set_num_threads(4)` are provided.

## 1. Data generator

The provided `make_dataset(n, seed)` returns `(X, lengths, y)`. Tensor `X` has shape `(n, MAX_LEN)` and is filled with `PAD_ID` beyond the real length. `lengths` contains the lengths, and `y` contains the labels.

**Task.** Check the following properties on the training set:

1. class balance, expected near 50/50,
2. mean length in both classes and the absence of a meaningful difference.

The second check rules out a simple length-based heuristic in place of the order of `A` and `B`.

## 2. Order-insensitive baseline

Build a classifier that averages token embeddings without attention. Padding must be excluded. The provided helper `masked_mean_pool(x, real_mask, lengths)` implements the principle introduced in module 9. It should be used in the appropriate place in `MeanPoolBaseline.forward`.

The provided loop trains the baseline. Its result will be the reference point in section 10.

## 3. Positional encoding

The function `sinusoidal_positional_encoding(seq_len, d_model)` is provided. Its output must be added at the appropriate point in the architecture from section 6.

## 4. Padding mask

Build `key_padding_mask` from `x` and `PAD_ID`. For `nn.MultiheadAttention`, `True` denotes padding that should be ignored. This is the opposite convention from the Boolean `attn_mask` in `F.scaled_dot_product_attention`.

A sequence without padding should have a mask containing only `False`. Reversing the convention leads to the problem examined in section 9.

## 5. Transformer block

`TinyTransformerBlock` should contain:

1. `nn.MultiheadAttention` with the mask from section 4,
2. a residual connection and LayerNorm,
3. an FFN built as Linear, ReLU, Linear,
4. a second residual connection and LayerNorm.

This is a Post-LN block. The architecture should remain visible, so it must not be replaced by a single `nn.TransformerEncoderLayer`. Q, K, and V projections are provided by `nn.MultiheadAttention`.

## 6. Aggregation and classifier

`Classifier` should combine `nn.Embedding`, optional positional encoding, `TinyTransformerBlock`, masked aggregation, and a classification layer.

Aggregation uses `masked_mean_pool`. `real_mask` is the negation of `key_padding_mask`, and the denominator should come from `lengths` rather than `MAX_LEN`.

A provided assertion compares the same sequence processed alone and in a batch with a longer sequence. The representations must be identical. A difference indicates a masking or aggregation error.

## 7. Training with and without positional encoding

The provided `train_eval(model, epochs, lr, batch_size, seed)` supports every configuration. Train `Classifier` with `use_pos_enc=True` and `use_pos_enc=False` under the same budget. Store the models as `model_with_pe` and `model_without_pe`.

`model_with_pe` is reused in section 8 and extensions R1 and R2. It does not need to be trained again.

A provided loop repeats both configurations for seeds 0, 1, and 2. The interpretation in section 10 should determine whether chance-level performance without PE is systematic.

## 8. Test evaluation

Evaluate `model_with_pe` on the test set for the first time in this section. Report accuracy and the length distribution of misclassified examples.

## 9. Diagnostic task

`BuggyClassifier` defines a mask and trains without raising an exception, but it uses the reversed mask convention.

The task requires:

1. identifying the faulty line and explaining the problem,
2. running training and inspecting the loss, with attention to sequences of length `MAX_LEN`,
3. explaining softmax behavior when no key is allowed,
4. correcting the mask convention,
5. using `torch.isfinite` to confirm a finite loss and comparing accuracy with the correct model.

For a sequence without padding, the reversed mask blocks every position. Softmax then receives a row of only $-\infty$ values, which produces `NaN` in the explicit implementation.

## 10. Conclusions

Answers should use results from sections 2 and 7-9.

1. Which elements mix information, add position, and reduce the sequence to one vector?
2. What is the accuracy gap between the baseline and Transformer with PE, and what does it show about order?
3. Is performance without PE stable across three seeds?
4. Does the section 6 assertion confirm correct padding exclusion?
5. What does the experiment show about attention, and what does it not prove? High accuracy alone does not prove understanding of the `A` and `B` relationship.
6. Which limitation prevents generalizing the result to other sequential tasks?

## Optional extensions

Every extension reuses `model_with_pe` without retraining.

- **R1.** Validation accuracy in at least three length ranges, such as $[10,19]$, $[20,29]$, and $[30,40]$.
- **R2.** Attention weights for a correct example and the query at position `A`. A large weight toward `B` does not prove understanding of the relationship.
- **R3.** A stack of two `TinyTransformerBlock` instances and a comparison of accuracy or stability.
- **R4.** A change from Post-LN to Pre-LN and a comparison of training stability.

## Estimated time allocation

| Block | Time |
|---|---:|
| Sections 0-1, setup, data, and balance | 8 min |
| Section 2, baseline | 8 min |
| Sections 3-4, positional encoding and mask | 12 min |
| Section 5, Transformer block | 20 min |
| Section 6, aggregation, classifier, and assertion | 13 min |
| Section 7, training and three seeds | 12 min |
| Section 8, test evaluation | 6 min |
| Section 9, diagnostics | 13 min |
| Section 10, conclusions | 8 min |
| **Core** | **100 min** |
| Buffer | 15-20 min |
| **Total** | **about 120 min** |

If time is limited, sections 4-6 and 9 have priority. The full solution, including three seeds and diagnostics, runs on a CPU in under one minute. Class time should be spent mainly on masking, aggregation, and analysis.

## Grading criteria

- Class balance and independence of length from the label are confirmed numerically.
- The baseline excludes padding from the average.
- `key_padding_mask` uses `True` to denote padding.
- The block explicitly contains attention, residual, LayerNorm, FFN, residual, and LayerNorm.
- Aggregation excludes padding, and the section 6 assertion passes.
- The comparison with and without PE covers multiple seeds.
- The test set is first used in section 8 and only for evaluation.
- The diagnosis identifies the reversed mask and a fully masked row for a sequence without padding.
- Conclusions refer to measured results and state the limits of the experiment.

A complete reference solution is stored locally by the instructor in `decisions/solutions/10_attention/` and is not part of this repository.
