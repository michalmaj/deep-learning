Language: **English (default)** | [Polski](README.pl.md)

# Lab 10: a small Transformer for order-dependent sequence classification

Time: about 2 hours (**100-105 minutes of required core + 15-20 minutes of
buffer**). Builds on
[`lectures/10_attention/lecture.en.md`](../../lectures/10_attention/lecture.en.md)
(scaled dot-product attention, multi-head attention, masking, positional
encoding, the Transformer block), but on a **different, harder task** than
the lecture: sequences of **variable length** (real padding, which the
lecture's Experiment D did not require) with **distractors** (random tokens
carrying zero information).

**Sections 0-10 are the required core. Sections R1-R4 (at the end of the
notebook) are optional extensions** -- do them only if time remains after
finishing the core. No core grading criterion requires completing the
extensions.

## Task

Every sequence (random length from $[10,40]$) contains exactly two
distinguished tokens, `A` and `B`, at random positions, with the remaining
positions filled by random distractor tokens. **Label = whether `A` appears
before `B`.** Sequence length is drawn **independently** of the label
(verify this in section 1) -- the model cannot "cheat" the task with a
heuristic that ignores order (e.g. "shorter sequences = class 0").

## 0. Setup (given)

Library imports, device, random seed, `torch.set_num_threads(4)`.

## 1. Data: the task generator (given + a short task)

`make_dataset(n, seed)` (given) returns `(X, lengths, y)`: `X` is a tensor
`(n, MAX_LEN)` filled with `PAD_ID` beyond each sequence's real length,
`lengths` are the real lengths, `y` are the labels.

**Task:** verify on the training set:
1. class balance (about 50/50),
2. that mean sequence length **does not differ** meaningfully between classes
   (if it did, the model could "solve" the task by length alone, not by the
   order of `A`/`B` -- exactly the trap this lab's brief warns about).

## 2. A baseline without order -- your task

Build a classifier based on averaging token embeddings (no attention) --
it **must skip padding** when averaging. You are given a helper
`masked_mean_pool(x, real_mask, lengths)` -- exactly the same principle as in
module 9, section 6 (already mastered, we do not reimplement it from
scratch) -- you decide **where and how** to use it in
`MeanPoolBaseline.forward`. Train it right away (the training loop is
given) -- it is your reference point in section 10.

## 3. Positional encoding (given)

The `sinusoidal_positional_encoding(seq_len, d_model)` function from the
lecture -- we do not reimplement it (already fully verified there), only
**decide how and where** to attach it in the architecture (section 6).

## 4. Padding mask -- your task

Build a `key_padding_mask` from `x` and `PAD_ID`. **Critical:** you are
using `nn.MultiheadAttention`, so following the lecture (the "An important
implementation note" section), `key_padding_mask` must use the convention
**`True` = this is padding, ignore it** -- **not** `True` = allowed (this
convention is **opposite** to `F.scaled_dot_product_attention`). Mix up
these conventions and you will land squarely in the diagnostic task in
section 9 -- before you get there, verify your own mask: a sequence with no
padding should have a mask that is entirely `False`.

## 5. Transformer block -- your task

Build `TinyTransformerBlock`: self-attention (`nn.MultiheadAttention` -- the
Q/K/V projections do not need to be written by hand, this is already
verified in the lecture) **+ residual connection + LayerNorm + feed-forward
(Linear->ReLU->Linear) + a second residual connection + a second LayerNorm**
(Post-LN, as in `lecture.md`). **You** combine these components -- do not
hide the architecture behind a single call to `nn.TransformerEncoderLayer`.
Remember to pass the `key_padding_mask` from section 4 into `self.mha(...)`.

## 6. Aggregation and classifier -- your task

Build `Classifier`: `nn.Embedding` (+ positional encoding from section 3, if
`use_pos_enc=True`) -> the `TinyTransformerBlock` from section 5 ->
**aggregation to a single vector per sequence** (use `masked_mean_pool` from
section 2 -- the same principle: `real_mask` is the negation of
`key_padding_mask`, dividing by `lengths`, not by `MAX_LEN`) -> a
classification layer.

**Verification (a ready-made assertion is waiting in the notebook):** the
same sequence, processed **alone** and **batched** with another, longer
sequence (so padded differently), must give an **identical** representation
in both cases. If it does not pass -- you have a bug in the mask (section 4)
or in the aggregation (this section), before you even get to training.

## 7. Training (given) and comparing with vs. without positional encoding -- your task

`train_eval(model, epochs, lr, batch_size, seed)` is given (the same loop for
all configurations in the rest of the lab). Train **two** configurations of
your `Classifier` -- `use_pos_enc=True` and `use_pos_enc=False` -- with the
same training budget, and save them as `model_with_pe`/`model_without_pe`.
**`model_with_pe` is reused in section 8 and in extensions R1/R2 -- you do
not train it a second time.**

Also given is a short loop over **3 seeds** (0, 1, 2), running `train_eval`
for both variants -- training takes mere seconds on CPU, so running it 6x
takes less time than writing it from scratch would. **Your task is
interpreting** the result (section 10, question 3): is "without PE = chance
level" systematic, or is it a fluke of a single unlucky run?

## 8. Evaluation on the test set -- your task

Evaluate `model_with_pe` from section 7 (this is the **first time** in this
lab that you use the test set for **evaluation**, not for retraining):
accuracy + the length distribution of misclassified examples (do errors
concentrate at particular lengths?).

## 9. Diagnostic task

The code below (in the notebook, `BuggyClassifier`) **looks** correct --
it defines a mask, builds the block, trains without raising an exception --
but has a bug related to the **mask convention**, exactly the one section 4
warns about.

**Task:**
1. Diagnose the bug -- which line, and why.
2. Run training and observe what happens to the loss (hint: check whether the
   training set contains sequences **with no padding at all**, i.e. of
   length `MAX_LEN` -- and what happens to attention masking when **every**
   position of such a sequence ends up "blocked" by the inverted convention).
3. Explain mathematically, referring to the "Rows with all keys masked"
   section in `lecture.md`: what exactly happens to `softmax` when, for a
   given query, zero keys are allowed?
4. Fix the code (invert the mask convention).
5. Verify that after the fix the loss is always finite (`torch.isfinite`) and
   that the model trains to an accuracy close to the original, correct
   implementation.

## 10. Conclusions -- your task

Answer, justifying with real numbers from your own run (sections 2, 7-9):

1. Can you point by hand to which architectural elements (sections 5-6) are
   responsible for: (a) mixing information between positions, (b)
   introducing positional information, (c) reducing the sequence to a single
   vector?
2. How large is the accuracy gap between the baseline (section 2) and your
   Transformer **with** PE (section 7) -- and what does that say about
   whether the task actually requires order?
3. Is the "without PE = chance level" result stable across several seeds
   (section 7, given loop), or is it a one-run fluke?
4. Is the masking correct -- do you have proof (the assertion from section 6)
   that the model does not use information contained in the padding?
5. What exactly does (and does **not**) this experiment show about the
   attention mechanism? Does high accuracy by itself prove that the model
   "understands" the `A`/`B` order relation, or only that it effectively
   minimizes loss on this specific dataset?
6. What is at least one limitation of this experiment that prevents
   generalizing the conclusions to arbitrary sequential tasks?

## Extensions (optional -- R1-R4, if time remains)

All of them reuse `model_with_pe` from section 7 -- **without** retraining.

- **R1.** Validation-set accuracy in at least 3 length buckets (e.g.
  $[10,19]$, $[20,29]$, $[30,40]$) -- is accuracy stable, or does it drop for
  longer/shorter sequences?
- **R2.** For one correctly classified example: the attention weights
  (query = position of `A`, any head). **Do not interpret** a high weight
  toward `B` as "proof" that the model "understands" the order relation.
- **R3.** Stack **two** `TinyTransformerBlock`s one after another -- does
  this improve accuracy or stability on longer sequences?
- **R4.** Swap Post-LN for Pre-LN (norm before the sublayer, not after) and
  check whether anything changes in training stability.

## Time budget (indicative, not verified in a real class)

| Block | Time |
|---|---:|
| Sections 0-1 (setup, data, balance verification) | 8 min |
| Section 2 (baseline without order) | 8 min |
| Sections 3-4 (positional encoding -- review, padding mask) | 12 min |
| Section 5 (Transformer block) | 20 min |
| Section 6 (aggregation + classifier + verification) | 13 min |
| Section 7 (training, with/without PE comparison, multi-seed) | 12 min |
| Section 8 (test evaluation) | 6 min |
| Section 9 (diagnostic task) | 13 min |
| Section 10 (conclusions) | 8 min |
| **Core (sections 0-10)** | **100 min** |
| Buffer | 15-20 min |
| **Total** | **about 120 min** |

Priority if time runs short even within the core: sections 4-6 (the
architecture's core) and 9 (diagnostics) -- these are the elements with no
sensible shortened variant. Training takes mere seconds on CPU for a single
run (verified: the full model solution, including the 3-seed loop and the
diagnostics, executes in under a minute) -- the time budget should go into
correctly implementing masking/aggregation and into analysis, not into
waiting for training.

**Extensions R1-R4 are optional** and are not counted into the core -- see
the section above.

## Grading criteria

All of the following criteria concern the **core (sections 0-10)** -- none
requires completing extensions R1-R4.

- Class balance and the independence of sequence length from the label are
  verified numerically (section 1).
- The baseline (section 2) correctly skips padding when averaging.
- `key_padding_mask` (section 4) uses the correct convention for
  `nn.MultiheadAttention` (`True` = padding).
- The Transformer block (section 5) is built from visible components
  (attention + residual + LayerNorm + FFN + residual + LayerNorm) -- not a
  single call to `nn.TransformerEncoderLayer`.
- The aggregation (section 6) excludes padding -- the "same sequence, same
  result regardless of batch padding" assertion passes.
- The with/without PE comparison (section 7) is checked across several
  seeds (the given loop), not a single run.
- The test set is used only in section 8, for evaluation, not for
  retraining.
- The diagnostic task (section 9) points to a specific cause (inverted mask
  convention + a fully masked row for a sequence with no padding), not a
  generality.
- The conclusions refer to specific numbers from your own run and explicitly
  answer what the experiment does **not** prove, not only what it does.

A full model solution exists locally with the instructor
(`decisions/solutions/10_attention/`) and is not part of this repository.
