Language: **English (default)** | [Polski](README.pl.md)

# Lab 08: a content-based image retrieval system

Time: about 2 hours. Builds on [`lectures/08_embeddings/lecture.en.md`](../../lectures/08_embeddings/lecture.en.md)
(embeddings, Euclidean distance, cosine similarity, L2 normalization,
k-NN), but uses **a different dataset and a different task** than the
lecture -- EuroSAT (satellite imagery, not Caltech101) and **retrieval**,
not classification or visualization. The product of this lab is a small,
working system: a query (image) -> a ranking of the most similar images
from a database.

## 0. Setup (provided)

Library imports, device, random seed.

## 1. Data: gallery and queries -- your task (partly) + provided

EuroSAT, 10 terrain classes. We split the data into two **disjoint** sets:

- **gallery** -- the database of images we "search" (40 images/class, 400
  total),
- **queries** -- separate images for which we look for similar ones in
  the gallery (10 images/class, 100 total, **different images from the
  gallery**).

**Task:** verify programmatically that the `gallery_indices` and
`query_indices` sets do not overlap (a provided assertion is waiting to
be finished). This is not a formality -- if a query image is also in the
gallery, the search engine "cheats": it will find itself with a
similarity of $1.0$, which artificially inflates every relevance metric.
This is the **first and most important methodological mistake** this lab
is meant to avoid.

## 2. Extracting embeddings -- your task

Load `resnet18` with `ResNet18_Weights.IMAGENET1K_V1` weights and extract
embeddings (the `avgpool` output, **not** `fc` -- see the lecture for why
logits are the wrong choice for similarity comparisons) for `gallery_imgs`
and `query_imgs`.

**Task:** set the model to a mode that **does not** compute gradients and
**does not** update BatchNorm statistics during extraction (two separate
mechanisms -- see module 7, *Common misconceptions* section:
`requires_grad` controls something different from `model.eval()`, and
both are different from `torch.no_grad()`/`torch.inference_mode()`). Use
**both** correct mechanisms at once -- getting the correct combination is
part of the task, not just the result.

Measure and record the extraction time.

### The validity of an in-memory "cache"

Extracting embeddings for the whole database ($500$ images) only takes a
few seconds here -- we compute it **once**, in a single cell, and reuse
it many times in later sections (we do not recompute the features from
scratch for every metric experiment).

**Task (Markdown, 2-3 sentences):** if the database were 1000x larger
(and extraction took hours, not seconds), you would save the embeddings
to disk, to avoid recomputing them every time you run the notebook. Under
which **four** conditions would such a saved cache stop being correct
(i.e. you would need to recompute it from scratch)? Be specific -- not
"when something changes."

## 3. L2 normalization -- your task

Implement `l2_normalize(X)`, which returns the embeddings divided by
their norm (with a safeguard against dividing by zero -- see the
lecture, the zero-vector section). Verify: after normalization, every
vector has a norm close to $1$ (a provided assertion).

## 4. Top-k search -- your task

Implement `search(query_embedding, gallery_embeddings, k)`, which returns
the indices of the $k$ most similar gallery images by **cosine
similarity**, sorted from the best. **Write this yourself** -- there is
no ready-made library function to call; it is the matrix multiplication
and sorting you already know from the lecture.

Verify your implementation on a small, hand-constructed example with a
precisely known result (provided in the notebook -- $6$ vectors in $2$D,
$2$ classes, the ranking result computed in advance) -- your function
must return exactly the expected order.

## 5. Precision@k -- your task

Define relevance: a result is considered relevant if its class matches
the query's class. **This is a simplification** -- label agreement is
not a complete measure of semantic similarity (two images of the same
terrain class can look very different, two images of different classes
can look similar) -- name this explicitly in your answer in section 9.

Implement `precision_at_k(query_embeddings, query_labels,
gallery_embeddings, gallery_labels, k)`, which returns the **average**
Precision@k over all queries: for each query, what fraction of its $k$
nearest neighbors share its label.

Verify on the same small example from section 4 -- the result must match
the manual computation.

Compute Precision@1, Precision@5, Precision@10 for the ResNet18
embeddings (L2-normalized) on the real data.

## 6. Baseline: pixel distance -- your task

Compute the same metrics (Precision@1/5/10) for **raw, flattened pixels**
instead of ResNet18 embeddings (the same gallery/queries set, the same
`search` and `precision_at_k` functions -- do not write new code, use
what you already have). Compare the results.

## 7. Effect of normalization and metric -- your task

Compare Precision@5 for **three** variants on the ResNet18 embeddings:
1. cosine similarity on L2-normalized embeddings (section 5),
2. a plain dot product on **unnormalized** embeddings,
3. Euclidean distance on unnormalized embeddings.

Use **the same** gallery and queries set for all three variants.

**Task (Markdown, 3-4 sentences):** which variant did best? Is the
difference large or negligible? How does this relate to what you learned
in the lecture about cosine similarity and vector norm?

## 8. Visualization and error analysis -- your task

Choose $2$-$3$ queries: at least one with **high** Precision@5 and at
least one with **low** Precision@5. Draw the query together with its
top-5 results (a provided function is waiting for the data -- images +
their labels).

**Task (Markdown, 3-5 sentences):** for the low-Precision@5 example --
look at the returned images. Do the wrong results look "sensibly
similar" (e.g. similar terrain texture, a confused class) or do they look
like random hits? Which EuroSAT classes are most often confused in the
search results (you can count this systematically across all queries,
not just the selected examples)?

## 9. Conclusions -- your task

Answer, justified with actual numbers from sections 5-8:

1. By how much do ResNet18 embeddings beat the raw-pixel baseline
   (specific Precision@k numbers)? Is this advantage surprising, given
   that EuroSAT is a domain distant from ImageNet (module 7, lab)?
2. Did the choice of metric/normalization (section 7) have a large effect
   on the result? If you had to build a retrieval system from scratch,
   would you normalize the embeddings? Why?
3. Recall the caveat from section 5: label agreement is a simplified
   relevance measure. Did you find, in section 8, an example where a
   result had a **different** label but looked sensibly similar (or the
   reverse: the same label but looked completely different)? What does
   that say about the limits of this metric?
4. What is at least one limitation of this experiment that prevents you
   from generalizing the conclusions to any arbitrary image-retrieval
   system?

## Time budget (approximate, not verified in class)

| Block | Time |
|---|---|
| Sections 0-1 (setup, gallery/queries) | 10 min |
| Section 2 (extraction + cache question) | 15 min |
| Section 3 (L2 normalization) | 10 min |
| Section 4 (top-k search + verification on an example) | 20 min |
| Section 5 (Precision@k + verification on an example) | 20 min |
| Section 6 (pixel baseline) | 10 min |
| Section 7 (effect of metric/normalization) | 15 min |
| Section 8 (visualization + error analysis) | 15 min |
| Section 9 (conclusions) | 10 min |

Sections 0-8 are the mandatory core. Extracting the whole database takes
a few seconds on CPU -- the time budget should go into implementing the
ranking/metrics and analyzing the results, not waiting for computation.

**Extension (optional, if time remains):** implement a real on-disk cache
(`torch.save`/`torch.load`) with an explicit validity tag (weights
version name, preprocessing representation, list of image indices, layer
name) -- consistent with your answer from section 2. Or: check the effect
of gallery size (e.g. half the images) on Precision@k.

## Grading criteria

- The gallery and queries are disjoint -- the assertion in section 1
  passes.
- Embedding extraction uses the correct combination of model mode and a
  gradient-disabling context.
- `l2_normalize`, `search`, and `precision_at_k` are implemented
  independently and give the correct result on the hand-constructed
  example from sections 4-5 (not just "look reasonable" on the real
  data).
- Precision@k is computed with an explicit denominator (number of
  queries) and an explicit definition of relevance (class agreement,
  named explicitly as a simplification).
- The comparison in sections 6-7 uses the same gallery/queries set for
  every variant.
- The conclusions in section 9 refer to specific numbers from your own
  run.

A full reference solution exists locally with the instructor
(`decisions/solutions/08_embeddings/`) and is not part of this
repository.
