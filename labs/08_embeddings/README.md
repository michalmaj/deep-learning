Language: **English (default)** | [Polski](README.pl.md)

# Lab 08. A similar-image retrieval system

This lab takes about 2 hours and builds on
[`lectures/08_embeddings/lecture.en.md`](../../lectures/08_embeddings/lecture.en.md).
It covers embeddings, Euclidean distance, cosine similarity, L2 normalization, and
k-NN. The task is to build a small EuroSAT image-retrieval system. A query image
should produce a ranking of the most similar database items.

## 0. Setup (provided)

Library imports, device selection, and the random seed.

## 1. Gallery and queries (partly provided)

The EuroSAT data is divided into two disjoint sets.

- **gallery** contains 40 images from each class, or 400 total,
- **queries** contains 10 different images from each class, or 100 total.

**Task.** Confirm programmatically that `gallery_indices` and `query_indices` have no
common elements. If a query image also appears in the gallery, the search finds the
same image with similarity $1.0$ and artificially inflates the metrics.

## 2. Embedding extraction

Load `resnet18` with `ResNet18_Weights.IMAGENET1K_V1` and extract the `avgpool`
output for `gallery_imgs` and `query_imgs`. The `fc` logits are not used.

The model mode should prevent BatchNorm statistics from changing. A separate context
must disable gradient-graph construction. Correct use of both mechanisms is part of
the task. Measure the extraction time as well.

### Cache validity

Embeddings for all 500 images are calculated once and stored in memory for the
remaining sections.

**Markdown task, 2-3 sentences.** A database 1000 times larger would justify saving
embeddings to disk. Identify four specific changes that would invalidate such a cache
and require extraction to run again.

## 3. L2 normalization

`l2_normalize(X)` should divide each embedding by its norm and protect against
division by zero. A provided assertion checks that the resulting vector norms are
close to $1$.

## 4. Top-k search

Implement `search(query_embedding, gallery_embeddings, k)` to return the indices of
the $k$ gallery images with the largest cosine similarity, ordered from best to worst.
The implementation should use matrix multiplication and sorting without a ready-made
neighbor-search function.

A small example contains six 2D vectors from two classes and a known ranking. The
function must return exactly the expected order.

## 5. Precision@k

A result is treated as relevant when its class matches the query class. This is a
simplification of semantic similarity and should be acknowledged in the conclusions.

`precision_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels,
k)` should return average Precision@k across all queries. Verify it first on the
manual example, then calculate Precision@1, Precision@5, and Precision@10 for the
L2-normalized ResNet18 embeddings.

## 6. Pixel baseline

Calculate the same metrics for raw, flattened pixels. The comparison must use the
same gallery and queries and the existing `search` and `precision_at_k` functions.

## 7. Effect of normalization and metric

Compare Precision@5 for three variants.

1. Cosine similarity on L2-normalized embeddings.
2. Dot product on unnormalized embeddings.
3. Euclidean distance on unnormalized embeddings.

All variants must use identical gallery and query sets.

**Markdown task, 3-4 sentences.** Identify the best variant, the size of the
difference, and how the result relates to the role of vector norm described in the
lecture.

## 8. Visualization and error analysis

Select exactly two queries, one with high and one with low Precision@5. Display each
query together with its five highest-ranked results.

**Markdown task, 3-5 sentences.** For the weaker query, assess whether incorrect
results share terrain texture or class characteristics, or appear random. Name the
specific classes visible in the example.

## 9. Conclusions

Answers should cite results from sections 5-8.

1. By how much do ResNet18 embeddings outperform the pixel baseline according to
   Precision@k? Is the result surprising for a domain distant from ImageNet?
2. How much did the metric and normalization affect the result? Should a retrieval
   system built from scratch normalize its embeddings?
3. Did the analysis include a differently labeled but sensibly similar result, or a
   same-label result with a clearly different appearance? What does this show about
   the limits of class-based Precision@k?
4. Which experimental limitation prevents generalizing the result to arbitrary image
   retrieval systems?

## Approximate time budget

| Block | Time |
|---|---|
| Sections 0-1 (setup, gallery/queries) | 10 min |
| Section 2 (extraction + cache question) | 15 min |
| Section 3 (L2 normalization) | 10 min |
| Section 4 (top-k + verification) | 20 min |
| Section 5 (Precision@k + verification) | 20 min |
| Section 6 (pixel baseline) | 10 min |
| Section 7 (metric and normalization) | 15 min |
| Section 8 (visualization and errors) | 10 min |
| Section 9 (conclusions) | 10 min |

Sections 0-9 form the required core. Database extraction takes a few seconds on CPU,
leaving most of the time for ranking, metric implementation, and analysis.

**Optional extensions.** Possible additions include counting the most frequent class
error pairs, creating an on-disk cache with an explicit validity tag, or testing the
effect of gallery size.

## Grading criteria

- The gallery and queries are disjoint, and the section 1 assertion passes.
- Extraction uses the correct model mode and a no-gradient context.
- `l2_normalize`, `search`, and `precision_at_k` are implemented independently and
  pass the manual example.
- Precision@k has an explicit denominator and a clear class-based relevance rule.
- Every comparison uses the same gallery and queries.
- Conclusions cite specific measured results.

A full reference solution exists locally with the instructor in
`decisions/solutions/08_embeddings/` and is not part of this repository.
