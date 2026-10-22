Language: **English (default)** | [Polski](lecture.md)

# Lecture 08. Embeddings and feature-space geometry

## Learning goals

1. explain how an embedding differs from raw data, hand-designed features, and logits,
2. calculate and interpret Euclidean distance and cosine similarity,
3. describe how L2 normalization changes representation geometry,
4. test empirically whether a learned neighborhood is useful,
5. recognize the limitations of PCA, t-SNE, and UMAP visualizations,
6. present `nn.Embedding` as a trainable vector table for identifiers.

## Why this matters

In module 7, a CNN backbone served as a feature extractor. Its output entered a new head, but the geometry of that vector was not analyzed. The central question is now whether distance between representations corresponds to similarity between objects.

Useful geometry supports similar-image search, recommendations, and semantic search. Numerical proximity does not always have the expected meaning, so it must be evaluated on data.

## Intuition

A representation records an object, such as an image, word, or user, in a form suitable for computation. Raw pixels also form a vector, but the data format determines its structure. An embedding is learned. A network adapts its dimensions to the objective minimized during training.

❌ Every numerical vector automatically provides a useful interpretation of similarity.

✅ Distance always has geometric meaning, but its semantic meaning depends on how the vector was created. Raw pixels and CNN features are both sequences of numbers, yet they can produce very different neighborhoods. The experiment compares them without assuming that the embedding is superior.

## Formal model

An embedding maps an object to a vector with a fixed number of dimensions.

$$\mathbf e = f_\theta(x), \qquad x \in \mathcal X,\quad \mathbf e \in \mathbb R^d$$

$\mathcal X$ denotes the input space, such as $3\times224\times224$ images, sentences, or product identifiers. $\mathbb R^d$ is the representation space, and $\theta$ denotes the mapping parameters. The ResNet18 `avgpool` output has $d=512$.

The bold symbol $\mathbf e$ denotes an embedding vector. The symbol $z$ remains reserved for logits, while plain $e$ can denote Euler's number or an epoch index. Context and typography separate these meanings.

### Embeddings from a CNN

In ResNet18, the embedding is the global-pooling output before the `fc` layer.

| Object | Source | Dimension for 10 classes | Use |
|---|---|---:|---|
| Embedding $\mathbf e$ | `avgpool` output | $512$ | similarity and search |
| Logits $z$ | `fc` output before softmax | $10$ | loss-function input |
| Probabilities $\hat y=\text{softmax}(z)$ | `softmax(fc(...))` | $10$ | prediction interpretation |

❌ Logits can replace the embedding because they are only a later stage of the same model.

✅ Logits are tied to the classes of a specific task. Their dimension and meaning change with the head. The `avgpool` embedding is more general and independent of a particular classification layer. This is why module 7 replaced `fc` rather than `avgpool`.

## Similarity and distance

### Euclidean distance

$$d(\mathbf a,\mathbf b) = \|\mathbf a - \mathbf b\|_2 = \sqrt{\textstyle\sum_i (a_i-b_i)^2}$$

### Dot product and cosine similarity

$$\mathbf a \cdot \mathbf b = \textstyle\sum_i a_i b_i, \qquad \cos(\mathbf a,\mathbf b) = \frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|_2\,\|\mathbf b\|_2} \in [-1,1]$$

Cosine similarity measures the angle and ignores vector length. Euclidean distance uses both direction and length. The metrics therefore answer different questions and can rank neighbors differently.

### L2 normalization

A normalized embedding is $\hat{\mathbf e} = \mathbf e / \|\mathbf e\|_2$ and has norm 1. For unit vectors,

$$\|\hat{\mathbf a}-\hat{\mathbf b}\|_2^2 = 2 - 2\cos(\hat{\mathbf a},\hat{\mathbf b}).$$

The notebook confirms this relation with error around $10^{-6}$. After normalization, ranking by Euclidean distance is identical to ranking by cosine similarity because both are monotonic transformations of the same value. Without normalization, the rankings can differ.

❌ L2 normalization always improves search quality.

✅ Normalization removes the effect of vector length. It helps only when the norm does not carry useful information. The effect should be measured.

### The zero vector

For $\mathbf e=\mathbf 0$, the norm is zero, so $\mathbf e/\|\mathbf e\|_2$ is undefined. PyTorch returns `nan` in this case. A stable implementation adds a small $\varepsilon$ to the denominator, for example `e / (e.norm() + eps)`.

### k nearest neighbors

Given a query $\mathbf q$ and vectors $\{\mathbf e_1,\ldots,\mathbf e_N\}$, k-NN selects the $k$ smallest distances or largest similarities. Here $k$ denotes the number of neighbors rather than the convolutional filter size from module 6.

## Example of different rankings

Consider query $\mathbf q=(1,0)$ and candidates $\mathbf a=(10,0)$ and $\mathbf b=(0.9,0.9)$.

$$d(\mathbf q,\mathbf a) = 9.0 \qquad d(\mathbf q,\mathbf b) \approx 0.906$$
$$\cos(\mathbf q,\mathbf a) = 1.0 \qquad \cos(\mathbf q,\mathbf b) \approx 0.707$$

Euclidean distance selects $\mathbf b$, while cosine similarity selects $\mathbf a$. Vector $\mathbf a$ points in the same direction as the query but has ten times its norm. Scale therefore dominates Euclidean distance and does not affect cosine similarity.

## Dimensionality reduction and PCA

ResNet18 embeddings have 512 dimensions and cannot be plotted directly. PCA finds mutually orthogonal directions $\mathbf u_1,\mathbf u_2,\ldots$ along which the data has maximum variance. The first direction captures the largest spread, and each following direction captures the largest remaining spread after earlier directions are considered.

Projection onto the first two directions is

$$\mathbf e' = (\mathbf u_1^\top \mathbf e,\ \mathbf u_2^\top \mathbf e) \in \mathbb R^2.$$

The notebook calculates the covariance matrix and its eigendecomposition with `torch.linalg.eigh`. The two eigenvectors with the largest eigenvalues define the plot. Low-variance directions are discarded together with any information encoded in them.

❌ A 2D PCA plot shows class separability in the full space.

✅ PCA maximizes variance and does not use labels. Good 2D separation confirms that some structure exists. Poor separation does not prove that classes overlap in all 512 dimensions.

t-SNE and UMAP are nonlinear visualization methods that often create clearer clusters but distort global geometry.

❌ Cluster sizes and distances in a t-SNE plot correspond to the original space.

✅ t-SNE primarily preserves local neighborhoods. Cluster size and distance between separate clusters are not reliable measures of density or difference in the input space.

## Representations learned from relations

The experiment uses embeddings from an ImageNet classifier. A useful neighborhood is a side effect of the classification objective. *Metric learning* and *contrastive learning* optimize geometry directly. Similar pairs or triplets are pulled together, while different ones are pushed apart. The training objective therefore shapes both classification behavior and the arrangement of the learned space.

## `nn.Embedding` for discrete identifiers

An image passes through a network to produce an embedding. A word, token, or user identifier instead indexes a table that stores a separate vector for each identifier.

```python
import torch.nn as nn

V, d = 6, 4
embedding = nn.Embedding(V, d)
token_ids = torch.tensor([0, 3, 1])
vectors = embedding(token_ids)  # shape (3, 4)
```

`nn.Embedding(V, d)` stores a trainable $V\times d$ matrix in `embedding.weight`. Its forward operation reads selected rows rather than performing matrix multiplication. The gradient updates only rows used in the batch. The notebook confirms that all other rows remain unchanged.

❌ Newly created token embeddings already have semantic structure.

✅ Initial vectors are random. Relationships emerge only through training toward a specific objective, such as neighboring-word prediction or translation. Modules 9 and 10 develop this topic for sequences.

## Engineering implications

- **Extraction.** One backbone forward pass produces an embedding. The result can be cached while the model and preprocessing remain unchanged.
- **Memory.** A database of $N$ float32 embeddings with dimension $d$ occupies about $N\cdot d\cdot4$ bytes. For $N=10^6$ and $d=512$, this is about 2 GB.
- **Search.** Exact comparison with the full database costs $O(N\cdot d)$. Large collections use approximate nearest-neighbor search such as FAISS or vector databases.
- **Preprocessing.** Vectors produced with a different normalization or input size are not directly comparable.
- **Distribution shift.** Domain shift in new queries can reduce search quality without any code change.

## Key points

- An embedding $\mathbf e=f_\theta(x)$ is a learned representation distinct from logits and raw data.
- Euclidean distance uses the norm, while cosine similarity measures direction.
- L2 normalization aligns both rankings but does not guarantee better results.
- PCA shows directions of maximum variance rather than maximum class separation.
- `nn.Embedding` is a trainable table whose structure emerges during learning.

## Questions for students

1. What values do cosine similarity and Euclidean distance take for vectors with the same direction and different lengths?
2. Why is the `avgpool` output more suitable than `fc` for comparing images?
3. What happens when an embedding with a near-zero norm is normalized without an $\varepsilon$ safeguard?
4. How can the similarity of two points close together in a 2D PCA plot be verified in the full feature space?

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. Journal of Machine Learning Research, 9, 2579-2605.
- Jolliffe, I. T., & Cadima, J. (2016). *Principal component analysis: a review and recent developments*. Philosophical Transactions of the Royal Society A, 374(2065), 20150202.
- PyTorch documentation. *torch.nn.Embedding*. `https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html`.
- PyTorch documentation. *torch.linalg.eigh*. `https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html`.
