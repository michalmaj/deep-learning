Language: **English (default)** | [Polski](lecture.md)

# Lecture 08 — From representation to similarity: embeddings and the geometry of feature space

## Learning goals

1. Understand what an embedding is -- a vector representation of an object -- and how it differs from hand-designed features and from a classifier's output (logits, probabilities).
2. Correctly compute and interpret Euclidean distance and cosine similarity, know when they give a different result and why.
3. Understand the role of L2 normalization and know that it does not guarantee better results.
4. Be able to verify empirically whether learned representations actually give a more useful neighborhood than raw data -- do not assume it in advance.
5. Understand the limitations of visualizing high-dimensional representations (PCA, briefly t-SNE/UMAP) -- what they show, and what they do not prove.
6. Learn `nn.Embedding` as a learned table of vectors indexed by identifiers -- preparation for modules 09-10 (sequences, attention, Transformer).

## Why this matters

In module 7, a CNN backbone served as a feature extractor -- we took its output and fed it to a new classification head, without really asking *what* that intermediate vector *is* or *why* it works at all. This lecture answers that question directly: what a vector representation is, when the distance between two such vectors means something, and how to use that property (finding similar objects, recommendations, semantic search) -- and when it does *not* mean what we would intuitively expect.

## Intuition

A **representation** is a way of writing down an object (an image, a word, a user) as something you can compute with. An image's raw pixels are also a representation -- a vector of numbers -- but a representation **imposed by hand** through the way the data is stored (the pixel grid, channel order), not through what is actually relevant in the image. An **embedding** is a **learned** representation: a neural network, trained on some task, decides for itself which dimensions of the vector are useful, based on what helped minimize the loss.

❌ "Since I wrote the data down as a vector of numbers, the distance between vectors automatically means something."

✅ Distance in a numerical space always has geometric meaning (that is just mathematics), but **semantic meaning** (whether nearby vectors correspond to "similar" objects in the sense we care about) depends entirely on **how** the vector was produced. A vector of raw pixels and a feature vector from a trained CNN have the same mathematical form (a sequence of numbers), but completely different geometric usefulness -- we verify this numerically in experiment B, we do not assume it.

## Formal model

An embedding is a mapping from an input object to a vector of a fixed number of dimensions:

$$\mathbf e = f_\theta(x), \qquad x \in \mathcal X,\quad \mathbf e \in \mathbb R^d$$

where $\mathcal X$ is the **input space** (e.g. $3\times224\times224$ images, sentences, product identifiers), $\mathbb R^d$ is the **representation space** of dimension $d$ (e.g. $d=512$ for ResNet18's `avgpool` output), and $\theta$ are the parameters of the mapping (the network's weights).

**A note on notation.** We use the bold $\mathbf e$ for the embedding vector -- a **new symbol**, not $z$. In this course $z$ has meant **logits** since module 2 (a layer's output *before* sigmoid/softmax, e.g. `BCEWithLogitsLoss` operates on $z$) -- an embedding is **not** a logit (see below for why these are two different objects). $\mathbf e$ has nothing to do with Euler's number $e$ (which only appeared in $e^{-z}$ in module 2, not in this lecture) or with the epoch index $e$ from module 5 -- in this document, bold $\mathbf e$ always denotes a vector, the context is unambiguous.

**Embeddings from CNN hidden layers.** A CNN backbone (modules 6-7) is a composition of convolutional layers ending (in ResNet18) with global pooling -- the output of that pooling, **before** the final `fc` layer, is exactly $\mathbf e$. Three different objects that are easy to confuse:

| Object | Source | Dimension (ResNet18, 10 classes) | Used for |
|---|---|---|---|
| Embedding $\mathbf e$ | `avgpool` output | $512$ | similarity comparisons, search |
| Logits $z$ | `fc` output (before softmax) | $10$ (number of classes) | input to the loss function |
| Probabilities $\hat y=\text{softmax}(z)$ | `softmax(fc(...))` | $10$ | interpreting the model's "confidence" |

❌ "I can use the logits (the `fc` output) instead of the `avgpool` embedding -- it's essentially the same thing, just a different step."

✅ Logits are **specific to the particular classification task** (10 or 1000 specific classes) -- their dimensionality and meaning are tied to the labels the model was trained on. The `avgpool` embedding is a **more general** representation, independent of the specific head -- that is why in module 7 we replaced `fc`, not `avgpool`, when transferring the model to a new task. For similarity comparisons between images, the **embedding, not the logits**, is the right choice -- the logits of two images of different, unknown-to-the-original-task classes can end up close for incidental reasons (both classes "don't fit" any known category in a similar way), not because the images themselves are similar.

## Mathematics: similarity and distance

**Euclidean distance:**

$$d(\mathbf a,\mathbf b) = \|\mathbf a - \mathbf b\|_2 = \sqrt{\textstyle\sum_i (a_i-b_i)^2}$$

**Dot product** and **cosine similarity:**

$$\mathbf a \cdot \mathbf b = \textstyle\sum_i a_i b_i, \qquad \cos(\mathbf a,\mathbf b) = \frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|_2\,\|\mathbf b\|_2} \in [-1,1]$$

Cosine similarity measures the **angle** between vectors, ignoring their length. Euclidean distance measures **the difference in both direction and length**. These are two different questions ("do they point in a similar direction?" vs. "how far apart are their endpoints?") and can give **different nearest-neighbor rankings** -- we show this below numerically, not just declaratively.

**L2 normalization:** $\hat{\mathbf e} = \mathbf e / \|\mathbf e\|_2$, so that $\|\hat{\mathbf e}\|_2 = 1$. For vectors normalized to unit length:

$$\|\hat{\mathbf a}-\hat{\mathbf b}\|_2^2 = 2 - 2\cos(\hat{\mathbf a},\hat{\mathbf b})$$

(verified numerically on random vectors in the notebook -- a difference on the order of $10^{-6}$, within floating-point error). This means that **for L2-normalized vectors, the nearest-neighbor ranking by Euclidean distance and by cosine similarity is identical** (a monotonic transformation of the same quantity) -- but **only after normalization**. Before normalization, the two metrics can differ.

❌ "L2 normalization always improves search quality."

✅ Normalization **changes the geometry** (it removes the effect of vector length on distance), but whether this helps depends on whether the length of a vector in your representation carries useful information, or just noise. We check this experimentally in the lab.

**An edge case: the zero vector.** $\mathbf e = \mathbf 0 \Rightarrow \|\mathbf e\|_2 = 0 \Rightarrow \mathbf e/\|\mathbf e\|_2 = \mathbf 0/0$ -- **undefined** (in PyTorch: `nan`, verified in the notebook). In practice, a small $\varepsilon$ is added to the denominator (`e / (e.norm() + eps)`) -- this is not cosmetic, it is the only way to keep the function from blowing up on a degenerate input.

**k nearest neighbors (k-NN).** For a query $\mathbf q$ and a set of vectors $\{\mathbf e_1,\ldots,\mathbf e_N\}$, $k$-NN is the $k$ vectors with the smallest $d(\mathbf q,\mathbf e_i)$ (or the largest $\cos(\mathbf q,\mathbf e_i)$). **A note on notation:** $k$ here is the number of neighbors -- a **different meaning** than $k$ in module 6 (the convolutional filter size $k\times k$). Same symbol, different context -- the lecture's topic disambiguates it, there is no real risk of confusion in practice, but it is worth naming explicitly.

## Worked example: different rankings for different metrics

Query $\mathbf q=(1,0)$. Two candidates: $\mathbf a=(10,0)$, $\mathbf b=(0.9, 0.9)$.

$$d(\mathbf q,\mathbf a) = \|(1,0)-(10,0)\|_2 = 9.0 \qquad d(\mathbf q,\mathbf b) = \|(1,0)-(0.9,0.9)\|_2 \approx 0.906$$
$$\cos(\mathbf q,\mathbf a) = \frac{10}{1\cdot 10} = 1.0 \qquad \cos(\mathbf q,\mathbf b) = \frac{0.9}{1\cdot\sqrt{1.62}} \approx 0.707$$

**Euclidean distance picks $\mathbf b$ as the closer neighbor** (smaller distance: $0.906 < 9.0$), **cosine similarity picks $\mathbf a$** (larger similarity: $1.0 > 0.707$) -- because $\mathbf a$ points in *exactly* the same direction as the query, despite its large norm. Verified programmatically in the notebook. This is not a contrived pathological example -- it is exactly the situation where **the vector's scale** ($\mathbf a$'s norm is $10\times$ larger than $\mathbf q$'s) **dominates Euclidean distance**, while cosine similarity is immune to it by definition.

## Dimensionality reduction: the intuition behind PCA

Embeddings typically have dozens to hundreds of dimensions (ResNet18: $512$) -- you cannot "see" them directly. PCA (*Principal Component Analysis*) finds directions $\mathbf u_1,\mathbf u_2,\ldots$ (unit vectors, mutually orthogonal) in embedding space along which the data has **maximum variance** -- $\mathbf u_1$ is the direction of the largest spread of the data, $\mathbf u_2$ is the direction of the largest spread **among the remaining directions, orthogonal to $\mathbf u_1$**, and so on. Projecting a point onto the first two directions:

$$\mathbf e' = (\mathbf u_1^\top \mathbf e,\ \mathbf u_2^\top \mathbf e) \in \mathbb R^2$$

gives something you can plot on a plane. Implementation-wise (notebook): we compute the covariance matrix of the embeddings and its eigendecomposition (`torch.linalg.eigh`) -- $\mathbf u_1,\mathbf u_2$ are the eigenvectors corresponding to the two largest eigenvalues. We do not require a full derivation of SVD -- it is enough to understand **what PCA optimizes** (the variance of the projection) and **what that means for the information retained**: directions of **small** variance (e.g. $\mathbf u_{500}, \mathbf u_{501}$) are discarded, and along with them -- whatever was encoded in them.

❌ "PCA in 2D shows whether the classes are well-separable in the full space."

✅ PCA shows the directions of **maximum variance**, not necessarily the directions that **best separate the classes** (these are different criteria -- PCA knows nothing about labels). Good class separation in 2D is **sufficient** evidence that *some* structure exists, but poor separation in 2D **is not** evidence of poor separation in the full $512$ dimensions -- the first two directions may simply not be the ones that separate the classes.

**t-SNE / UMAP (briefly).** These are other, nonlinear visualization methods that often give *visually* clearer clusters than PCA -- but at the cost of distorting the global geometry.

❌ "t-SNE shows the true distances and cluster sizes from the original space."

✅ t-SNE deliberately distorts distances to highlight local neighborhood structure -- **cluster size in a t-SNE plot does not correspond** to size/density in the original space, and the distance between two separate clusters **is not** a reliable measure of how "different" they are. We do not develop a comparison of algorithms here -- that is a topic for a separate study, not the core of this lecture.

## Learning representations from relations between examples -- a brief preview

How does an embedding "know" that two images should be close to each other? In our experiment B, the embedding comes from a network trained for **classification** on ImageNet -- closeness in this space is a side effect of that task, not its direct goal. There is a family of methods that **directly** optimize the geometry: *metric learning* / *contrastive learning* -- instead of "predict class $y$," the training objective is "for pairs/triplets of examples, pull closer together those that should be similar, push apart those that should not" (e.g. the same object from two photos vs. different objects). This is a topic we will return to alongside more advanced architectures -- here just the intuition: the training objective **directly shapes** the geometry of the learned space, not just whether the model classifies well.

## `nn.Embedding`: a learned table of vectors for discrete identifiers

Everything above concerned embeddings **computed** from continuous data (image -> network -> vector). For **discrete** data (a word, a token, a user identifier) there is no natural continuous input -- instead, a **table** of vectors is used, one for each possible identifier, read out by indexing:

```python
import torch.nn as nn

V, d = 6, 4  # V = vocabulary size (number of possible tokens), d = embedding dimension
embedding = nn.Embedding(V, d)   # a V x d table, randomly initialized, trainable
token_ids = torch.tensor([0, 3, 1])
vectors = embedding(token_ids)   # (3, 4) -- simply reading the corresponding rows
```

`nn.Embedding(V, d)` is a $V\times d$ matrix stored as an ordinary, trainable parameter (`embedding.weight`) -- `forward` is **indexing** (reading rows), not matrix multiplication. The gradient flowing backward updates **only the rows** that occurred in a given batch (verified in the notebook: rows not used at all get no gradient whatsoever, so they do not change -- exactly like the frozen layers in module 7, except here the "freezing" comes from absence in the data, not from `requires_grad`).

❌ "Token embeddings immediately have a sensible semantic structure -- similar words are close to each other, because that's just how `nn.Embedding` works."

✅ A **freshly initialized** `nn.Embedding` is simply random vectors -- there is no semantic relationship between them at all, until training (on some objective -- predicting a neighboring word, translation, anything) starts modifying them so that they help minimize the loss. This is exactly the same mechanism as the CNN embedding above: the structure of the space is **an effect of training toward a specific objective**, not a property of the mere fact "this is an embedding." A full discussion of how such an objective is trained for sequences (RNN, attention, Transformer) -- modules 9 and 10.

## Engineering implications

- **Extraction cost** is usually one forward pass through the backbone per image -- computed once, it can be **cached** (module 7) and reused many times, as long as the backbone and preprocessing do not change.
- **Memory**: $N$ images $\times$ $d$ dimensions $\times$ 4 bytes (float32) -- for $N=10^6$, $d=512$: about $2$GB. The embedding's dimensionality therefore has a direct effect on the cost of storing the whole database.
- **Search cost**: comparing one query against $N$ vectors is $O(N\cdot d)$ -- linear in the size of the database. For large $N$ (millions+), **approximate** nearest-neighbor search is used (ANN -- *approximate nearest neighbors*, e.g. FAISS, vector databases) instead of an exact comparison against everything -- an accuracy/speed trade-off, not the topic of this lecture, but worth knowing it exists.
- **Preprocessing consistency** is critical: an embedding computed with different preprocessing (a different normalization, a different input size) **is not comparable** with the rest of the database, even if it comes from the same model.
- **Quality degradation over time**: if the distribution of new queries/documents shifts relative to the data the model was trained on (similar to the *domain shift* from module 7), search quality can degrade with no change in the code at all.

## What to remember

- The embedding $\mathbf e=f_\theta(x)$ is a **learned** representation, different from the logits $z$ and from the raw input data -- all three are vectors, but of different usefulness.
- Euclidean distance and cosine similarity measure different things and can give a different ranking -- cosine ignores vector length, Euclidean does not.
- L2 normalization changes the geometry (after normalization, Euclidean and cosine rankings coincide), but does not automatically guarantee better results.
- PCA shows the directions of maximum variance, not maximum class separability -- good/poor separation in 2D does not directly prove anything about the full space.
- `nn.Embedding` is a learned table of vectors -- semantic structure is an effect of training, not a default property.

## Questions for students

1. You have two vectors with the same direction but different lengths. What is their cosine similarity? What is their Euclidean distance? Does it depend on the length?
2. Why is it better to use the `avgpool` output rather than the `fc` output of a classification model for similarity comparisons between images?
3. You have an embedding $\mathbf e$ with a norm close to zero. What happens if you normalize it by dividing by the norm without an $\varepsilon$ safeguard? How do you fix it?
4. Two points are close together on a 2D PCA plot. What additional check would you perform before concluding that they are "similar" in the full feature space?

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. (the backbone used to extract embeddings -- ResNet18, module 7)
- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. Journal of Machine Learning Research, 9, 2579-2605.
- Jolliffe, I. T., & Cadima, J. (2016). *Principal component analysis: a review and recent developments*. Philosophical Transactions of the Royal Society A, 374(2065), 20150202.
- PyTorch documentation: *torch.nn.Embedding*, `https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html`.
- PyTorch documentation: *torch.linalg.eigh*, `https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html`.
