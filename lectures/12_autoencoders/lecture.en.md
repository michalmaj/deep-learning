Language: **English (default)** | [Polski](lecture.md)

# Lecture 12 — Autoencoders (AE), variational autoencoders (VAE), and latent space

## Learning goals

1. Understand learning through reconstruction as a form of **unlabeled** (*self-supervised*) learning, and know when such reconstruction does, and does **not**, force a useful representation.
2. Be able to formally define an autoencoder (AE): encoder, bottleneck, decoder, reconstruction loss -- and compute a small numerical example.
3. Understand the difference between AE (a deterministic vector **e**) and VAE (a probabilistic distribution over **z**) -- and **why** this difference matters for generating new data.
4. Be able to derive the ELBO at a level appropriate for basic probability/analysis, distinguishing maximizing the ELBO from minimizing the training loss (= $-$ELBO).
5. Know the KL formula for a diagonal Gaussian, the reparameterization trick, and understand why it is necessary for gradient propagation through sampling.
6. Critically evaluate applying reconstruction error to anomaly detection -- know what such a procedure is, and what it is **not**.

## Why this matters

Module 8 learned representations (embeddings) through a **supervised or semi-supervised** task tied to similarity. Module 11 built a text generative model through **autoregression** -- predicting the next token. This lecture asks about a **third** way of learning representations and generating data: **reconstruction**. The network has no access to any label -- it only has the input data $x$ and the task "reconstruct $x$ from your own, internal, usually smaller representation." This lets us ask questions that have not come up before: is such a representation *automatically* useful? Is a model that reconstructs well automatically a good generative model? VAE answers "no" to both versions of this question in a formalized way -- and that is the main reason to learn VAE **after** AE, not instead of AE.

We do not develop diffusion models here (another, separate way of generating data -- module 13) and we do not re-derive the Transformer architecture (module 10) or the autoregression mechanism (module 11) -- the encoders/decoders in this module are simple MLP networks, not Transformers.

## Intuition: learning through reconstruction

❌ "A network that learns to reconstruct its input well must have learned something 'deep' about the data."

✅ The network learns **only what is needed to minimize the reconstruction loss** on a given training set, under the given architectural constraints. If the constraints are weak (e.g. a sufficiently large bottleneck), the network may learn a **trivial** mapping (nearly the identity) -- with no compression, with no "understandable" structure at all. We will see this directly in Experiment A (the "Overcomplete vs. undercomplete" section below) -- this is not a theoretical exception, just a measured phenomenon.

A good representation **can** emerge from reconstruction -- but it only emerges when the architecture **forces** non-trivial compression (a bottleneck smaller than the data's dimension) or when we add other constraints/regularization. None of this is guaranteed by the mere fact that "the network learns to reproduce the data."

## The autoencoder (AE): formal model

**Notation.** $x \in \mathbb R^{d_x}$ -- the input data ($d_x$ extends the $d$-with-index pattern from modules 08-10, here: the input dimension). **e** $\in \mathbb R^{d_e}$ (bold, module 08: representation/embedding) -- the bottleneck representation, dimension $d_e$. $\hat x$ -- the reconstruction (the $\hat y$ = prediction pattern, module 08, here applied to the whole input, not to a label).

$$\mathbf e = f_{\text{enc}}(x), \qquad \hat x = f_{\text{dec}}(\mathbf e)$$

Reconstruction loss (usually MSE for continuous data):

$$L_{\text{AE}} = \frac{1}{N}\sum_{n=1}^N \lVert x^{(n)} - \hat x^{(n)} \rVert_2^2$$

($L$ = loss, following this course's convention since module 01; $N$ = batch/dataset size, following the convention from modules 01/02/06/07).

### A small numerical example

Take $x = [1.0,\ 0.5,\ -0.5,\ 0.0] \in \mathbb R^4$ ($d_x=4$), a linear encoder with no bias to $d_e=2$ with weights

$$W_{\text{enc}} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 0 \end{pmatrix}$$

$$\mathbf e = W_{\text{enc}} x = \begin{pmatrix} 1\cdot 1.0 \\ 1\cdot 0.5 + 1\cdot(-0.5) \end{pmatrix} = \begin{pmatrix} 1.0 \\ 0.0 \end{pmatrix}$$

A decoder, also linear, $W_{\text{dec}} = W_{\text{enc}}^\top$ (a decoder trying to "invert" the projection):

$$\hat x = W_{\text{dec}}\, \mathbf e = \begin{pmatrix}1\\0\\0\\0\end{pmatrix}\cdot 1.0 + \begin{pmatrix}0\\1\\1\\0\end{pmatrix}\cdot 0.0 = \begin{pmatrix}1.0\\0.0\\0.0\\0.0\end{pmatrix}$$

$$L_{\text{AE}} = (1.0-1.0)^2+(0.5-0.0)^2+(-0.5-0.0)^2+(0.0-0.0)^2 = 0.5$$

Verified programmatically in the notebook (Experiment A, the opening cell): exactly the same numbers, not just computed on paper. This specific decoder is **not** optimal for this data (the second coordinate of $\mathbf e$ "cut off" the information about the difference between $x_2$ and $x_3$ irreversibly) -- this shows directly that the encoder/decoder architecture alone **does not guarantee** good reconstruction; you have to **train** the weights to minimize the loss, which we do next in the notebook on real images.

## Undercomplete vs. overcomplete: where compression comes from

- **Undercomplete AE**: $d_e < d_x$. The bottleneck is **narrower** than the input -- the network **must** lose some information (you cannot invertibly map $\mathbb R^{d_x}$ to $\mathbb R^{d_e}$ and back when $d_e<d_x$, for general full-rank data). This is the mechanism that **forces** compression.
- **Overcomplete AE**: $d_e \geq d_x$. The bottleneck does **not** limit the theoretical ability to map invertibly -- without additional regularization (e.g. sparsity, input/denoising noise, a norm constraint), the network can learn (nearly) the identity.

❌ "An autoencoder always compresses data into a smaller, useful representation."

✅ **Not true in general** -- measured directly on synthetic, full-rank data in $\mathbb R^8$: an overcomplete AE ($d_e=32$) reached a reconstruction MSE $\approx 0$ (a trivial identity mapping, zero compression), while an undercomplete AE ($d_e=2$) on **the same data** could not get below MSE $\approx 0.69$ -- the bottleneck **had to** force information loss. Compression is not a property of "being an autoencoder" -- it is a consequence of a **specific** architectural constraint on **specific** data. Verified in the notebook (Experiment A, the verification cell).

## The linear autoencoder and PCA

A special, instructive case: a **linear** encoder and decoder, no activation, MSE loss. In this case (Baldi & Hornik, 1989) the optimal solution **spans the same subspace** as the $d_e$ first principal components of PCA (module 08: `torch.linalg.eigh`) -- at full training convergence. This is a mathematical theorem about the task's global optimum, not an observation from one specific training run -- the correctness of the AE-PCA relationship itself does not require, and we do not invoke here, a single measured numerical example.

❌ "A linear autoencoder will automatically learn exactly the PCA eigenvectors (the same basis, in the same order as variance)."

✅ **Not true in this strong form.** The AE will learn a basis that spans **the same subspace** -- but this basis need not be orthonormal, need not be ordered by variance, and can be a **different** basis of the same subspace (the encoder and PCA can "rotate" that space differently). The reconstruction (MSE error) will be the same as for the optimal PCA projection onto that subspace -- but the encoder's raw weights generally are **not** equal to the PCA eigenvectors directly. If you want to **measure** whether two subspaces (e.g. the one spanned by the encoder's weights and the one spanned by the PCA components) are "the same" subspace, compare them via **principal angles** between the subspaces -- not by directly comparing basis vectors, their signs, or their order, because the basis of each of these subspaces is ambiguous up to rotation/reflection within it.

❌ "This AE/PCA equivalence happens automatically, regardless of how long we train."

✅ **Not true -- it requires sufficient convergence.** The theorem speaks about the **optimal** solution to the task -- an AE stopped before convergence (too few epochs, an inappropriate learning rate, stuck at a worse point) need not yet span that subspace, even if its reconstruction error is already close to the optimal PCA MSE. Reconstruction error and subspace agreement (measured by principal angles, not by MSE alone) are **separate** properties of the training state: in the limit they converge together, but before reaching that limit, a low reconstruction error **does not by itself prove** that the AE has "found" the same subspace as PCA.

## Why AE is not automatically a good generative model

An AE has a decoder $f_{\text{dec}}: \mathbb R^{d_e} \to \mathbb R^{d_x}$ -- you can feed it **any** vector $\mathbf e$ and get an "image" back. The problem: AE **does not learn** which vectors $\mathbf e$ are "sensible" (correspond to real data) -- the bottleneck only minimizes reconstruction error **for vectors that actually came from encoding real training data**. The space $\mathbb R^{d_e}$ outside the regions actually occupied by encoded training data is **undefined** -- the decoder can give anything there, usually something nonsensical.

❌ "I can draw a random vector $\mathbf e$ and feed it to the AE's decoder to generate a new, sensible example."

✅ AE **has no** mechanism that organizes the space $\mathbf e$ into some specific, known distribution (e.g. $N(0,I)$) that we could sample from. This is exactly **the** motivation for VAE's probabilistic approach, below -- VAE explicitly **designs** the latent space to have a known, fixed distribution "a priori" (a prior), from which sampling makes sense.

## VAE: a probabilistic model

Instead of a deterministic $\mathbf e = f_{\text{enc}}(x)$, VAE models the **joint distribution** of the observation $x$ and a latent variable $\mathbf z \in \mathbb R^{d_z}$ (bold **z** -- a **new** usage in this course; the plain, non-bold $z$ is logits, module 02+, also used as $z_t$ in module 11 -- both can appear in the same paragraph, e.g. with a decoder using `binary_cross_entropy_with_logits`, so the bold-vs-plain distinction here is substantive, not cosmetic):

- **Prior** $p(\mathbf z)$ -- the "a priori" distribution over the latent space, usually $\mathbf z \sim N(0, I)$.
- **Decoder (observation model)** $p_\theta(x \mid \mathbf z)$ -- how probable it is to observe $x$, given a code $\mathbf z$.
- **True posterior** $p(\mathbf z \mid x)$ -- the distribution over codes, **given a specific** $x$. This is what we would like to know when encoding -- but it is **intractable** in general:

$$p(\mathbf z \mid x) = \frac{p_\theta(x \mid \mathbf z)\,p(\mathbf z)}{p(x)}, \qquad p(x) = \int p_\theta(x \mid \mathbf z)\,p(\mathbf z)\, d\mathbf z$$

The denominator $p(x)$ requires integrating over the **entire** space $\mathbf z$ -- for a neural network as $p_\theta(x\mid \mathbf z)$ this integral has no closed form and is computationally infeasible in high dimensions.

**Variational inference, the intuition.** Since $p(\mathbf z\mid x)$ is intractable, we **approximate** it with a convenient family of distributions $q_\phi(\mathbf z \mid x)$ (here: a diagonal Gaussian, below) -- called the **approximate posterior** -- and we train the parameters $\phi$ (the encoder network) so that $q_\phi(\mathbf z\mid x)$ is **as close as possible** to the true $p(\mathbf z \mid x)$, in the sense of KL divergence (below). $q_\phi(\mathbf z\mid x)$ is never **identical** to $p(\mathbf z\mid x)$ -- it is an approximation from a restricted family of distributions, not an exact reconstruction.

## ELBO: derivation

We want to maximize the data log-likelihood $\log p_\theta(x)$, but it is intractable (it requires the same integral as above). We introduce an arbitrary distribution $q_\phi(\mathbf z \mid x)$ and write:

$$\log p_\theta(x) = \log \int p_\theta(x,\mathbf z)\, d\mathbf z = \log \int q_\phi(\mathbf z\mid x) \frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\, d\mathbf z = \log \mathbb E_{q_\phi(\mathbf z\mid x)}\!\left[\frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right]$$

By Jensen's inequality ($\log$ is concave, $\log \mathbb E[\cdot] \geq \mathbb E[\log(\cdot)]$):

$$\log p_\theta(x) \geq \mathbb E_{q_\phi(\mathbf z\mid x)}\!\left[\log \frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right] =: \mathcal L(\theta,\phi; x)$$

$\mathcal L$ (script L, mathcal -- a **new symbol in this course**, deliberately distinguished from the plain $L$ = training loss since module 01) is the **ELBO** (*Evidence Lower BOund*) -- a lower bound on the data log-likelihood that **can** be computed (or estimated by sampling). Expanding $p_\theta(x,\mathbf z) = p_\theta(x\mid \mathbf z)\,p(\mathbf z)$:

$$\mathcal L(\theta,\phi;x) = \underbrace{\mathbb E_{q_\phi(\mathbf z\mid x)}\big[\log p_\theta(x\mid \mathbf z)\big]}_{\text{reconstruction term}} - \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big)}_{\text{KL term}}$$

**Two components, two different roles:**

1. **Expected reconstruction log-likelihood** -- how well the decoder reconstructs $x$ from a sample $\mathbf z\sim q_\phi(\mathbf z\mid x)$. The higher, the better the reconstruction.
2. **KL divergence** between the approximate posterior and the prior -- how far $q_\phi(\mathbf z\mid x)$ has moved from $p(\mathbf z)$. The **lower**, the closer to the organized, known prior distribution.

❌ "We maximize the ELBO and minimize the training loss -- that is the same operation, the same sign."

✅ We **maximize** $\mathcal L$ (the ELBO) -- but in practice frameworks (and this course) **minimize** a loss. That is why we define the training loss as the **negative** ELBO:

$$L_{\text{VAE}}(\theta,\phi;x) := -\mathcal L(\theta,\phi;x) = -\mathbb E_{q_\phi(\mathbf z\mid x)}\big[\log p_\theta(x\mid \mathbf z)\big] + D_{\mathrm{KL}}\!\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big)$$

Minimizing $L_{\text{VAE}}$ $\equiv$ maximizing $\mathcal L$ $\equiv$ (from below) maximizing $\log p_\theta(x)$. Three different formulations of **the same** objective -- but only the middle/last one is something we actually maximize; what the code **minimizes** is $L_{\text{VAE}}$, with the opposite sign.

❌ "'Reconstruction loss + KL loss' is an arbitrarily invented loss function, made of two independently chosen pieces."

✅ **Not true** -- this is the **specific, derived** form of $-$ELBO for **specific** probabilistic assumptions (a prior $N(0,I)$, a specific $p_\theta(x\mid\mathbf z)$, $q_\phi$ from the family of diagonal Gaussians -- below). Changing, e.g., the decoder's distribution (Bernoulli vs. Gaussian, below) changes the **specific form** of the reconstruction term -- but still within the same derivation, not by arbitrarily appending a new term.

## A Gaussian latent space

A standard choice: a prior $p(\mathbf z) = N(0, I)$, a **diagonal** Gaussian approximate posterior:

$$q_\phi(\mathbf z \mid x) = N(\boldsymbol\mu(x),\ \mathrm{diag}(\boldsymbol\sigma^2(x)))$$

The encoder (network $\phi$) does **not** return a single vector **e** as in AE -- it returns **two** vectors, the distribution's parameters: $\boldsymbol\mu \in \mathbb R^{d_z}$ (the mean -- a free symbol, no collision in this course) and $\log\boldsymbol\sigma^2 \in \mathbb R^{d_z}$ (the log-variance, **`logvar`** in code).

**A note on notation.** We do not use bare $\sigma$ for standard deviation in this lecture's prose -- $\sigma(\cdot)$ (always with an argument in parentheses) has been reserved since module 02 for the sigmoid activation function. Here $\sigma^2$ (or $\log\sigma^2$) only ever appears as a compound form (with a `²` or `log`), never bare -- and in code and most of the text we use `logvar` directly, following the convention of the reference library (Kingma & Welling, 2013) and PyTorch.

### Why the encoder returns `mu`/`logvar`, not a ready-made sample

If we wanted to sample $\mathbf z \sim N(\boldsymbol\mu, \mathrm{diag}(\boldsymbol\sigma^2))$ **directly**, the sampling operation has no gradient with respect to $\boldsymbol\mu,\boldsymbol\sigma^2$ -- you cannot propagate error "through" a random draw the way you can through an ordinary layer.

### The reparameterization trick

The solution: we move all the randomness into a **separate** variable $\varepsilon$ independent of the network's parameters, and write $\mathbf z$ as a **deterministic** function of $\boldsymbol\mu$, `logvar`, and $\varepsilon$:

$$\mathbf z = \boldsymbol\mu + \exp(0.5 \cdot \text{logvar}) \odot \boldsymbol\varepsilon, \qquad \boldsymbol\varepsilon \sim N(0, I)$$

($\odot$ = elementwise multiplication). **A note on notation.** $\varepsilon$ (regular epsilon) here denotes the reparameterization noise -- a different glyph from the numerical-safety constant $\varepsilon$ from module 08 (`\varepsilon`, the variant epsilon) -- different uses, different glyphs, a disambiguation note at first use, following this course's established rule.

❌ "The reparameterization trick removes randomness from the VAE model."

✅ **Not true** -- the randomness is **still present** (we sample $\boldsymbol\varepsilon$ at every forward pass). The trick only **changes where in the computational graph this randomness enters** -- $\boldsymbol\varepsilon$ is an **input** (no gradient needed for it), and $\boldsymbol\mu$ and `logvar` are **ordinary, differentiable** outputs of the network. The gradient flows through $\boldsymbol\mu$ and `logvar` exactly as through any other layer -- the sampling itself is no longer "on the path" of gradient propagation.

**A numerical example (2 latent dimensions), verified against PyTorch.** $\boldsymbol\mu=[0.5,\ {-}0.3]$, logvar$=[{-}0.2,\ 0.4]$, a fixed $\boldsymbol\varepsilon=[1.0,\ {-}0.5]$ (normally sampled, fixed here for verification):

$$\exp(0.5\cdot(-0.2)) = 0.9048, \qquad \exp(0.5\cdot 0.4) = 1.2214$$
$$\mathbf z = [0.5 + 0.9048\cdot 1.0,\ \ {-}0.3 + 1.2214\cdot({-}0.5)] = [1.4048,\ {-}0.9107]$$

Verified programmatically: `torch.allclose` with the value computed by the formula, within numerical tolerance (not bit-exact in the `==` sense, since the operations are on `float32`). For the loss $\mathcal L = \sum_i z_i^2$, the computed gradient: $\partial \mathcal L/\partial \boldsymbol\mu = 2\mathbf z$ (verified programmatically: `mu.grad` $=[2.8097,\ {-}1.8214]$, exactly $2\mathbf z$), and $\partial\mathcal L/\partial\text{logvar} = \mathbf z\cdot\boldsymbol\varepsilon\cdot\exp(0.5\cdot\text{logvar})$ (verified programmatically: `logvar.grad` $=[1.2711,\ 0.5562]$, consistent with this formula). Both gradient values are finite (`torch.isfinite`) -- this is direct, computed proof that the gradient **actually flows** through the reparameterization to both of the distribution's parameters.

### KL divergence for a diagonal Gaussian vs. $N(0,I)$

For $q = N(\mu_i, \sigma_i^2)$ (one dimension $i$) and a prior $N(0,1)$, the analytical formula (the standard result for two one-dimensional Gaussians):

$$D_{\mathrm{KL}}(N(\mu_i,\sigma_i^2)\,\|\,N(0,1)) = \frac{1}{2}\left(\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2\right)$$

We sum over all $d_z$ dimensions (independence of coordinates in a diagonal Gaussian):

$$D_{\mathrm{KL}}\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big) = \frac{1}{2}\sum_{i=1}^{d_z}\left(\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2\right) = -\frac{1}{2}\sum_{i=1}^{d_z}\left(1 + \text{logvar}_i - \mu_i^2 - e^{\text{logvar}_i}\right)$$

**A numerical example, verified against `torch.distributions`.** For $\boldsymbol\mu=[0.5,{-}0.3]$, logvar$=[{-}0.2,0.4]$: KL per dimension $=[0.1344,\ 0.0909]$, sum $=0.2253$. Verified programmatically: `torch.allclose` between this formula and `torch.distributions.kl_divergence(Normal(mu,sigma), Normal(0,1))` with `atol=1e-6` -- agreement within numerical tolerance (not bit-exact in the `==` sense). Non-negativity of KL (a property of KL divergence in general) verified on $1000$ random $(\boldsymbol\mu,\text{logvar})$ pairs -- always $\geq 0$ (up to numerical error), never negative.

## Loss-scaling discipline: sum vs. mean

This is a **practical**, easy-to-miss bug: the reconstruction loss is a sum/mean over **pixels** (dimension $d_x$), and KL is a sum/mean over **latent dimensions** ($d_z$) -- and both are then aggregated over the **batch** ($N$). If these conventions are not **explicit and consistent**, the relative weight between the two loss terms changes **unintentionally**.

The convention used in this lecture's notebook (explicitly, for comparability): sum over pixels for reconstruction, sum over latent dimensions for KL, **mean over the batch** for both:

```python
recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / x.shape[0]
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
```

❌ "Using `reduction='mean'` instead of `reduction='sum'` for KL is an insignificant implementation detail."

✅ **Not true -- a measured, real effect.** Switching from a sum to a mean over latent dimensions ($d_z=8$ in our experiment) divides the effective KL weight by $d_z$ -- a model trained with this "buggy" version achieved **worse** (smaller) regularization: KL $3.76$ vs. $14.4$ for the correct version, with a deceptively **"better"** (lower) reconstruction loss ($221.6$ vs. $230.7$) -- because the model effectively had much "freer" access to the latent space. This is not just a theoretical warning -- measured in the notebook (a verification cell, not in the main Experiments A-D).

### Choosing the reconstruction loss and the observation-distribution assumption

The ELBO's reconstruction term is $\mathbb E_q[\log p_\theta(x\mid\mathbf z)]$ -- its **specific form** depends on the assumed $p_\theta(x\mid\mathbf z)$:

- **Bernoulli** (pixel values in $[0,1]$ as probabilities) -> $-\log p_\theta(x\mid\mathbf z)$ reduces to **binary cross-entropy** per pixel. Assumption: every pixel is an independent Bernoulli variable -- an **approximation** for grayscale data, not an exact model (pixels in practice are neither binary nor independent), but standard and sufficient for KMNIST/Fashion-MNIST in this course.
- **A Gaussian with fixed variance** -> $-\log p_\theta(x\mid\mathbf z)$ reduces (up to an additive constant) to **MSE**. Assumption: the observation noise is Gaussian with a **fixed** (not learned) variance -- otherwise MSE is not an exact equivalent of the log-likelihood.

We use `binary_cross_entropy_with_logits` (not a separate sigmoid + BCE) -- numerically more stable (one, combined operation, with no explicit `log(sigmoid(...))`, which can give `-inf` for extreme logits).

## Beta-VAE: KL weight as a hyperparameter

An extension: we weight the KL term with a coefficient $\beta$ (**reused** from module 04, where $\beta$ means the momentum/Adam coefficients -- a different topic, disambiguated by context, following this course's established rule for symbols too standard in the literature to change):

$$L_{\beta\text{-VAE}} = -\mathbb E_{q_\phi}[\log p_\theta(x\mid\mathbf z)] + \beta \cdot D_{\mathrm{KL}}\big(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\big)$$

❌ "Beta-VAE with $\beta\neq 1$ is still the same ELBO, just 'tuned'."

✅ **Not true.** For $\beta=1$ we recover the standard ELBO for the probabilistic model described above. For $\beta\neq 1$, $L_{\beta\text{-VAE}}$ is **not** $-$ELBO for **the same** probabilistic model -- it is a different, modified objective, empirically motivated (wanting stronger regularization of the latent space), not derived from the same probabilistic assumptions. This distinction matters: $\beta$-VAE does not "improve" the ELBO derivation, it **departs** from it.

❌ "Increasing $\beta$ always gives more interpretable / disentangled factors in the latent space."

✅ **Not true in general** -- this was the **hypothesis/motivation** of the original $\beta$-VAE paper, not a guarantee. A larger $\beta$ more strongly "squeezes" $q_\phi$ toward the prior $N(0,I)$ -- this **measurably** reduces KL (we will see this in Experiment D), but a reduction in KL **is not the same thing** as interpretability of the latent factors, and in the extreme case (very large $\beta$) it leads to **posterior collapse** (below), i.e. the **loss** of useful information in $\mathbf z$ -- the opposite of "a better representation."

### Posterior collapse

When KL $\to 0$ for **all** examples, $q_\phi(\mathbf z\mid x) \approx p(\mathbf z)$ regardless of $x$ -- the encoder "gives up" and ignores the input, the decoder has to generate $\hat x$ practically **independently** of the specific $x$. This is a degenerate case: a low KL loss, but reconstruction degrades drastically, because $\mathbf z$ no longer carries information about $x$. We will see a tendency in this direction in Experiment D at a large $\beta$ (without full collapse, within this course's epoch budget).

## Latent space: reconstruction, interpolation, generation -- three different operations

❌ "Reconstructing an example, interpolating between examples, and generating from the prior are essentially the same decoder operation."

✅ The decoder is the **same** function in all three cases -- but the **input** to the decoder comes from three **different** sources:

1. **Reconstruction**: $\mathbf z = \boldsymbol\mu(x)$ (or a sample from $q_\phi(\mathbf z\mid x)$) for a **specific, real** $x$.
2. **Interpolation**: $\mathbf z = (1-\alpha)\boldsymbol\mu(x_a) + \alpha\boldsymbol\mu(x_b)$ for $\alpha\in[0,1]$, between the codes of **two real** examples.
3. **Generation**: $\mathbf z \sim p(\mathbf z) = N(0,I)$ -- a sample **from the prior**, unrelated to any specific, real example.

Measured in the notebook (Experiment C): the pixel difference between the reconstruction of a real image and an image generated from a random prior sample is **significant** (not close to zero) -- confirming that these really are different operations, not just different names for the same activity.

❌ "Interpolation in VAE latent space is always semantically smooth, for every pair of examples."

✅ **This is not guaranteed.** The smoothness of interpolation depends on how well the model has organized the latent space -- for a small model, short training, or pairs of examples from distant regions of the space, the intermediate steps can look like "blurry" transitions with no clear visual sense, rather than a smooth, interpretable transformation. We check this empirically in Experiment C (pixel differences between consecutive interpolation steps), we do not assume it in advance.

❌ "A two-dimensional visualization of the latent space shows all the properties of the VAE model."

✅ A model with a **two-dimensional** latent space (Experiment C) is **specifically** chosen so it can be easily visualized -- but this is a drastic limitation of the model's capacity compared to, e.g., the $16$-dimensional version from Experiment B. A 2D plot is instructive for intuition, but it does **not** represent what a higher-dimensional latent space would look like/behave like -- these are two separate experiments with separate conclusions, not one plot "showing everything."

## Anomaly detection via reconstruction error -- an introduction (developed in the lab)

Intuition: if an AE/VAE is trained **only** on "normal" data, it may reconstruct it better than unusual data (since it never saw anomaly patterns during training) -- a high reconstruction error **may** suggest an anomaly.

❌ "A high reconstruction error is a reliable, complete definition of an anomaly."

✅ **Not true** -- reconstruction error is **one, imperfect** signal, not a full probability-density model of the data. A model can **reconstruct well** an anomaly example that is visually/structurally similar to the normal class (we will see this as a **measured**, not hypothetical, case in the lab) -- and it can **reconstruct poorly** some normal examples that are unusual within their own class. We develop this fully in the lab, with specific metrics and error analysis.

❌ "A generative model (VAE) is automatically a better anomaly detector than a deterministic AE."

✅ **This is not guaranteed** -- VAE optimizes a different objective (the ELBO, with KL regularization), not "optimal normal/anomaly separation." Which model gives better separation of reconstruction error between classes is an **empirical question** for a specific task, not something that can be declared in advance based on "VAE > AE" in general.

## Relationship to modules 08 and 11

- **Module 08**: learned embeddings came from a task tied to **similarity/classification** -- here the representation (**e** or **z**) results from **reconstruction**, a completely different training objective. Two different ways of arriving at a "vector representation of data" -- not variants of the same task.
- **Module 11**: an autoregressive model generates data **sequentially, token by token**, with no explicit, continuous latent space. VAE generates data **all at once**, from an explicit, continuous space $\mathbf z$, from which you can sample and interpolate -- two different, separate generative-model paradigms, not one "general" way of generating data.

## What we observe

All the numbers below come from an actual execution of `examples.ipynb` (CPU, this machine -- not a universal benchmark), the KMNIST dataset, a subset of $8000$ training / $2000$ test images. See the notebook for the full output (plots, additional seeds).

- **Experiment A (AE, effect of latent dimension).** $20$ epochs, $3$ random seeds: $d_e=2 \to$ test MSE $\approx 0.083$; $d_e=8 \to \approx 0.063$; $d_e=32 \to \approx 0.052$. In this specific, finite run, **monotonically** better with larger $d_e$ -- but we treat this as an observation of this experiment, not a universal law (a larger bottleneck could, under other conditions, overfit or not help, consistent with the warning above).
- **Experiment B (VAE, $d_z=16$).** Over $20$ epochs, the reconstruction loss dropped from $\approx 384$ to $\approx 220$, while KL **grew** from $\approx 1.6$ to $\approx 16.8$ -- during training, the model "discovers" that better reconstruction requires using more of the latent space (a larger departure from the prior), not just minimizing one term.
- **Experiment C (VAE, $d_z=2$).** Interpolation between an example of class $2$ and an example of class $9$ (7 steps): pixel differences between consecutive steps $[0.0293,\ 0.0402,\ 0.0536,\ 0.0566,\ 0.0501,\ 0.0411]$ -- stable, with no sudden jumps, a locally smooth transition, despite the small, 2-dimensional model. Reconstruction of a real image vs. generation from a random prior sample: mean pixel difference $0.0881$ -- non-trivial, clearly different results, not coincidentally similar.
- **Experiment D (beta sweep, $d_z=8$).** $\beta=0$: KL $\approx 207$ (no regularization -- the latent space is practically unshaped by the prior, behavior close to a plain AE). $\beta=1$: KL $\approx 15$, reconstruction $\approx 223$ (the reference point). $\beta=5$: KL $\approx 5.6$. $\beta=25$: KL $\approx 0.67$ (strong compression toward the prior), but reconstruction degraded to $\approx 322$ -- both phenomena ("too weak" and "too strong" regularization) **actually occurred** in this experiment, no need to correct the narrative or look for a different configuration.
- **Mean-vs-sum verification (KL reduction).** Described above in the loss-scaling-discipline section -- a measured, not just declared, effect.

## Common misconceptions

(A collected summary -- each has a full ❌/✅ block above, in the relevant section)

1. AE need not learn a semantically interpretable representation -- it depends on architectural constraints, not on the mere fact of reconstruction.
2. Good reconstruction does not automatically imply a good generative model (AE does not organize the latent space for sampling).
3. VAE does not generate new data by adding noise to input images -- it generates by sampling from the prior and decoding.
4. The prior $p(\mathbf z)$ and the approximate posterior $q_\phi(\mathbf z\mid x)$ are different distributions -- the model minimizes their distance (KL), it does not equate them.
5. The reparameterization trick does not remove randomness -- it changes where in the computational graph that randomness enters, enabling gradient propagation.
6. Minimizing the reconstruction loss alone is not the VAE's full training objective -- the KL term is an integral part of $-$ELBO, not an optional addition.
7. A higher KL divergence is not automatically a sign of a "better" model -- it is only a measure of distance from the prior, not a quality measure.
8. Beta-VAE does not guarantee disentangled latent factors -- that was the motivating hypothesis, not a proven property.
9. Sampling from the prior and reconstructing a specific input are two different operations, with different inputs to the decoder.
10. A low reconstruction error does not guarantee that an example comes from the training-data distribution -- the model may reconstruct well unusual/new data if it is structurally similar to the training data.
11. Higher visual quality of a generated image does not prove greater probabilistic fidelity of the model (these are separate evaluation criteria).
12. A two-dimensional latent space is useful for visualization, but may limit reconstruction quality relative to a higher-dimensional space -- this is a trade-off, not a "free" convenience.
13. A generative model (VAE) is not automatically a good anomaly detector -- that is a separate task with separate requirements.

## Engineering implications

- **Encoder/decoder cost** grows with network size and data dimension, not with the mere fact of "being a VAE" -- the MLP architecture used here is deliberately small (CPU, a few seconds for full training), it does not represent the cost of production generative models on large images.
- **Latent dimension is a cost/quality/interpretability trade-off** -- a larger $d_z$/$d_e$ usually (measured in this experiment) gives better reconstruction, but reduces the interpretability of visualizations and increases memory/compute cost at large scale.
- **AE as compression has limits** -- the compression is lossy and bounded by what the model actually learned on the training data; it is not a substitute for a general compression algorithm.
- **Generating new data from a VAE requires a deliberately designed latent space** (prior + KL) -- this is exactly the practical difference relative to AE, not only a theoretical one.
- **The anomaly-detection threshold must be chosen from validation data, never test data** -- the cost of an error (a missed anomaly vs. a false alarm) is a business/engineering decision, not a purely statistical one (developed in the lab).
- **Data drift** -- an AE/VAE model trained on data from one period can stop accurately reflecting the "normality" of data from another period; the threshold and the model require periodic revalidation in a practical deployment.
- Denoising autoencoders and other AE variants (not developed further here) add regularization by reconstructing **from a noised** input -- a brief mention, not a separate block.

## What to remember

- AE: $\mathbf e = f_{\text{enc}}(x)$, $\hat x = f_{\text{dec}}(\mathbf e)$, a reconstruction loss -- compression is **forced** only by a bottleneck narrower than the data, not by the mere fact of training on reconstruction.
- VAE: a probabilistic model with a prior $p(\mathbf z)$, a decoder $p_\theta(x\mid\mathbf z)$, and an approximate posterior $q_\phi(\mathbf z\mid x)$ -- the encoder returns $\boldsymbol\mu$/`logvar`, not a ready-made sample.
- ELBO $\mathcal L$ (maximized) $=$ reconstruction term $-$ KL; the training loss $L_{\text{VAE}} = -\mathcal L$ (minimized) -- this is a specific, derived expression, not an arbitrarily composed function.
- $D_{\mathrm{KL}}(N(\mu,\sigma^2)\|N(0,1)) = \frac12(\sigma^2+\mu^2-1-\log\sigma^2)$ per dimension, verified against `torch.distributions`, always $\geq 0$.
- Reparameterization: $\mathbf z = \boldsymbol\mu + \exp(0.5\cdot\text{logvar})\odot\boldsymbol\varepsilon$ -- moves the randomness to an input independent of the parameters, enabling gradient propagation.
- Sum vs. mean reduction changes the relative weight of KL/reconstruction -- it must be explicit and consistent, otherwise the training result changes unintentionally.
- Reconstruction, interpolation, and generation from the prior are three different operations with three different sources of input to the decoder.
- Reconstruction error is an imperfect anomaly signal -- not a full density model, and not a guarantee that VAE "automatically" detects anomalies better than AE.

## Questions for students

1. You have an AE with $d_e > d_x$ (overcomplete) trained with no additional regularization. What behavior do you expect, and why? How would this change if you added strong $L_2$ regularization to the encoder's weights?
2. Explain why $-$ELBO has the form "reconstruction loss + KL," not some other arbitrary combination of two losses. What would formally change if you changed the decoder's assumed distribution from Bernoulli to Gaussian?
3. You have a trained VAE with $d_z=32$. Describe exactly how (1) reconstructing a specific test image, (2) interpolating between two test images, (3) generating a new image differ -- in terms of what exactly you feed to the decoder in each case.
4. Why can a very large $\beta$ in Beta-VAE lead to posterior collapse? How would you recognize this effect, looking only at the reconstruction-loss and KL plots over training?
5. You have an AE model trained only on "normal" images, which has a low reconstruction error on a certain anomaly class. Does that mean the model is "broken"? How would you investigate this, and how would you communicate it as a system limitation, not an implementation bug?

## References

- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114 (the VAE source paper, ELBO, the reparameterization trick).
- Baldi, P., & Hornik, K. (1989). *Neural networks and principal component analysis: Learning from examples without local minima*. Neural Networks, 2(1), 53-58 (the linear-AE/PCA equivalence).
- Higgins, I., et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR (Beta-VAE, the disentanglement hypothesis).
- Clanuwat, T., et al. (2018). *Deep Learning for Classical Japanese Literature*. arXiv:1812.01718 (KMNIST -- see also `data/registry.yml`).
- An, J., & Cho, S. (2015). *Variational Autoencoder based Anomaly Detection using Reconstruction Probability*. SNU Data Mining Center, Technical Report (anomaly detection via reconstruction error/probability -- context for the lab).
- PyTorch documentation: `torch.distributions.kl_divergence`, `https://pytorch.org/docs/stable/distributions.html#torch.distributions.kl.kl_divergence`.
- PyTorch documentation: `torch.nn.functional.binary_cross_entropy_with_logits`, `https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html`.
