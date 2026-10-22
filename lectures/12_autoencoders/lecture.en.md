Language: **English (default)** | [Polski](lecture.md)

# Lecture 12. Autoencoders, VAEs, and latent space

## Learning goals

This lecture covers the following topics:

1. reconstruction as a form of learning without labels,
2. the structure of an autoencoder and the role of its bottleneck,
3. the difference between a deterministic AE and a probabilistic VAE,
4. the ELBO derivation and its connection to the training loss,
5. reparameterization and the KL formula for a diagonal Gaussian,
6. the limits of reconstruction-based anomaly detection.

## Why reconstruction matters

Module 8 learned representations through similarity or classification. Module
11 introduced autoregressive generation, where tokens are predicted in
sequence. Reconstruction offers a third approach. The model receives unlabeled
data $x$ and must recover it from an internal, usually smaller representation.

This objective does not guarantee a useful representation. The network learns
only what reduces reconstruction error. With an oversized bottleneck, it can
nearly copy the input. Limited capacity or additional regularization is needed
to force the model to select information that matters for reconstruction.

This module uses small MLPs. Diffusion models follow in module 13, while the
Transformer architecture and autoregression were covered in modules 10 and 11.

## The autoencoder

An autoencoder consists of an encoder, an intermediate representation, and a
decoder:

$$\mathbf e = f_{\text{enc}}(x), \qquad \hat x = f_{\text{dec}}(\mathbf e).$$

$x \in \mathbb R^{d_x}$ is the input, $\mathbf e \in \mathbb R^{d_e}$ is the
bottleneck representation, and $\hat x$ is the reconstruction. Continuous data
often uses MSE:

$$L_{\text{AE}} = \frac{1}{N}\sum_{n=1}^N
\lVert x^{(n)} - \hat x^{(n)} \rVert_2^2.$$

### A numerical example

Let

$$x = [1.0,\ 0.5,\ -0.5,\ 0.0] \in \mathbb R^4$$

and

$$W_{\text{enc}} =
\begin{pmatrix}1&0&0&0\\0&1&1&0\end{pmatrix}.$$

The linear encoder without a bias gives

$$\mathbf e = W_{\text{enc}}x =
\begin{pmatrix}1.0\\0.0\end{pmatrix}.$$

With $W_{\text{dec}}=W_{\text{enc}}^\top$, the reconstruction is

$$\hat x = [1.0,\ 0.0,\ 0.0,\ 0.0],$$

and the sum of squared errors is

$$L_{\text{AE}} = 0 + 0.25 + 0.25 + 0 = 0.5.$$

The notebook reproduces this result programmatically. The second coordinate of
the code lost the difference between $x_2$ and $x_3$. An encoder-decoder
architecture alone therefore does not ensure good reconstruction. Its weights
must be trained for the data and objective.

## Where compression comes from

An undercomplete autoencoder has $d_e < d_x$. Its narrower bottleneck forces
some information loss for general full-rank data. An overcomplete model has
$d_e \geq d_x$, so this mechanism is absent. Without regularization, it can
learn a mapping close to the identity.

The notebook compares both cases on synthetic data in $\mathbb R^8$. A model
with $d_e=32$ reached an MSE close to zero, while a model with $d_e=2$ could not
go below approximately $0.69$. Compression came from a specific architectural
constraint, not from the model's name.

Other useful constraints include sparse activations, norm penalties, and the
reconstruction of corrupted inputs in a denoising autoencoder.

## A linear autoencoder and PCA

For a linear encoder and decoder, no activation, and an MSE loss, the global
optimum spans the same subspace as the first $d_e$ principal components. This
is a mathematical result about the optimal solution, not a guarantee about an
arbitrary point during training.

The encoder weights need not equal the PCA eigenvectors. They may form another
basis of the same subspace, without orthonormality or variance ordering.
Subspace agreement should be measured with principal angles, not by directly
comparing vector signs and order.

Before convergence, a model may have low reconstruction error without full
agreement with the PCA subspace. MSE and principal angles describe different
properties of the training state.

## Why a plain AE is not enough for generation

Any vector $\mathbf e$ can be passed to an AE decoder, but training does not
identify which regions of $\mathbb R^{d_e}$ correspond to real data. The
decoder only sees codes produced from training examples. A random point outside
those regions may yield a meaningless output.

Generation therefore needs a known latent distribution that can be sampled.
The prior in a VAE serves this purpose.

## The variational autoencoder

A VAE describes a joint distribution over an observation $x$ and a latent
variable $\mathbf z \in \mathbb R^{d_z}$. It includes:

- a prior $p(\mathbf z)$, usually $N(0,I)$,
- a decoder $p_\theta(x\mid\mathbf z)$,
- a posterior $p(\mathbf z\mid x)$.

Bayes' theorem gives

$$p(\mathbf z\mid x) =
\frac{p_\theta(x\mid\mathbf z)p(\mathbf z)}{p(x)}, \qquad
p(x)=\int p_\theta(x\mid\mathbf z)p(\mathbf z)\,d\mathbf z.$$

The integral over the latent space is generally intractable for a neural
decoder. The posterior is therefore approximated by a convenient family
$q_\phi(\mathbf z\mid x)$. The encoder with parameters $\phi$ is trained to
make this approximation close to the true posterior.

## The ELBO

The objective is to maximize $\log p_\theta(x)$. After introducing
$q_\phi(\mathbf z\mid x)$, it can be written as

$$\log p_\theta(x)=
\log \mathbb E_{q_\phi(\mathbf z\mid x)}
\left[\frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right].$$

Jensen's inequality gives the lower bound

$$\log p_\theta(x) \geq
\mathbb E_{q_\phi(\mathbf z\mid x)}
\left[\log\frac{p_\theta(x,\mathbf z)}{q_\phi(\mathbf z\mid x)}\right]
=: \mathcal L(\theta,\phi;x).$$

$\mathcal L$ is the ELBO, or *Evidence Lower Bound*. Expanding the joint
distribution gives

$$\mathcal L(\theta,\phi;x)=
\underbrace{\mathbb E_{q_\phi}\left[\log p_\theta(x\mid\mathbf z)\right]}
_{\text{reconstruction}}
-
\underbrace{D_{\mathrm{KL}}
\left(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\right)}_{\text{KL}}.$$

The first term rewards accurate reconstruction. The second limits the distance
between the approximate posterior and the prior. The ELBO is maximized, while
training code usually minimizes its negative:

$$L_{\text{VAE}}=-\mathcal L
=-\mathbb E_{q_\phi}\left[\log p_\theta(x\mid\mathbf z)\right]
+D_{\mathrm{KL}}
\left(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\right).$$

The sum of reconstruction loss and KL is not arbitrary. It follows from
specific assumptions about the prior, approximate posterior, and observation
distribution. Changing the observation model changes the first term.

## A Gaussian latent space

A standard VAE assumes

$$p(\mathbf z)=N(0,I), \qquad
q_\phi(\mathbf z\mid x)=
N\left(\boldsymbol\mu(x),
\operatorname{diag}(\boldsymbol\sigma^2(x))\right).$$

The encoder returns two vectors: the mean $\boldsymbol\mu$ and the log variance
$\log\boldsymbol\sigma^2$, named `logvar` in code. Storing the logarithm helps
keep the variance positive and the computation stable.

### Reparameterization

A direct random draw does not provide an ordinary gradient path with respect
to the distribution parameters. Randomness is instead moved to an independent
variable $\boldsymbol\varepsilon$:

$$\mathbf z = \boldsymbol\mu
+\exp(0.5\cdot\text{logvar})\odot\boldsymbol\varepsilon,
\qquad \boldsymbol\varepsilon\sim N(0,I).$$

The model remains stochastic, but $\mathbf z$ becomes a differentiable
function of `mu` and `logvar`. Gradients can therefore reach the encoder.

For $\boldsymbol\mu=[0.5,-0.3]$, `logvar` $=[-0.2,0.4]$, and
$\boldsymbol\varepsilon=[1.0,-0.5]$,

$$\exp(-0.1)=0.9048, \qquad \exp(0.2)=1.2214,$$

$$\mathbf z=[1.4048,-0.9107].$$

For $\mathcal L=\sum_i z_i^2$,
$\partial\mathcal L/\partial\boldsymbol\mu=2\mathbf z$. The notebook confirms
`mu.grad` $=[2.8097,-1.8214]$ and `logvar.grad` $=[1.2711,0.5562]$. Both
gradients are finite.

### KL divergence

For one dimension,

$$D_{\mathrm{KL}}
\left(N(\mu_i,\sigma_i^2)\,\|\,N(0,1)\right)
=\frac12\left(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2\right).$$

Summing over $d_z$ dimensions gives

$$D_{\mathrm{KL}}
=-\frac12\sum_{i=1}^{d_z}
\left(1+\text{logvar}_i-\mu_i^2-e^{\text{logvar}_i}\right).$$

For the previous values, the contributions are $0.1344$ and $0.0909$, for a
total of $0.2253$. The result matches
`torch.distributions.kl_divergence` with `atol=1e-6`. Non-negativity was also
checked on 1000 random parameter pairs.

## Scaling the loss terms

Reconstruction is aggregated over pixels, KL over latent dimensions, and both
over the batch. An implicit switch from `sum` to `mean` changes their relative
weight. The notebook sums over features and averages over the batch:

```python
recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum") / x.shape[0]
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
```

With $d_z=8$, averaging rather than summing over latent dimensions divided the
effective KL weight by eight. In the measured run, KL fell from $14.4$ to
$3.76$, while reconstruction loss fell from $230.7$ to $221.6$. The lower
reconstruction loss did not indicate a better VAE. It reflected weaker latent
regularization.

### Observation model and reconstruction loss

The form of reconstruction depends on $p_\theta(x\mid\mathbf z)$:

- a Bernoulli distribution leads to binary cross-entropy per pixel,
- a Gaussian with fixed variance leads, up to a constant, to MSE.

Bernoulli is an approximation for grayscale images because pixels are neither
truly binary nor independent. It is sufficient for KMNIST and Fashion-MNIST in
this course. `binary_cross_entropy_with_logits` combines sigmoid and BCE in one
numerically stable operation.

## Beta-VAE and posterior collapse

Beta-VAE gives the KL term a weight $\beta$:

$$L_{\beta\text{-VAE}}=
-\mathbb E_{q_\phi}[\log p_\theta(x\mid\mathbf z)]
+\beta D_{\mathrm{KL}}
\left(q_\phi(\mathbf z\mid x)\,\|\,p(\mathbf z)\right).$$

For $\beta=1$, this is the standard negative ELBO. For $\beta\neq1$, it is a
modified training objective rather than the ELBO of the same probabilistic
model. A larger $\beta$ usually pulls the posterior more strongly toward the
prior, but it does not guarantee interpretable or disentangled factors.

Excessive KL weight can cause posterior collapse. In that state,
$q_\phi(\mathbf z\mid x)\approx p(\mathbf z)$ regardless of the input. The code
stops carrying information about $x$, and reconstruction degrades. A low KL is
therefore not a standalone measure of model quality.

## Three operations in latent space

The same decoder can receive a code from three different sources:

1. reconstruction uses $\boldsymbol\mu(x)$ or a sample from
   $q_\phi(\mathbf z\mid x)$ for a specific input,
2. interpolation uses
   $\mathbf z=(1-\alpha)\boldsymbol\mu(x_a)+\alpha\boldsymbol\mu(x_b)$,
3. generation draws $\mathbf z\sim N(0,I)$ without reference to a particular
   example.

These are not three names for the same operation. In Experiment C, the mean
pixel difference between a reconstruction and a prior sample was $0.0881$.

Interpolation is not guaranteed to be semantically smooth. It depends on
training, model capacity, and the locations of both endpoints. A model with
$d_z=2$ is easy to visualize but has much less capacity than the $d_z=16$
version. A two-dimensional plot does not describe every property of the larger
model.

## Anomaly detection through reconstruction

An AE trained only on normal data may reconstruct it better than unusual data.
High reconstruction error can then serve as an anomaly score.

The signal is imperfect. An unusual example that is structurally similar to
the normal class may reconstruct well. A rare normal example may instead have
high error. The threshold must be selected on validation data, never on the
test set, and should reflect the costs of false alarms and missed anomalies.

A VAE is not inherently a better detector than an AE. It optimizes the ELBO,
not class separation by reconstruction error. Their comparison is an empirical
question for a particular dataset.

## Links to earlier modules

- In module 8, representation followed from similarity or classification. Here
  it follows from reconstruction.
- The model in module 11 generated one token at a time. A VAE decodes a sample
  from an explicit, continuous latent space.

These are different training objectives and generative paradigms, even though
all of them produce vector representations of data.

## Experimental results

The following numbers come from running `examples.ipynb` on a CPU with 8000
KMNIST training images and 2000 test images. They are not a universal
benchmark.

- **Experiment A.** After 20 epochs and across 3 seeds, test MSE was about
  $0.083$ for $d_e=2$, $0.063$ for $d_e=8$, and $0.052$ for $d_e=32$. A
  larger bottleneck helped in this run, but this is not a general law.
- **Experiment B.** For a VAE with $d_z=16$, reconstruction fell from about
  $384$ to $220$, while KL rose from about $1.6$ to $16.8$. Better
  reconstruction required a larger departure of the posterior from the prior.
- **Experiment C.** For $d_z=2$, interpolation between classes 2 and 9 had
  consecutive pixel differences $[0.0293, 0.0402, 0.0536, 0.0566, 0.0501,
  0.0411]$. The local transition was smooth, with no sudden jump.
- **Experiment D.** For $d_z=8$, $\beta=0$ gave KL near $207$. At $\beta=1$,
  KL was about $15$ and reconstruction about $223$. KL fell to $5.6$ for
  $\beta=5$ and $0.67$ for $\beta=25$, but reconstruction worsened to about
  $322$.

The last experiment shows the cost of both weak and excessive regularization.

## Engineering implications

- Latent dimension creates a trade-off among cost, reconstruction quality, and
  ease of interpretation.
- AE compression is lossy and tied to the training data. It is not a general
  compression algorithm.
- VAE generation needs a prior and KL regularization to organize latent space.
- An anomaly detector requires separate threshold validation and error analysis.
- Data drift can change the distribution of normal examples, so the model and
  threshold need periodic reassessment.
- Cost depends primarily on architecture and data dimension. The small MLP in
  this module does not reflect the scale of production image models.

## Key points

- An AE compresses only when architecture or regularization truly forces it.
- A VAE learns a distribution over latent codes, and its encoder returns `mu`
  and `logvar`.
- The ELBO is maximized, while $L_{\text{VAE}}=-\mathcal L$ is minimized.
- Reparameterization preserves randomness while enabling gradient propagation.
- `sum` and `mean` conventions change the effective weight of loss terms.
- Reconstruction, interpolation, and generation use different input codes.
- Reconstruction error is not a complete definition of an anomaly.

## Review questions

1. What behavior is expected from an overcomplete AE without regularization?
2. Why does negative ELBO have the form reconstruction plus KL? What changes
   with a Gaussian rather than Bernoulli observation model?
3. How do decoder inputs differ for reconstruction, interpolation, and
   generation?
4. How can posterior collapse be recognized in reconstruction and KL plots?
5. How should low reconstruction error for an anomaly class be investigated and
   presented as a limitation rather than an implementation bug?

## References

- Kingma, D. P., Welling, M. (2013). *Auto-Encoding Variational Bayes*.
  arXiv:1312.6114.
- Baldi, P., Hornik, K. (1989). *Neural networks and principal component
  analysis: Learning from examples without local minima*. Neural Networks,
  2(1), 53-58.
- Higgins, I. et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a
  Constrained Variational Framework*. ICLR.
- Clanuwat, T. et al. (2018). *Deep Learning for Classical Japanese
  Literature*. arXiv:1812.01718.
- An, J., Cho, S. (2015). *Variational Autoencoder based Anomaly Detection
  using Reconstruction Probability*. SNU Data Mining Center.
- PyTorch documentation for `torch.distributions.kl_divergence` and
  `torch.nn.functional.binary_cross_entropy_with_logits`.
