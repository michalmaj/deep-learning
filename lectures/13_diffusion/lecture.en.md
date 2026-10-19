Language: **English (default)** | [Polski](lecture.md)

# Lecture 13 — Diffusion models: learning to generate by denoising

## Lecture goals

1. Understand forward diffusion as a controlled, mathematically precise
   process of adding noise — not something the model has to learn.
2. Be able to derive and compute the closed form $x_t = \sqrt{\bar\alpha_t}
   x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ for a given $x_0$, $t$, $\varepsilon$.
3. Understand that the model predicts **noise**, not the data directly —
   and how this prediction is used in reverse diffusion.
4. Be able to implement a small DDPM independently: schedule, training,
   reverse sampling.
5. Observe data structure emerging from pure noise, not the reconstruction
   of a given sample.
6. Investigate the effect of the noise schedule, the number of steps, and
   noise-prediction quality on generation quality — **measured**, not just
   claimed.
7. Critically evaluate a generator's quality — not just "does the plot look
   good."
8. Compare diffusion with VAE (module 12) and autoregressive generation
   (module 11) without declaring one approach the winner.

## Why this matters

Module 11 taught **sequential, token-by-token** generation: each next
element depends on the previous ones, in one model pass per token. Module
12 taught generation **by sampling from an explicitly designed latent
space**: one decoder pass, one sample $\mathbf z$ → one generated example.
Diffusion introduces a **third, qualitatively different** mechanism: we
start from pure noise and **iteratively** transform it, calling the same,
relatively simple network many times. This is not a variant of VAE or
autoregression — it is a separate way of decomposing a hard task
("generate a realistic sample") into many easier subtasks ("remove a
little noise, given the noise level").

We do not train Stable Diffusion here, do not download weights, do not use
the `diffusers` library as a dependency — the DDPM mechanism is fully
visible in the code, on 2D data, so that nothing is a "black box."

## Intuition

❌ "Diffusion is an algorithm that repeatedly applies a smoothing/denoising
filter to an image, like in photo editing."

✅ Classical denoising (e.g. a Gaussian filter, a median filter) **removes a
known type of corruption from a SPECIFIC, fixed image** — it does not
create new data. A diffusion model does something fundamentally different:
it learns **what the training data looks like in general** (their
distribution), and then generation consists of **repeatedly** invoking
that knowledge, starting from a **pure, random** starting point — not from
a noised version of an existing example we have to reconstruct. That is
the difference between "fix THIS photo" and "invent a NEW photo, knowing
what photos of this type look like."

**Key idea (two sides of the same process):**

1. **Forward diffusion**: corrupt the training data by adding noise in a
   controlled, known way, step by step, until it loses all structure.
2. **Reverse diffusion**: teach a network to reverse **one small step** of
   this process — then apply that learned operation repeatedly, starting
   from pure noise, to "grow" a new sample.

## A brief comparison of generative model families

| Family | How it generates | What it learns |
|---|---|---|
| Autoregressive (module 11) | Element by element, conditioned on previous ones | $P_\theta(x_t\mid x_{<t})$ |
| VAE (module 12) | One sample $\mathbf z\sim p(\mathbf z)$ → one decoder pass | A probabilistic latent model, ELBO |
| GAN (mentioned briefly, not developed) | One generator pass from noise, trained adversarially with a discriminator | An implicit data distribution, a generator-discriminator game |
| Diffusion (this lecture) | Iteratively reversing noising, many passes of the same network | Predicting the noise $\varepsilon_\theta(x_t,t)$ |

We do not develop the derivation of adversarial training (GAN) here — it
is a separate topic, requiring its own game theory and training
stabilization, outside the scope of this module.

## Forward diffusion: the mathematics

**Notation.** $x_0$ — a clean data sample (emphasis: "clean", before any
noising). $x_t$ — the sample after $t$ noising steps. $T$ — the total,
fixed-in-advance number of diffusion steps.

**Notation note ($x_t$, third meaning in this course).** Module 9: $x_t$ =
a continuous RNN input vector at time $t$ (index = **position in the
sequence**). Module 11: $x_t$ = a discrete token id at position $t$
(already disambiguated from module 9). Here: $x_t$ = a continuous
point/image after $t$ noising steps — the index $t$ denotes the **noise
level for ONE sample**, not a position in a sequence of many elements. The
indexing axis is different, despite the same letter.

**Notation note ($T$).** Modules 9-11: $T$ = sequence length (the number
of time steps/tokens processed by an RNN/attention/LLM). Here: $T$ = the
number of **diffusion** steps for one, single sample — a different
counting axis (the noise level of one point, not the number of elements in
a sequence of many objects), analogous to the $x_t$ distinction above. Too
standard in the DDPM literature to invent a new symbol — the context
(diffusion vs. sequence) unambiguously resolves which $T$ is meant.

$\beta_t$ — the noise variance added at step $t$ ($0<\beta_t<1$).

**Notation note ($\beta$, third meaning).** Module 4: $\beta_1,\beta_2$ —
the Adam momentum coefficients. Module 12: bare $\beta$ — the KL weight in
Beta-VAE. Here: $\beta_t$, **always with the index $t$** — this
distinguishes it from both previous ones (bare in module 12, double-indexed
$1,2$ in module 4). Too standard in the DDPM literature (Ho, Jain, Abbeel,
2020) to invent a new symbol.

$\alpha_t := 1-\beta_t$, $\bar\alpha_t := \prod_{s=1}^t \alpha_s$ (the
cumulative product).

**Notation note ($\alpha$, third meaning).** Module 10: the attention
weights $\alpha_{ij}$ (double index). Module 12: the VAE interpolation
coefficient, bare $\alpha\in[0,1]$. Here: $\alpha_t$/$\bar\alpha_t$,
**single-indexed by time** — the indexing pattern (bare / $_{ij}$ / $_t$)
distinguishes all three uses.

**The forward process** (a Markov chain — each step depends **only** on
the previous one, not on the whole history):

$$q(x_t \mid x_{t-1}) = N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)$$

This is a **fixed process, with no learned parameters** — we set $\beta_t$
in advance (the schedule, below), we do not learn it. **A key difference
from module 12**: the VAE's $q_\phi(\mathbf z\mid x)$ is a **learned**
approximate posterior (hence the index $\phi$); here the forward $q$ has
**no** parameter index at all — it's an ordinary, defined formula, not
something the network learns.

### Deriving the closed form

A basic property: the sum of independent Gaussian variables is Gaussian,
with variances adding (after appropriate rescaling). Expanding the
recurrence $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,z_t$
($z_t\sim N(0,I)$ independent at each step) and substituting repeatedly:

$$x_t = \sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}\,x_0 + (\text{the sum of rescaled independent noises})$$

The sum of independent Gaussians with variances $(1-\alpha_t)\bar\alpha_t/\alpha_t,
\ldots$ sums to a variance of $1-\bar\alpha_t$ (an inductive proof — we do
not repeat the full calculation here, the result is standard in the DDPM
literature). The final result, the **closed form**:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \varepsilon\sim N(0,I)$$

**Notation note ($\varepsilon$).** The same glyph and the same conceptual
role as in module 12 (the VAE reparameterization noise: a fixed, standard
Gaussian noise used for an explicit, differentiable sample construction) —
this is a **continuation**, not a collision. Diffusion extends the same
idea from a single step (VAE) to an entire chain.

**Why this is so useful during training.** Without the closed form, to
obtain $x_t$ for training we would need to simulate the **entire** chain of
$t$ steps ($x_0\to x_1\to\cdots\to x_t$), which is costly and unnecessary.
The closed form lets us **sample $t$ and $\varepsilon$ independently, and
construct $x_t$ in one step** — exactly what we do in the training loop
(Experiment B).

### Worked numeric example

$x_0 = [2.0,\ {-}1.0]$ (a 2D point), $\varepsilon=[0.5,\ {-}0.3]$ (fixed,
for verification), a linear schedule $\beta\in[0.0001,\ 0.02]$, $T=10$:

| $t$ | $\bar\alpha_t$ | $x_t$ (closed form) |
|---|---|---|
| 1 | $0.9999$ | $[2.0049,\ {-}1.0030]$ |
| 3 | $0.9931$ | $[2.0347,\ {-}1.0215]$ |
| 6 | $0.9667$ | $[2.0577,\ {-}1.0380]$ |
| 10 | $0.9037$ | $[2.0564,\ {-}1.0437]$ |

Verified programmatically in the notebook (Experiment A) — the same
numbers, not just a hand calculation. We can see: for small $t$,
$x_t\approx x_0$ (a small change); as $t$ grows, $\bar\alpha_t$ drops, the
point increasingly "drifts" and the variance increases — but at $T=10$
with this particular schedule, $\bar\alpha_{10}=0.90$ still means a
**strong** original-data signal (see the section on the schedule below —
$T=10$ here is only for manual verification, not for actual training).

### Geometric loss of structure

In the notebook (Experiment A), we show how a set of points (a mixture of
Gaussians on a circle) gradually "blurs" into a uniform cloud as $t$
grows — the structure (eight separated clusters) fades smoothly, not
suddenly.

❌ "For any schedule and any (sufficiently large) number of steps $T$, the
distribution of $x_T$ is exactly $N(0,I)$."

✅ **Not true in general — it depends on the specific $\bar\alpha_T$.**
Measured in the notebook: for a linear schedule with $T=100$,
$\beta\in[0.0001,0.02]$: $\bar\alpha_{100}=0.364$ — a **significant
residual signal**, the distribution of $x_{100}$ **is not** close to pure
noise (the added variance is only $1-0.364=0.636$, not $1$). For the same
schedule with $T=1000$: $\bar\alpha_{1000}\approx 0.00004$ — practically
pure noise. **The choice of $T$ and the schedule $\beta$ jointly
determine** how close the final distribution is to $N(0,I)$ — this is not
an automatic consequence of "a large enough number of steps" without
reference to the actual values of $\beta$.

## Noise schedule

**Linear schedule**: $\beta_t$ grows **linearly** with $t$, from $\beta_1$
(small) to $\beta_T$ (larger, but still $<1$).

❌ "The name 'linear schedule' unambiguously says what is linear in this
schedule."

✅ The name must **indicate the quantity** that changes linearly — here:
$\beta_t$ itself, **not** $\bar\alpha_t$ (which changes nonlinearly, even
with a linear $\beta_t$, because it's a cumulative product) and **not**
SNR (signal-to-noise, below). Different papers define "linearity" for
different quantities — one always has to check which quantity is meant.

**Cosine schedule (intuition)**: $\bar\alpha_t$ is defined **directly** via
a cosine function (not $\beta_t$ directly), so that the signal
($\bar\alpha_t$) decays **more slowly** in the middle of the process and
**faster** near the end, compared to a linear schedule. We do not derive
an optimality proof for either here — this is an empirical design choice.

Measured in the notebook: for $T=100$, $\bar\alpha_t$ at $t=25,50,75,100$:

| schedule | $t=25$ | $t=50$ | $t=75$ | $t=100$ |
|---|---|---|---|---|
| linear | $0.939$ | $0.777$ | $0.567$ | $0.364$ |
| cosine | $0.847$ | $0.494$ | $0.144$ | $\approx 0$ |

The cosine schedule keeps the signal **higher** in the first half, then
destroys it **more abruptly** near the end — a different destruction
"pace," not just a different formula.

**Signal-to-noise ratio (SNR)**: $\text{SNR}_t = \bar\alpha_t / (1-\bar\alpha_t)$
— the ratio of "how much signal remains" to "how much noise there is" at
step $t$. A large SNR → little noise; SNR→0 → practically pure noise. The
schedule defines the entire SNR **trajectory** from $t=0$ to $t=T$, not
just the final value.

## Reverse diffusion and DDPM: the mathematics

**Why reversing requires information about the data distribution.** The
true $q(x_{t-1}\mid x_t)$ (without conditioning on $x_0$) depends on the
**entire** data distribution $q(x_0)$ — in general, it is intractable in
closed form (a similar problem to the intractable $p(\mathbf z\mid x)$ in
module 12). Solution: we **approximate** it with a learned distribution
$p_\theta(x_{t-1}\mid x_t)$ — here, **again reusing the notation $p_\theta$
from module 12** (the VAE decoder), but now for **one step of the reverse
process**, not for "decoding from a latent."

### A model that predicts noise

Instead of training a network that directly predicts $x_{t-1}$ (or $x_0$),
DDPM trains a network that **predicts the noise** $\varepsilon$ that was
added:

$$\varepsilon_\theta(x_t, t) \approx \varepsilon \quad \text{(the noise actually used to construct } x_t \text{)}$$

❌ "Predicting the noise is a goal in itself — the diffusion model 'wants'
to learn to recognize noise."

✅ Predicting the noise is a **parameterization choice** for the generative
model — mathematically equivalent (via the closed form above) to
predicting $x_0$ or directly the mean of the reverse distribution. This
parameterization was chosen because it **empirically** gives a more stable
training process (Ho, Jain, Abbeel, 2020) — the ultimate goal is still
**generating data**, not "recognizing noise" as a separate skill.

### The simplified objective

$$L_{\text{simple}} = \mathbb E_{x_0,t,\varepsilon}\big[\lVert \varepsilon - \varepsilon_\theta(x_t,t)\rVert^2\big], \qquad x_t = \sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$$

**Where the training signal comes from (step by step):**

1. We sample $x_0$ from the training data (an ordinary batch).
2. We sample $t\sim \text{Uniform}\{1,\ldots,T\}$ — **a different $t$ for
   each example in the batch**, not one $t$ for the whole batch.
3. We sample $\varepsilon\sim N(0,I)$ — this particular noise, which we
   will "hide" inside $x_t$.
4. We construct $x_t$ via the closed form (without simulating the chain).
5. The model predicts $\varepsilon_\theta(x_t,t)$ — it **must know $t$**
   (below), because the task "what noise must be subtracted" depends on
   **how much** noise is already there.
6. We compare the predicted and the actual noise via MSE — the training
   signal is the **difference** between what the model predicted and what
   we actually added (so we know it from the construction, not from an
   observation).

❌ "A good (low) noise-prediction loss guarantees good generation samples."

✅ **Not true — these are separate questions.** A low $L_{\text{simple}}$
only says that the model predicts noise well **for individual, independent
$(x_t,t)$ pairs from the training distribution** — it says nothing
directly about the quality of the **multi-step, iterative** generation
process, where small errors can accumulate over many steps. Measured in
the notebook (Experiment D): the loss drops **moderately** between epoch
20 and epoch 200 ($0.392\to0.336$, about a 14% relative reduction), but
fraction-near-mode rises **dramatically** over the same interval
($0.462\to0.886$) — a small difference in loss, a huge difference in
quality.

### Relation to the variational formulation of DDPM

$L_{\text{simple}}$ **is not** arbitrarily invented — it derives from the
same ELBO logic as in module 12 (a lower bound on the log-likelihood, now
for a **chain** of latent variables $x_1,\ldots,x_T$ instead of a single
$\mathbf z$). The full variational DDPM objective is a weighted sum of KL
divergences between $q$ and $p_\theta$ at each step of the chain.

❌ "The simplified MSE $L_{\text{simple}}$ is automatically identical to
the full negative ELBO."

✅ **Not true.** Ho, Jain, and Abbeel (2020) show that $L_{\text{simple}}$
is the full $-$ELBO **with the $t$-dependent weights removed** from the
individual KL terms (each step weighted equally, regardless of how much it
"really" weighs in the full variational derivation). The authors report
that this simplified, reweighted version **empirically** gives better
samples than the full, "correct" ELBO — but this is a deliberate departure
from the exact variational derivation, analogous to $\beta$-VAE's
departure from the standard ELBO (module 12), not "the same thing, just
written differently."

### The reverse-process update (an explicit convention)

We adopt **one, explicit convention** (DDPM, with the variance fixed as
$\beta_t$ — not the only possible one, but the simplest, and the one we
implement):

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\right)$$

$$x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z, \qquad z\sim N(0,I),\ \sigma_t=\sqrt{\beta_t}, \quad \text{for } t>1$$

$$x_0 = \mu_\theta(x_1,1) \quad \text{(the last step: WITHOUT adding noise)}$$

**Why the last step is different.** The true posterior variance
$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ at $t=1$
gives $\tilde\beta_1=0$ (because $\bar\alpha_0:=1$ by definition — there is
no "step before $x_0$") — formally, no random term should be added at the
last step. Verified programmatically: $\tilde\beta_0=0.00000000$ for our
schedule.

❌ "The $\mu_\theta$ formula for adjacent steps $t$ and $t{-}1$ is
automatically correct if we skip several steps at once (e.g. with fewer
sampling steps)."

✅ **Not true — confirmed experimentally in a dramatic way.** Measured in
the notebook (Experiment D, a model trained with $T=1000$): correct
**respacing** (thinning the sequence of steps with a recomputed effective
$\alpha_{\text{eff}}=\bar\alpha_t/\bar\alpha_{\text{prev}}$ for **this
particular** thinned pair of steps) degrades **smoothly** as the number of
sampling steps decreases (mode coverage: always 8/8, fraction-near-mode
$0.836\to0.798\to0.730\to0.596$ for $100\to50\to20\to10$ steps). Naive
step-skipping (reusing the **single-step** formula $\beta_t/\alpha_t$ for a
**large** skip) degrades **catastrophically** (fraction-near-mode
$0.032\to0.006\to0.008\to0.000$, coverage drops from 7/8 to 0/8 modes) —
the same model, only the correctness of the sampling procedure decides the
difference between "a bit worse" and "practically not working."

**Indexing consistency with the code.** In the code (0-indexed `t_idx`
from `0` to `T-1`, corresponding to the mathematical $t=t_{\text{idx}}+1$),
`t_idx=0` is the last reverse step (no added noise), `t_idx=T-1` is the
first step (starting from pure noise $x_T$). Verified programmatically
that `alphas_bar` at `t_idx` and the formula agree with the manual
derivation.

### Conditioning on time

❌ "A small MLP can predict the noise just as well without knowing the
current noise level $t$ — after all, it sees the noised point $x_t$, that
should be enough."

✅ **Not true — a measured, not just stated, effect.** $x_t$ **alone** does
not tell the model **how much** noise it contains relative to the original
signal — the same point $x_t$ could have arisen from a nearby $x_0$ at a
large $t$ (large noise variance) or from a distant $x_0$ at a small $t$
(small variance) — the dependence of $\varepsilon$ on $x_t$ **changes**
with $t$, so the model **must** know $t$, to know **which** dependence it
is currently learning/using. Measured ($T=1000$): a model **without** time
conditioning (identical architecture, ignoring $t$) reached a clearly
higher (worse) final loss ($0.638$ vs $0.337$) and **drastically** worse
samples — fraction-near-mode dropped from $0.868$ to $0.026$, and the mean
distance to the nearest component center exploded from $0.51$ to $24.4$
(the samples practically diverge during 1000 iterative steps with no
information about the current noise level) — a difference that is
measurable and dramatic, not just intuitive.

**Intuition for the time embedding.** A simple, normalized scalar $t/T$
would also "work" in principle, but a sinusoidal embedding (the same idea
as the positional embeddings in module 10 — extended here to the **noise
level**, not **position in a sequence**) gives the network a smooth,
multi-frequency representation of $t$, easier for subsequent linear layers
to use than one raw scalar. We do not implement a full U-Net — a small MLP
with the concatenation $[x, \text{sinusoidal\_embedding}(t)]$ is fully
sufficient for teaching purposes on 2D data.

## What we observe

All the numbers below come from an actual run of `examples.en.ipynb`
(CPU, this machine) — a mixture of 8 Gaussians on a circle (radius=4,
std=0.3 per component), $2000$ training samples.

- **Experiment A.** Forward diffusion verified: the closed form agrees with
  the step-by-step simulation statistically (mean/variance over $50\,000$
  Monte Carlo trials for $t=301$: stepwise mean$=[1.886,
  0.625]$/closed mean$=[1.886,0.634]$ vs. the expected $[1.883,0.628]$ —
  agree within Monte Carlo error, not sample-for-sample identity).
  Geometrically: 8 separated clusters smoothly blur into a uniform cloud as
  $t$ grows. $\bar\alpha_T$: $T=1000$ linear →$0.0000404$ (pure noise),
  $T=100$ linear→$0.364$ (a significant residual signal), $T=100$
  cosine→$0.0000$ (faster destruction of the signal near the end).
- **Experiment B.** Training ($T=1000$, 200 epochs, MLP $2{+}16\to128\to128\to2$,
  2000 samples): time $0.78$s on CPU. The loss drops from $0.86$ (epoch 1)
  to $0.337$ (epoch 200), not monotonically smoothly — typical for this
  task.
- **Experiment C.** Reverse diffusion from $x_T\sim N(0,I)$: after $T=1000$
  steps, the generated samples cover all 8 modes of the training
  distribution (fraction-near-mode $0.868$, coverage 8/8, mean distance to
  the nearest center $0.515$ vs. a component std of $0.3$), the generation
  time for $500$ samples is $0.18$s. Individual points' trajectories are
  visibly smooth, not abrupt, transitions from noise to structure.
- **Experiment D.** Respaced correct vs. naive wrong step-skipping (a model
  trained with $T=1000$, $n\_steps\in\{100,50,20,10\}$): correct
  fraction-near-mode $0.836\to0.798\to0.730\to0.596$ (coverage always 8/8);
  naive wrong $0.032\to0.006\to0.008\to0.000$ (coverage drops from 7/8 to
  0/8). No-time-conditioning: final loss $0.638$ (without) vs $0.337$
  (with) — fraction-near-mode $0.026$ vs $0.868$, mean distance to center
  $24.4$ vs $0.515$ (samples without time conditioning practically diverge
  during 1000 iterative steps). Undertrained snapshots: loss
  $0.498\to0.392\to0.336$ (epoch 5/20/200), fraction-near-mode
  $0.142\to0.462\to0.886$ — loss and quality do **not** improve at the same
  pace.
- **Experiment E.** MMD² on known control cases: the same distribution
  $N(0,I)$ vs. $N(0,I)$ →$0.00044$; a shifted distribution →$1.144$
  (correctly distinguishes them, verified before use on real data). On the
  data from Experiment C: MMD²(test, generated)$=0.0052$ vs. MMD²(test, a
  fresh independent real sample)$={-}0.0004$ (close to zero, as for the
  same distribution) — the generated samples are measurably closer to the
  real distribution than random noise, but not identical to a new,
  independent real sample.

## Common misconceptions

(A summary — each has a full ❌/✅ box above or in the notebook)

1. Forward diffusion is not a process the model has to learn — it is a
   fixed, predefined formula.
2. Predicting noise is a parameterization choice for the generative model,
   not a goal in itself.
3. A good noise-prediction loss does not guarantee good generation samples.
4. Reverse diffusion is not simply "subtracting noise" from data — it is
   an iterative, probabilistic process with a precisely defined formula.
5. Generation does not require providing a real sample to reconstruct —
   it starts from pure, independent noise.
6. A larger number of steps does not automatically guarantee better
   quality (though too few steps, especially without proper respacing,
   does hurt).
7. Skipping/omitting timesteps requires a mathematically correct,
   appropriately recomputed procedure — one cannot reuse the adjacent-step
   formula for large skips.
8. The model does not need to flawlessly reproduce every mode of the data
   distribution.
9. Generated samples that look visually similar to training ones do not
   automatically prove memorization of specific examples.
10. A smaller training loss need not mean a better generative model.
11. DDPM and DDIM are different sampling procedures (below), not two names
    for the same algorithm.
12. Latent diffusion does not mean generation is computationally "free" —
    it still requires many network passes, just in a smaller space.
13. Diffusion models are not automatically better than VAE or
    autoregressive models — that depends on the task, and is measured, not
    assumed in advance.

## DDPM vs. DDIM (a brief overview)

DDPM (this lecture): **stochastic** sampling — at every step we add a
random term $\sigma_t z$ (except the last step). Two generation runs from
the same $x_T$ give **different** final samples.

DDIM (*Denoising Diffusion Implicit Models*, Song, Meng, Ermon, 2021): an
alternative sampling procedure, **deterministic** (or with a tunable
amount of stochasticity), using the **same** trained model
$\varepsilon_\theta$, but a different update formula — it allows for a
much **smaller number of sampling steps** at comparable quality, without
needing to retrain.

❌ "DDPM and DDIM are two names for the same sampling algorithm."

✅ **Not true** — they differ in the update formula and in stochasticity,
even though they can use the **same** trained model $\varepsilon_\theta$.
We do not implement DDIM in this module (out of scope), but respaced
sampling (Experiment D) is conceptually related: both techniques show that
the number of **sampling** steps need not equal the number of **training**
steps $T$, if the procedure is appropriately defined.

## Diffusion in pixel space vs. latent diffusion

What we implemented works **directly** in data space (2D points; for
images — directly on pixels). **Latent diffusion** (e.g. Stable Diffusion)
first encodes the data into a **smaller** representation space (via an
encoder, conceptually like the AE/VAE encoder from module 12), performs
the **entire** diffusion process in this smaller space, and at the end
**decodes** the result back into the full data space (e.g. pixels).

❌ "Latent diffusion means generation is computationally free (or trivially
fast)."

✅ **Not true.** Latent diffusion **reduces** the cost of each single
diffusion step (a smaller space = a smaller network per step) — but it
still requires **many** passes of the denoising network (exactly like our
experiment, just in a different space), **plus** the cost of
encoding/decoding between the data space and the latent space. This is a
**per-step** cost optimization, not an elimination of diffusion's iterative
nature.

## Relation to modern image generators (conceptually)

- A **U-Net** (or another denoising network) replaces our small MLP — an
  architecture suited to images (convolutions, module 6), but the **role**
  is identical: predict the noise $\varepsilon_\theta(x_t,t)$.
- **Text conditioning**: an additional input to the denoising network (a
  text embedding, conceptually like the embeddings from module 8 and the
  attention mechanism from module 10, combining a text representation with
  the image representation at each step).
- **Classifier-free guidance**: a technique for steering sampling — the
  model is trained **with** and **without** a condition (e.g. text), and
  during generation both predictions are combined to amplify the
  condition's effect. It does not require a separate classifier (unlike the
  older *classifier guidance* technique).
- **Latent diffusion**: described above — combines an encoder/decoder
  (module 12) with an iterative diffusion process.
- **Number-of-steps/cost tradeoff**: fewer steps = faster generation, but
  (without a suitable procedure, like respacing/DDIM) worse quality —
  exactly the tradeoff we measured in Experiment D, just in 2D.

We do not implement text-to-image, do not download weights, do not add
`diffusers` as a mandatory dependency of this module — the library can
serve (locally, optionally) as a reference point for further, independent
exploration, not as part of the required material.

## Engineering implications

- **Generation cost ≠ the cost of one network pass.** Even a small network
  generates relatively slowly if it must be called $T$ times
  **sequentially** for **one** sample (each step depends on the previous
  result) — the total cost is (the cost of one pass) $\times$ (the number
  of steps), not the cost of one pass.
- **The number of steps is a direct quality/time switch** — reducing the
  number of sampling steps (correctly, via respacing/DDIM) speeds up
  generation, at some cost to quality; omitting a suitable procedure while
  reducing steps breaks the result far more than a "reasonably less
  accurate" result would.
- **Batching during generation** amortizes the cost: generating $N$
  samples at once (a batch) costs disproportionately less than $N$ times
  the cost of generating one sample, because each network step processes
  the whole batch in parallel.
- **Training cost and generation cost are different budgets** — training
  (in this module: once, ~1s) is a **one-time** cost; generation (many
  network calls) is a **repeated** cost, paid for every new sample.
- **Scaling from 2D to images** does not change the mechanism, it changes
  the **per-step cost** (a bigger network, more memory) — the basic need
  for many steps remains.
- **Validating generated-data quality requires metrics, not just plots** —
  and no single metric is complete (Experiment E shows this directly: a
  metric that doesn't distinguish distribution shape may "favor" a worse
  model).
- **Diversity and coverage are hard to separate** from individual-sample
  quality — a model may generate very realistic but poorly diverse samples
  (covering only part of the modes) — a separate question from "does a
  single sample look good."

## What to remember

- Forward diffusion: $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$
  — a fixed process, the closed form lets us sample $x_t$ without
  simulating the entire chain.
- The model learns to predict the noise $\varepsilon_\theta(x_t,t)$ — it
  **must** know $t$, because the dependence of the noise on $x_t$ changes
  with the noise level.
- $L_{\text{simple}}$ is a simplified, reweighted version of $-$ELBO for
  the chain — not an arbitrarily invented loss, but also not exactly the
  full ELBO.
- Reverse diffusion: $\mu_\theta(x_t,t)$ from an explicit formula, the last
  step without added noise ($\tilde\beta_0=0$ formally).
- A smaller number of sampling steps requires a mathematically correct
  procedure (respacing) — naive step-skipping catastrophically degrades
  quality, measured, not just claimed.
- A good training loss does not guarantee good samples — generation
  quality has to be measured separately, with multiple metrics.
- Diffusion, VAE, and autoregressive models are three different, important
  approaches — none is universally best.

## Questions for students

1. You have $x_0=[1.0,2.0]$, a linear schedule $T=50$,
   $\beta\in[0.0001,0.02]$. How would you compute $\bar\alpha_{25}$ without
   simulating 25 chain steps? What would change if you used a different,
   fixed $\varepsilon$?
2. Why **must** a noise-prediction model receive $t$ as an input, while an
   image-classification model (module 6) does not need analogous
   information?
3. You have a trained DDPM model with $T=1000$. You want to generate
   samples in $50$ steps, not $1000$. What is the **correct** procedure,
   and how does it differ from "just do every 20th step"?
4. Explain why a low noise-prediction loss on the training set is not by
   itself proof that the model generates good samples.
5. You have two metrics for evaluating a generator: one says "model A is
   better," the other "model B is better." How would you decide which one
   to trust more for a specific task?

## Bibliography

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic
  Models*. NeurIPS 33. arXiv:2006.11239 (the main DDPM source paper, the
  $L_{\text{simple}}$ objective, the reverse-step formula).
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit
  Models*. ICLR. arXiv:2010.02502 (DDIM, deterministic sampling with fewer
  steps).
- Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion
  Probabilistic Models*. ICML. arXiv:2102.09672 (the cosine schedule,
  respaced sampling).
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015).
  *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML
  (the first paper introducing the probabilistic diffusion framework).
- Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent
  Diffusion Models*. CVPR. arXiv:2112.10752 (latent diffusion, context for
  the "Relation to modern generators" section).
