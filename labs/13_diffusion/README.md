Language: **English (default)** | [Polski](README.pl.md)

# Lab 13: implementing and analyzing a small diffusion model (DDPM) from scratch

Time: ca. 120 minutes. Builds on
[`lectures/13_diffusion/lecture.en.md`](../../lectures/13_diffusion/lecture.en.md)
(forward/reverse diffusion, DDPM), but on a **different problem** than the
lecture (there: a mixture of 8 Gaussians on a circle). Here: the
**two-moons** dataset -- a concave geometry, with a "hole" between the two
arms, that simple methods (e.g. a single fitted Gaussian) cannot
reproduce. This is **not** a copy of the lecture experiment with a changed
seed -- a different data shape requires the same formulas, but a different
interpretation of the results.

**Goal**: implement the mathematical core of DDPM independently (the
schedule, the closed-form forward diffusion, the objective, reverse
sampling), verify it, and critically evaluate the quality of the generated
samples -- **not** just check whether the plot "looks similar."

## 0. Setup and data (provided)

Library imports, a random seed, a function that generates the two-moons
dataset (pure NumPy/PyTorch, no `sklearn`).

## 1. Visualizing the data -- your task

Generate `X_train` (two-moons, $N=2000$) and draw a scatter plot. Explain
(in a comment or markdown): how does this shape differ from the Gaussian
mixture from the lecture -- in particular, why couldn't a single Gaussian
distribution (one mean, one covariance matrix) describe it well?

## 2. The schedule $\beta_t$ and computing $\bar\alpha_t$ -- your task

Implement a linear schedule: `betas = torch.linspace(beta_start,
beta_end, T)`, `alphas = 1 - betas`, `alphas_bar = torch.cumprod(alphas,
dim=0)`. Use $T=500$ (see the lecture: enough for $\bar\alpha_T$ to be
close to 0 for this $\beta$ range, without an unnecessarily high generation
cost).

**Verify**: print $\bar\alpha_1$, $\bar\alpha_{T/2}$, $\bar\alpha_T$ -- is
$\bar\alpha_T$ close enough to 0 (compare with the lecture's discussion of
the role of the schedule's final value)?

## 3. The closed-form forward diffusion -- your task

Implement `q_sample(x0, t, eps, alphas_bar)`:
$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$.

**Verify** (required, not optional): for a fixed $x_0$ and $t$, compare the
**statistics** (mean/variance over many samples of $\varepsilon$) of
sampling via the closed form against a step-by-step simulation (a `for`
loop over steps $1,\ldots,t$, with $q(x_s\mid x_{s-1})$). **Do not** expect
sample-for-sample identity -- these are two different, independent draws;
you are comparing **distributions**, not specific numbers.

## 4. The noise-prediction network and time conditioning -- your task

Implement the class `NoisePredictor(nn.Module)`:
- `forward(x, t)` returns the predicted noise (a 2D vector per example).
- It **must** use $t$ -- not just accept it as an argument. Use the
  provided `sinusoidal_embedding(t, dim)` function (given, below) and
  **concatenate** its result with $x$ before feeding the linear layers.

```python
def sinusoidal_embedding(t, dim=16):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)
```

**Task (comment/markdown)**: explain in your own words why $x_t$ **alone**
is not enough for the model to correctly predict the noise -- what is it
missing without knowing $t$?

## 5. The objective and the training loop -- your task

Implement one training step inside the provided epoch loop (the loop
skeleton -- `for epoch in range(epochs): ...` with the optimizer -- is
provided, the **content** of the step is your task):

1. Sample a batch $x_0$ from `X_train`.
2. Sample $t\sim\text{Uniform}\{0,\ldots,T-1\}$ -- **a different $t$ for
   each example in the batch**.
3. Sample $\varepsilon\sim N(0,I)$.
4. Construct $x_t$ (`q_sample` from section 3).
5. Compute $\varepsilon_\theta(x_t,t)$ and the MSE loss against
   $\varepsilon$.
6. Backward pass + an optimizer step.

Collect and print/visualize the loss per epoch (monitoring -- the plot is
ready to use in the visualization cell).

## 6. Reverse sampling -- your task

Implement one reverse step `ddpm_reverse_step(model, x, t_idx,
alphas, betas, alphas_bar)`:

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\right)$$

For `t_idx > 0`: $x_{t-1}=\mu_\theta+\sqrt{\beta_t}z$, $z\sim N(0,I)$. For
`t_idx == 0` (the last step): $x_0=\mu_\theta$ -- **without** adding noise
(see the lecture: why the last step is different).

Then: a function `ddpm_sample(model, n_samples, T, ...)` starting from
$x_T\sim N(0,I)$ and iterating step by step down to $t=0$.

**Verify the indexing**: print how many times the loop actually executes a
step with added noise, and how many without -- it should be $T-1$ times
with noise, $1$ time without.

## 7. Generating new samples and comparing distributions -- your task

Generate $1000$ new samples. Draw: (a) the training data, (b) the
generated samples, on the same axis scale. **Explicitly caption** that
there are no reconstructions among the generated points (we don't give the
model any real $x_0$ to reconstruct) -- all of them are **new** samples
generated from pure noise.

Compute **at least one** quantitative metric comparing the distributions
(see the "Evaluating generation" section below) on an **independent** test
set (a fresh two-moons sample, not `X_train`).

## 8. Checking a single configuration change -- your task

Choose **one** change (no more -- the brief requires isolating the effect
of a single factor):
- a different number of sampling steps (use **correct** respacing, see
  the lecture's Experiment D.2 -- do not skip steps with the single-step
  formula), **or**
- a different schedule (e.g. cosine, see the lecture), **or**
- a different number of training epochs (an undertrained vs. a
  well-trained model).

Measure the effect on generation quality (the same metric as in section
7).

## 9. A baseline -- your task

Fit a **single** Gaussian distribution to `X_train`
(`torch.distributions.MultivariateNormal` with the empirical `mean`/`cov`).
Generate $1000$ samples from it. Compute the same metrics (section 7) on
the same independent test set.

**Do not assume in advance** that the diffusion model will win on **every**
metric -- verify it. If one metric comes out ambiguous, check a
**second**, more shape-sensitive one (e.g. the mean distance to the
nearest neighbor in an independent reference set) -- explain why these two
metrics might give different conclusions.

## 10. Diagnostic task

The following, **separately provided** code snippet (not your own from
sections 4-6) implements `NoisePredictorColleague`, trains it, and
generates samples -- it does not raise an exception, it generates
**something**, but it has a bug: the network accepts `t` as an argument,
but **never uses** it in `forward`.

**Task:**
1. Train `NoisePredictorColleague` the same way as your own model
   (section 5).
2. Generate samples with the same reverse procedure (section 6).
3. Compare the metrics (section 7) with your correct model -- **do not**
   evaluate "by eye" only, compute the numbers.
4. Diagnose: why does the model without information about $t$ do
   **clearly worse**, despite still generating something (not crashing)?
   Refer back to what you wrote in the section-4 task.
5. Fix it (add time conditioning back) and verify the improvement
   quantitatively.

❌ "If the code doesn't raise an exception and generates something, the
implementation is probably correct."

✅ The absence of a runtime error is **not** proof that a generative model
is correct -- you have to **measure** quality, not just check that the code
"works" in the sense of "doesn't crash."

## 11. Conclusions -- your task

Answer, justifying with actual numbers/examples from sections 2-10:

1. Was $\bar\alpha_T$ close enough to $0$ for your schedule? What would
   have happened if it weren't?
2. How well (specific metrics) does your diffusion model reproduce the
   two-moons distribution on the independent test set?
3. How does it compare with the baseline (section 9)? Did both metrics
   give the same answer to "which model is better"?
4. What effect did the configuration change from section 8 have?
5. What error did the lack of time conditioning (section 10) introduce,
   and what general rule do you draw from it?
6. Are there regions of the data space that your model does **not**
   generate well (e.g. the "hole" between the two moons itself, or the
   ends of the arms)? How did you check this?

## Evaluating generation -- requirements

Do not evaluate the model solely on whether the point cloud "looks
similar." Required:

- **A correctly defined reference distribution**: `X_train` for training,
  an **independent** fresh two-moons set for evaluation (not the same
  points).
- **A comparison with the baseline** (section 9) -- not just "the
  diffusion model looks decent on its own."
- **At least one quantitative metric** -- recommended: MMD (Maximum Mean
  Discrepancy, RBF kernel) -- **verify it on a known case** (e.g. comparing
  two samples from the same distribution should give a result close to
  $0$) **before** using it on generated data.
- **An analysis of ungenerated/poorly-covered regions** -- check whether
  there are regions of the training data that the generator rarely/never
  hits.
- **Limitations stemming from a small number of samples** -- name them
  explicitly (e.g. "with 1000 samples, a difference between X and Y may be
  partly sampling noise, not a real difference between the models").

❌ "A larger spread of generated points means the model reproduced the
distribution better."

✅ Spread (e.g. variance) is **one** property of a distribution, not proof
of correctly reproducing its **shape** -- a model could have a similar
variance to the data while completely failing to reproduce its geometry
(e.g. filling the "hole" between the two moons, which doesn't exist in the
real data).

## Time budget (realistic, verified against actual CPU cost)

| Block | Time |
|---|---|
| Section 0 (setup, data) | 5 min |
| Section 1 (data visualization) | 5 min |
| Section 2 (schedule, ᾱ_t) | 8 min |
| Section 3 (closed form + verification) | 10 min |
| Section 4 (network + time conditioning) | 8 min |
| Section 5 (objective + training) | 13 min |
| Section 6 (reverse sampling) | 13 min |
| Section 7 (generation + comparison) | 8 min |
| Section 8 (configuration change) | 6 min |
| Section 9 (baseline) | 5 min |
| Section 10 (diagnostic task) | 10 min |
| Section 11 (conclusions) | 7 min |

Total: **98 min**, with a **22-minute** buffer (120 min total) -- more
than the required 15-minute minimum. Training (300 epochs on 2000 samples,
a small MLP) takes **a fraction of a second to ~1.5s** on CPU; generating
$1000$ samples, $T=500$ steps, takes **a fraction of a second**. The time
budget goes entirely into implementation, debugging, and analysis -- not
into waiting for the model.

## Grading criteria

- The schedule and $\bar\alpha_t$ are correct, $\bar\alpha_T$ is verified
  to be close enough to $0$.
- `q_sample` is verified statistically against the step-by-step
  simulation.
- The network uses $t$ (not just accepts it as an argument) -- the
  section-4 task is answered concretely, not with a generic statement.
- Reverse sampling correctly handles the last step (no added noise),
  verified programmatically.
- Generation is compared with an **independent** test set and with the
  baseline, with **at least one** verified quantitative metric.
- The diagnostic task correctly identifies the lack of time conditioning
  as the cause, with concrete numbers before/after the fix.
- Conclusions refer to specific numbers/examples from your own run.

A full reference solution exists locally with the instructor
(`decisions/solutions/13_diffusion/`) and is not part of this repository.
