Language: **English (default)** | [Polski](lecture.md)

# Lecture 04: optimization -- gradient descent, SGD, momentum, and Adam

## Learning goals

After this lecture, you should be able to:

1. derive and correctly write the update rules for plain gradient descent, mini-batch SGD, momentum (in PyTorch's convention), and Adam (with bias correction), with every symbol unambiguously defined,
2. distinguish four different objects that are easy to confuse: the loss gradient, the mini-batch gradient estimate, the optimizer's state, and the parameter update vector,
3. read a loss function's contour plot and an optimizer's trajectory, and explain from it why a given algorithm oscillates, diverges, or crawls slowly,
4. distinguish asymptotic momentum stability from convergence speed and short-horizon behavior,
5. explain Adam's mechanism (adaptive per-parameter scaling) and point out that this is a **different** mechanism than momentum's damping of oscillations, not a "better version" of it,
6. explain why comparing optimizers at one, shared learning rate and declaring a winner can be misleading.

## Why this matters

In lecture 3 you learned to compute the gradient -- `backward()` is no longer a black box. But the gradient is only a **direction**. *How* to use it -- how long a step to take, whether to remember previous steps, whether to adjust the step separately for each parameter -- is a separate, rich design decision, one that determines whether the network learns at all, whether training takes minutes or days, and whether it is stable at all. This is also the first lecture in this course where the *way* of training (rather than the architecture) becomes the main character -- and where you will see that a "better" optimizer is not an absolute notion, only one that depends on the specific problem.

## Intuition

### Optimization as a search across a loss landscape

Imagine the loss $L(\theta)$ as a landscape: the height at point $\theta$ is the loss value. Training is descending this landscape, step by step, with access only to the local slope (the gradient) at the current point -- not to a map of the whole terrain. The gradient $\nabla L(\theta)$ points in the direction of **steepest increase** at a given point (lecture 1); we go in the opposite direction.

What distinguishes optimization algorithms is **what we do with that direction** beyond just "go the opposite way": whether we take a long or short step, whether we remember which way we were going before, whether the step is the same for all parameters or adjusted separately for each.

### Why the landscape can be hard

Three classic problems with the loss's geometry:

- **Ill-conditioning**: the landscape is steep in one direction and flat in another (a narrow "valley") -- a step good for the steep direction is too large for the flat one, and vice versa. This is the main character of this lecture's experiment.
- **Oscillation**: with too large a step in the steep direction, the algorithm "jumps" across the bottom of the valley back and forth, instead of sliding into it.
- **Flat regions and saddle points**: in the high-dimensional, nonlinear losses of neural networks (unlike our two-dimensional, convex bowl in the experiment), there are regions with a near-zero gradient that are not a minimum -- the gradient "falls silent" there, even though there is still somewhere to descend to. We do not demonstrate this with a separate experiment (it would require a much larger, nonlinear parameter space than our 2D bowl) -- but the mechanisms you will learn today (momentum, adaptive scaling) also help in such regions, not only with oscillation.

### Why "one number" (the learning rate) is not enough

Plain gradient descent has one global, fixed step $\eta$ for all parameters and all directions at once. Momentum adds **memory** (previous steps influence the current one). Adam adds **per-parameter adaptation** (the step is scaled separately for each parameter, based on the history of its own gradients' magnitude). These are two **different** ideas for the same problem -- and, as you will see in the experiment, neither is universally better.

## Formal model

A reminder from lecture 1: training is iteratively adjusting parameters $\theta$ to reduce $L(\theta) = \frac1N\sum_{i=1}^N \ell(f(x_i;\theta), y_i)$. In this lecture we explicitly distinguish four objects that we have so far treated together:

| Object | Notation | What it is |
|---|---|---|
| Gradient (full) | $\nabla L(\theta_t)$ | the loss's derivative over the **entire** dataset at point $\theta_t$ -- rarely computed directly (too expensive) |
| Gradient estimate | $\hat g_t$ | a gradient computed on a **mini-batch** -- a random, cheap approximate version of $\nabla L(\theta_t)$ |
| Optimizer state | e.g. $v_t$ (momentum), $(m_t, v_t)$ (Adam) | the algorithm's memory of past $\hat g_t$ -- **not** a model parameter, stored separately (in PyTorch: `optimizer.state`) |
| Update vector | $\Delta\theta_t = \theta_{t+1}-\theta_t$ | what we actually add to the parameters -- built from $\hat g_t$ and the state, scaled by $\eta$ |

`backward()` computes $\hat g_t$ (stores it in `.grad`, lecture 3). `optimizer.step()` uses $\hat g_t$ and the state to compute $\Delta\theta_t$ and apply it to $\theta$. These are two **separate** steps -- just as in lecture 3 we stressed that `backward()` does not update the parameters.

## Mathematics

### Gradient descent (full batch)

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

$\eta > 0$ is the **learning rate** -- the only hyperparameter. Introduced in lecture 1; here we generalize it.

### Mini-batch SGD

Computing $\nabla L(\theta_t)$ over the full set of $N$ examples is expensive. Instead, we sample a mini-batch $B_t$ (usually $|B_t| \ll N$) and compute:

$$\hat g_t = \frac{1}{|B_t|}\sum_{i\in B_t} \nabla \ell\big(f(x_i;\theta_t), y_i\big), \qquad \theta_{t+1} = \theta_t - \eta\, \hat g_t$$

$\hat g_t$ is an **unbiased estimator** of $\nabla L(\theta_t)$: when $B_t$ is sampled uniformly from the dataset, $\mathbb E[\hat g_t] = \nabla L(\theta_t)$ -- on average it points in the right direction, but for a single step it has variance that **decreases** with batch size ($\text{Var}[\hat g_t] \propto 1/|B_t|$). This is the formal meaning behind the statement "a mini-batch is not the same as the full dataset" from the *Common misconceptions* section: it is not a different, worse gradient -- it is the same gradient in expectation, just noisy.

**Full-batch** ($|B_t|=N$): zero noise, but one step per full pass over the data (expensive).
**Stochastic** ($|B_t|=1$): maximum noise, the cheapest single step.
**Mini-batch**: the compromise used in practice almost always -- enough examples that the noise does not dominate the direction, few enough to fit in memory and take many steps per epoch.

### Momentum

**A note on convention.** Different textbooks define the momentum variable differently -- some use a moving average $v_t = \beta v_{t-1} + (1-\beta)\hat g_t$ (similar to Adam's first moment below). **PyTorch (`torch.optim.SGD(momentum=...)`) does not do this.** In this course we consistently use PyTorch's convention, because that is what our code actually uses (verified below, in the *Worked example*, by a direct comparison against `torch.optim.SGD`):

$$v_t = \beta v_{t-1} + \hat g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t \qquad (v_0 = 0,\ \text{so } v_1 = \hat g_1)$$

$v_t$ is the **optimizer state** (here: "velocity") -- $\beta\in[0,1)$ (usually $0.9$) controls how long we remember previous steps. The physical intuition: a ball rolling across the loss landscape has **inertia** -- it does not stop immediately when the slope changes, it continues moving, gradually corrected by the current gradient. When successive $\hat g_t$ consistently point the same way (a flat but one-directional slope), $v_t$ builds up and speeds up the movement. When successive $\hat g_t$ oscillate between opposite signs (a steep "back and forth" direction), their contributions to $v_t$ partially cancel out -- hence the damping of oscillations. **But**, as you will see in the experiment, too large a $\beta$ at the same $\eta$ as plain SGD can **amplify** oscillation instead of damping it -- this is not a free mechanism.

### Adam

Adam (*Adaptive Moment Estimation*, Kingma and Ba, 2015) keeps **two** optimizer states per parameter: an exponential moving average of the gradient ($m_t$, the "first moment", similar in spirit to momentum) and an exponential moving average of the gradient's **square** ($v_t$, the "second moment", a measure of the gradient's typical magnitude/scale -- a different $v_t$ than in the momentum section above, here a different object with the same letter, matching PyTorch's and the original paper's notation; context unambiguously distinguishes them):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\hat g_t^2 \qquad (m_0=v_0=0)$$

$\beta_1$ (default $0.9$) controls memory of the **direction** -- similar to momentum, but here in a weighted-average convention (with the factor $1-\beta_1$), not the PyTorch-SGD convention from the section above. $\beta_2$ (default $0.999$, i.e. longer memory than $\beta_1$) controls memory of the gradient's **scale** per parameter -- $v_t$ grows for parameters that consistently have large gradients, and shrinks for those with small ones.

**Bias correction.** Since $m_0=v_0=0$, the first few estimates are systematically biased low (pulled toward zero) -- for $t=1$: $m_1=(1-\beta_1)\hat g_1$, i.e. only $10\%$ of the true value $\hat g_1$ when $\beta_1=0.9$. We correct this by dividing by $(1-\beta^t)$, which for small $t$ is a small number (a strong correction) and approaches $1$ for large $t$ (the correction fades once the "warm-up" is over):

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

**Update:**

$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

$\varepsilon$ (default $10^{-8}$) prevents division by zero when $\hat v_t$ is close to zero. The effect of dividing by $\sqrt{\hat v_t}$: a parameter that historically had **large** gradients gets a **smaller** effective step (a large $\sqrt{\hat v_t}$ in the denominator); a parameter with **small** gradients gets a **larger** effective step. This is exactly the per-parameter adaptive scaling discussed in the *Intuition* section -- a mechanism SGD and momentum do not have at all.

## Worked example

Let's compute three update steps by hand for $f(\theta)=\frac12\theta^2$ (so $g(\theta)=\theta$), $\theta_0=5$, $\eta=0.5$ for all three algorithms (the same $\eta$ deliberately -- see how differently they react; in practice this is rarely a good choice for all of them at once, as we will show in the experiment), momentum $\beta=0.5$, Adam $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$.

**SGD:** $\theta_1 = 5 - 0.5\cdot5 = 2.5$; $\theta_2=2.5-0.5\cdot2.5=1.25$; $\theta_3=1.25-0.5\cdot1.25=0.625$.

**Momentum ($\beta=0.5$):**

| $t$ | $g_t=\theta_{t-1}$ | $v_t=\beta v_{t-1}+g_t$ | $\theta_t=\theta_{t-1}-\eta v_t$ |
|---|---|---|---|
| 1 | $5$ | $5$ | $5-0.5\cdot5=2.5$ |
| 2 | $2.5$ | $0.5\cdot5+2.5=5.0$ | $2.5-0.5\cdot5.0=0.0$ |
| 3 | $0$ | $0.5\cdot5.0+0=2.5$ | $0-0.5\cdot2.5=-1.25$ |

Notice step 3: the gradient at $\theta_2=0$ is $0$ (we are exactly at the minimum!), and yet $\theta$ **keeps moving**, to $-1.25$ -- because $v_3$ still "remembers" the previous motion. This is a literal illustration of inertia: momentum can **overshoot** the minimum, even when the current gradient is silent.

**Adam ($\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$):**

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | $\theta_t$ |
|---|---|---|---|---|---|---|
| 1 | $5.000$ | $0.500$ | $0.025$ | $5.000$ | $25.000$ | $4.500$ |
| 2 | $4.500$ | $0.900$ | $0.045$ | $4.737$ | $22.624$ | $4.002$ |
| 3 | $4.002$ | $1.210$ | $0.061$ | $4.466$ | $20.419$ | $3.508$ |

(the full derivation of the numbers in this table -- in the notebook, verified to the sixth digit against `torch.optim.Adam`). Notice how slowly Adam descends at the same $\eta=0.5$ as SGD ($\theta$: $5\to4.5\to4.0\to3.5$) -- Adam's effective step is roughly on the order of $\eta$ (because $\hat m_t/\sqrt{\hat v_t}\approx \pm1$ once the estimates stabilize), **regardless of the gradient's actual scale** -- quite unlike SGD, where the step scales directly with the gradient's magnitude. This is not a flaw or a virtue in itself -- it is simply a **different** mechanism, which has a direct consequence: a good $\eta$ for Adam is usually **not** a good $\eta$ for SGD, and vice versa (see the experiment).

## Experiment

The notebook [`examples.ipynb`](examples.ipynb), two experiments:

**1. Optimization geometry** -- the function $f(x,y)=\frac12(25x^2+y^2)$ (clearly different curvature: steep in $x$, flat in $y$), starting at $(1,1)$, minimum at $(0,0)$:
- SGD at several $\eta$: shows the theoretical stability boundary ($\eta<2/25=0.08$ for this problem) -- stable-but-slow below it ($\eta=0.02$), oscillating-but-convergent near the boundary ($\eta=0.06$), clearly divergent above it ($\eta=0.1$),
- Momentum at **the same** $\eta$ as the oscillating SGD: $\beta=0.3$ damps the oscillation and speeds up the flat direction; $\beta=0.9$ (a typical default choice) remains asymptotically stable but has a long, strongly oscillatory transient -- a smaller $\eta$ performs better within a short step budget,
- Adam at several $\eta$: the $x$ and $y$ trajectories are (for this problem) almost identical, despite the 25-fold difference in curvature -- adaptive scaling in practice.

**2. Training a small MLP** on synthetic data (two interleaving half-moons, $N=2000$, generated locally, an 80/20 train/validation split, mini-batches of 64 examples, an explicit index-shuffling function with a fixed seed):
- comparing SGD/momentum/Adam at a **shared** $\eta=0.01$ (a controlled experiment -- but we do **not** treat the result as a general verdict),
- the same comparison with a **separately tuned**, reasonable $\eta$ for each algorithm,
- a sweep of $\eta$ for SGD showing the transition from stable, through unstable, to divergent training.

## What we observe

- Plain SGD's stability boundary on our bowl is exactly where theory predicts ($\eta=2/a=0.08$ for $f=\frac12 a x^2$, $a=25$): at $\eta=0.06$ (below the boundary) the $x$ trajectory oscillates, but the loss drops to $0.0035$ after 40 steps; at $\eta=0.1$ (above the boundary) the loss grows exponentially -- after 40 steps, on the order of $10^{15}$.
- Momentum $\beta=0.3$ at $\eta=0.06$ (the same value that oscillates for SGD) damps the oscillation in $x$ **and** speeds up convergence in $y$ -- it beats SGD on both fronts at once.
- Momentum $\beta=0.9$ at the **same** $\eta=0.06$ looks poor over the first 20 steps: the loss jumps between $\approx0.1$ and $\approx15$ and ends that interval at $\approx1.92$. A longer run, however, reaches a loss of $\approx0.0066$ after 40 steps and $\approx0.000025$ after 100. The update matrix has spectral radius $\sqrt{0.9}\approx0.9487<1$ for both curvatures, so the configuration is asymptotically stable. The smaller $\eta=0.02$ is faster within a 40-step budget (loss $0.003$), but it is not required for convergence. **A short, oscillatory run is not enough to establish instability.**
- In this experiment, Adam gives $x$ and $y$ trajectories identical to the fourth decimal place at every step -- despite the 25-fold curvature difference. At too large an $\eta$ ($0.3$-$0.5$) it oscillates, but **in both directions at once** (isotropically), not just the steep one -- a different character of instability than SGD/momentum.
- On the MLP, at a shared $\eta=0.01$: Adam clearly wins (train/validation loss $\approx0.13/0.15$ after 30 epochs) over momentum ($\approx0.25/0.30$) and SGD ($\approx0.33/0.37$). After separately tuning $\eta$ for each (SGD $0.5$, momentum $0.1$, Adam $0.01$), the gap **largely disappears** ($\approx0.14/0.17$, $\approx0.13/0.15$, $\approx0.13/0.15$ respectively -- momentum and Adam are practically tied, SGD close behind). Adam's edge at a shared $\eta$ was largely **robustness to the choice of $\eta$**, not an unconditional qualitative advantage.
- Sweeping $\eta$ for SGD on the MLP: smooth convergence at $\eta\in\{1,3\}$, growing instability at $\eta=8$-$10$ (loss spikes), clear non-convergence at $\eta\ge15$.

## Common misconceptions

❌ The gradient points to the best, finite step to take.

✅ The gradient only points to the **direction** of steepest **local** increase (lecture 1) -- it says nothing about how *far* to go in that direction. How much and what to do with that direction is the entire content of this lecture: a fixed step (SGD), a step with memory (momentum), a per-parameter adaptive step (Adam).

❌ A larger learning rate always means faster effective learning.

✅ Beyond a certain boundary (depending on the loss's curvature and the algorithm), a larger $\eta$ means oscillation or divergence, not faster learning -- demonstrated directly: SGD at $\eta=0.1$ on our bowl **diverges exponentially**, even though at $\eta=0.06$ it still converged (albeit oscillating).

❌ Adam guarantees better generalization (or even just a better result) than SGD.

✅ In our geometric experiment, with separately tuned $\eta$, a well-tuned SGD/momentum matches Adam or slightly beats it. The advantage Adam showed at a **shared** $\eta$ on the MLP was largely robustness to a poorly chosen $\eta$, not a guarantee of a better minimum or better generalization -- and it concerns one specific problem, it is not a general law.

❌ `optimizer.step()` computes the gradients.

✅ `backward()` (lecture 3) computes the gradients and stores them in `.grad`. `optimizer.step()` **reads** `.grad`, updates the optimizer's state (e.g. $v_t$, $m_t$), and applies the resulting update to the parameters -- these are two separate steps with different responsibilities.

❌ Using mini-batches means every step uses the same gradient as the full dataset.

✅ The mini-batch gradient $\hat g_t$ is a **random, unbiased estimator** of the full gradient $\nabla L(\theta_t)$ -- it points in the right direction *on average*, but a single step has variance depending on batch size. A larger batch means less noise, not a different direction.

❌ A drop in training loss guarantees a drop in validation loss.

✅ These are two separate quantities, measured on different data. In our MLP experiment they happen to move together (with no overfitting in this short training run) -- but this is not a general guarantee; the mechanisms through which they diverge (overfitting) are the topic of lecture 5.

## Engineering implications

- **Optimizer state's memory cost**: SGD without momentum stores no state at all; SGD with momentum stores one tensor per parameter ($v_t$) -- doubling the memory taken by the model's parameters themselves; Adam stores **two** tensors per parameter ($m_t$, $v_t$) -- tripling it. For large models this is a real engineering cost, not a theoretical detail.
- **Per-step compute cost**: Adam does more arithmetic per step (two moving averages, a division, a square root) than SGD -- with millions of parameters and millions of steps this difference accumulates, though the cost of the forward/backward pass itself usually still dominates over `optimizer.step()`.
- **Stability as a design cost**: a narrow, "safe" range of $\eta$ for SGD on an ill-conditioned problem (you will see this in the lab too) means more time spent tuning the hyperparameter before actual training -- Adam's adaptivity is sometimes chosen for exactly this practical reason, not because of a better convergence theory.
- **AdamW and learning-rate schedules -- a preview, not the topic**: in practice Adam is often used in its AdamW variant (which implements weight decay differently -- a full discussion when we cover regularization in lecture 5) and rarely with a constant $\eta$ for the whole training run -- it is typical to decrease $\eta$ over the course of training (a *learning rate scheduler*). These are directions for further study, not part of this lecture.

## What to remember

- The gradient is a direction; the optimization algorithm decides what to do with it -- a fixed step (SGD), a step with memory (momentum), a per-parameter adaptive step (Adam).
- A mini-batch gradient is an unbiased, but noisy, estimator of the full gradient -- not a different gradient.
- PyTorch's momentum convention ($v_t=\beta v_{t-1}+\hat g_t$) differs from the moving average sometimes used in textbooks -- do not mix them up.
- Adam normalizes the step separately for each parameter based on the history of its gradients' magnitude -- a different mechanism than momentum's damping of oscillations.
- Adding momentum changes both the stability region and the rate at which oscillations decay -- assess a learning rate for the specific algorithm and over a sufficiently long horizon instead of treating poor progress after a few steps as instability.
- Comparing optimizers at one, shared $\eta$ mainly tells you about robustness to the choice of $\eta$, not which algorithm is "generally better".

## Questions for students

1. Why, in step 3 of the momentum worked example, does the parameter keep moving, even though the gradient at that point is zero?
2. Explain why Adam's bias correction has the largest effect over the first few steps, and then practically fades away.
3. Given an ill-conditioned problem (very different curvature in different directions), why does Adam handle it "automatically", while plain SGD requires a very carefully chosen $\eta$?
4. You are given a training run where the loss grows exponentially from the first few steps. What is the most likely first hypothesis to check, and why that one specifically?

## References

- Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015 (arXiv:1412.6980).
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). *On the importance of initialization and momentum in deep learning*. ICML 2013.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/ -- chapter 8 (Optimization for Training Deep Models).
- PyTorch Contributors. *torch.optim* documentation. https://pytorch.org/docs/stable/optim.html

Full bibliography entries: [`references/references.bib`](../../references/references.bib).
