Language: **English (default)** | [Polski](lecture.md)

# Lecture 04. Gradient descent, SGD, momentum, and Adam

## Learning goals

1. derive and correctly write the update rules for plain gradient descent, mini-batch SGD, momentum (in PyTorch's convention), and Adam (with bias correction), with every symbol unambiguously defined,
2. distinguish four objects that are easy to confuse: the loss gradient, the mini-batch gradient estimate, the optimizer state, and the parameter update vector,
3. interpret a loss function's contour plot and an optimizer trajectory to explain why an algorithm oscillates, diverges, or converges slowly,
4. distinguish asymptotic momentum stability from convergence speed and short-horizon behavior,
5. explain Adam's mechanism (adaptive per-parameter scaling) and point out that this is a **different** mechanism than momentum's damping of oscillations, not a "better version" of it,
6. explain why comparing optimizers at one, shared learning rate and declaring a winner can be misleading.

## Why this matters

Lecture 3 explained how to compute a gradient, so `backward()` no longer needs to be treated as a black box. A gradient, however, provides only a **direction**. The step length, the use of previous steps, and separate scaling for individual parameters are independent design choices. They determine whether a network learns, how long training takes, and whether the process remains stable. This lecture therefore focuses on the training procedure rather than the architecture. It also shows why a "better" optimizer can only be defined in relation to a specific problem.

## Intuition

### Optimization as a search across a loss landscape

The loss $L(\theta)$ can be viewed as a landscape whose height at point $\theta$ equals the loss value. Training moves downhill one step at a time. Only the local slope at the current point is available, not a map of the entire terrain. The gradient $\nabla L(\theta)$ points toward the **steepest increase**, as introduced in lecture 1, so optimization moves in the opposite direction.

Optimization algorithms differ in how they use that direction. They may take a longer or shorter step, retain information about earlier movement, or scale the step separately for each parameter.

### Why the landscape can be hard

Three common difficulties arise from the geometry of the loss.

- **Ill-conditioning.** The landscape is steep in one direction and flat in another, forming a narrow valley. A step suited to one direction is poorly suited to the other. This is the central geometry in the experiment.
- **Oscillation.** If the step is too large in the steep direction, the algorithm repeatedly crosses the bottom of the valley instead of settling into it.
- **Flat regions and saddle points.** High-dimensional, nonlinear neural-network losses may have regions with a near-zero gradient that are not minima. The gradient carries little information there even though lower values remain available. A separate experiment would require a larger nonlinear parameter space than the two-dimensional convex bowl used here. Momentum and adaptive scaling can also help in such regions, not only with oscillation.

### Why "one number" (the learning rate) is not enough

Plain gradient descent uses one fixed, global step $\eta$ for all parameters and directions. Momentum adds **memory**, so previous steps affect the current one. Adam adds **per-parameter adaptation**, scaling each parameter according to the history of its gradient magnitude. These are different approaches to the same problem, and neither is universally better.

## Formal model

Training repeatedly adjusts parameters $\theta$ to reduce $L(\theta) = \frac1N\sum_{i=1}^N \ell(f(x_i;\theta), y_i)$, as introduced in lecture 1. Four related objects need to be kept separate.

| Object | Notation | What it is |
|---|---|---|
| Full gradient | $\nabla L(\theta_t)$ | the derivative of the loss over the **entire** dataset at point $\theta_t$; rarely computed directly because of its cost |
| Gradient estimate | $\hat g_t$ | a gradient computed on a **mini-batch**, giving a cheap random approximation of $\nabla L(\theta_t)$ |
| Optimizer state | e.g. $v_t$ for momentum or $(m_t, v_t)$ for Adam | the algorithm's memory of earlier $\hat g_t$ values; stored separately from model parameters in `optimizer.state` |
| Update vector | $\Delta\theta_t = \theta_{t+1}-\theta_t$ | the value added to the parameters, constructed from $\hat g_t$, the optimizer state, and $\eta$ |

`backward()` computes $\hat g_t$ and stores it in `.grad`, as described in lecture 3. `optimizer.step()` uses $\hat g_t$ and the optimizer state to compute $\Delta\theta_t$ and apply it to $\theta$. These are two separate operations. In particular, `backward()` does not update the parameters.

## Mathematics

### Gradient descent (full batch)

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

$\eta > 0$ is the **learning rate** and the only hyperparameter in this rule. It was introduced in lecture 1 and is generalized here.

### Mini-batch SGD

Computing $\nabla L(\theta_t)$ over the full set of $N$ examples is expensive. A mini-batch $B_t$, usually with $|B_t| \ll N$, provides a cheaper estimate:

$$\hat g_t = \frac{1}{|B_t|}\sum_{i\in B_t} \nabla \ell\big(f(x_i;\theta_t), y_i\big), \qquad \theta_{t+1} = \theta_t - \eta\, \hat g_t$$

$\hat g_t$ is an **unbiased estimator** of $\nabla L(\theta_t)$. When $B_t$ is sampled uniformly, $\mathbb E[\hat g_t] = \nabla L(\theta_t)$. The estimate points in the correct direction on average, while an individual step has variance that **decreases** with batch size, $\text{Var}[\hat g_t] \propto 1/|B_t|$. A mini-batch therefore does not produce a different or inferior gradient. It produces a noisy estimate of the same gradient.

- **Full batch** ($|B_t|=N$) has no sampling noise, but each step requires a full pass over the data.
- **Stochastic optimization** ($|B_t|=1$) has maximum noise and the lowest cost per step.
- **Mini-batch optimization** is the usual compromise. A batch contains enough examples to limit noise but remains small enough to fit in memory and permit multiple steps per epoch.

### Momentum

**Convention.** Momentum is defined differently across textbooks. Some sources use the moving average $v_t = \beta v_{t-1} + (1-\beta)\hat g_t$, which resembles Adam's first moment below. **PyTorch's `torch.optim.SGD(momentum=...)` uses another convention.** The course follows the PyTorch definition because it matches the implementation. The worked example verifies it directly against `torch.optim.SGD`.

$$v_t = \beta v_{t-1} + \hat g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t \qquad (v_0 = 0,\ \text{so } v_1 = \hat g_1)$$

$v_t$ is the **optimizer state**, often described as velocity. The value $\beta\in[0,1)$, usually $0.9$, determines how long previous steps remain influential. A ball rolling across the loss landscape provides a useful analogy. Its inertia prevents it from stopping immediately when the slope changes, while the current gradient gradually corrects its motion. When successive $\hat g_t$ values point in the same direction, $v_t$ accumulates and movement accelerates. When their signs alternate, the contributions partially cancel and reduce oscillation. This effect is not automatic. With the same $\eta$ as plain SGD, a large $\beta$ can initially amplify oscillations instead of damping them.

### Adam

Adam (*Adaptive Moment Estimation*, Kingma and Ba, 2015) keeps **two** optimizer states per parameter. The first, $m_t$, is an exponential moving average of the gradient and is similar in spirit to momentum. The second, $v_t$, is an exponential moving average of the gradient's square and describes its typical scale. The symbol $v_t$ denotes a different object than in the momentum section, following the notation used by PyTorch and the original paper.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\hat g_t^2 \qquad (m_0=v_0=0)$$

$\beta_1$, whose default is $0.9$, controls memory of the **direction**. Unlike PyTorch SGD momentum, it uses a weighted average with the factor $1-\beta_1$. $\beta_2$ defaults to $0.999$ and therefore has a longer memory. It tracks the gradient's **scale** separately for each parameter. $v_t$ grows for consistently large gradients and decreases for small ones.

**Bias correction.** Since $m_0=v_0=0$, the first estimates are systematically pulled toward zero. For $t=1$, $m_1=(1-\beta_1)\hat g_1$, which equals only $10\%$ of $\hat g_1$ when $\beta_1=0.9$. Division by $(1-\beta^t)$ corrects this bias. The denominator is small during the first steps, so the correction is strong, and approaches $1$ as $t$ grows.

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

**Update:**

$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

$\varepsilon$, which defaults to $10^{-8}$, prevents division by zero when $\hat v_t$ is close to zero. Division by $\sqrt{\hat v_t}$ reduces the effective step for a parameter with historically large gradients and increases it for one with small gradients. This is the per-parameter adaptive scaling introduced earlier. SGD and momentum do not provide this mechanism.

## Worked example

The following example computes three update steps for $f(\theta)=\frac12\theta^2$, so $g(\theta)=\theta$, with $\theta_0=5$. All three algorithms deliberately use $\eta=0.5$ to expose their different behavior. In practice, a shared value is rarely suitable for every optimizer, as the experiment will show. Momentum uses $\beta=0.5$, while Adam uses $\beta_1=0.9$, $\beta_2=0.999$, and $\varepsilon=10^{-8}$.

**SGD:** $\theta_1 = 5 - 0.5\cdot5 = 2.5$; $\theta_2=2.5-0.5\cdot2.5=1.25$; $\theta_3=1.25-0.5\cdot1.25=0.625$.

**Momentum ($\beta=0.5$):**

| $t$ | $g_t=\theta_{t-1}$ | $v_t=\beta v_{t-1}+g_t$ | $\theta_t=\theta_{t-1}-\eta v_t$ |
|---|---|---|---|
| 1 | $5$ | $5$ | $5-0.5\cdot5=2.5$ |
| 2 | $2.5$ | $0.5\cdot5+2.5=5.0$ | $2.5-0.5\cdot5.0=0.0$ |
| 3 | $0$ | $0.5\cdot5.0+0=2.5$ | $0-0.5\cdot2.5=-1.25$ |

At step 3, the gradient at $\theta_2=0$ is zero because the parameter is exactly at the minimum. Nevertheless, $\theta$ **keeps moving** to $-1.25$ because $v_3$ retains the previous motion. This illustrates inertia directly. Momentum can **overshoot** a minimum even when the current gradient is zero.

**Adam ($\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$):**

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | $\theta_t$ |
|---|---|---|---|---|---|---|
| 1 | $5.000$ | $0.500$ | $0.025$ | $5.000$ | $25.000$ | $4.500$ |
| 2 | $4.500$ | $0.900$ | $0.045$ | $4.737$ | $22.624$ | $4.002$ |
| 3 | $4.002$ | $1.210$ | $0.061$ | $4.466$ | $20.419$ | $3.508$ |

The notebook contains the full derivation and matches `torch.optim.Adam` to six decimal places. At the same $\eta=0.5$, Adam follows $5\to4.5\to4.0\to3.5$ and descends more slowly than SGD. Once the estimates stabilize, $\hat m_t/\sqrt{\hat v_t}\approx \pm1$, so Adam's effective step remains close to $\eta$ regardless of the gradient's scale. In SGD, the step scales directly with the gradient magnitude. Neither behavior is inherently better. The mechanisms differ, so a suitable $\eta$ for Adam is usually not suitable for SGD, and vice versa.

## Experiment

The notebook [`examples.ipynb`](examples.ipynb) contains two experiments.

**1. Optimization geometry.** The function $f(x,y)=\frac12(25x^2+y^2)$ is steep in the $x$ direction and flat in the $y$ direction. Each trajectory starts at $(1,1)$, and the minimum is at $(0,0)$.

- SGD with several values of $\eta$ demonstrates the theoretical stability boundary $\eta<2/25=0.08$. It is stable but slow at $\eta=0.02$, oscillatory but convergent at $\eta=0.06$, and divergent at $\eta=0.1$.
- Momentum receives the **same** $\eta$ as oscillatory SGD. A value of $\beta=0.3$ damps oscillation and accelerates movement along the flat direction. With $\beta=0.9$, the method remains asymptotically stable but has a long, strongly oscillatory transient. A smaller $\eta$ performs better under a short step budget.
- Adam runs with several values of $\eta$. Its $x$ and $y$ trajectories are almost identical despite the 25-fold difference in curvature, illustrating adaptive scaling.

**2. Training a small MLP.** The synthetic dataset contains two interleaving half-moons and $N=2{,}000$ examples. It is generated locally, split 80/20 into training and validation sets, and arranged into mini-batches of 64. A dedicated function shuffles indices with a fixed seed.

- SGD, momentum, and Adam are compared at a **shared** $\eta=0.01$. This is a controlled experiment, not a general ranking.
- A second comparison uses a reasonable $\eta$ tuned **separately** for each optimizer.
- An SGD sweep shows the transition from stable training through instability to divergence.

## Observations

- The SGD stability boundary agrees with theory, $\eta=2/a=0.08$ for $f=\frac12 a x^2$ and $a=25$. At $\eta=0.06$, the $x$ trajectory oscillates, but the loss falls to $0.0035$ after 40 steps. At $\eta=0.1$, the loss grows exponentially and reaches the order of $10^{15}$ after 40 steps.
- Momentum with $\beta=0.3$ and $\eta=0.06$ damps oscillation in $x$ and accelerates convergence in $y$. It improves both aspects relative to SGD in this setting.
- Momentum with $\beta=0.9$ and the same $\eta=0.06$ performs poorly during the first 20 steps. The loss jumps between $\approx0.1$ and $\approx15$ and ends this interval at $\approx1.92$. A longer run reaches $\approx0.0066$ after 40 steps and $\approx0.000025$ after 100. The update matrix has spectral radius $\sqrt{0.9}\approx0.9487<1$ for both curvatures, so the configuration is asymptotically stable. The smaller $\eta=0.02$ is faster within a 40-step budget and reaches a loss of $0.003$, but it is not required for convergence. **A short, oscillatory run is not enough to establish instability.**
- Adam produces $x$ and $y$ trajectories that match to four decimal places despite the 25-fold curvature difference. With an excessively large $\eta$, from $0.3$ to $0.5$, it oscillates in both directions at once. This differs from the instability seen in SGD and momentum.
- At the shared $\eta=0.01$, Adam reaches training and validation losses of about $0.13/0.15$ after 30 epochs, compared with $0.25/0.30$ for momentum and $0.33/0.37$ for SGD. After separate tuning to $0.5$ for SGD, $0.1$ for momentum, and $0.01$ for Adam, the gap **largely disappears**. The respective results are about $0.14/0.17$, $0.13/0.15$, and $0.13/0.15$. Adam's initial advantage mainly reflected robustness to the choice of $\eta$.
- The SGD sweep on the MLP shows smooth convergence at $\eta\in\{1,3\}$, increasing instability at $\eta=8$-$10$, and clear non-convergence at $\eta\ge15$.

## Common misconceptions

❌ The gradient points to the best, finite step to take.

✅ The gradient indicates only the **direction** of the steepest **local** increase, as described in lecture 1. It does not determine the step length. SGD uses a fixed step, momentum adds memory, and Adam adapts the scale separately for each parameter.

❌ A larger learning rate always means faster effective learning.

✅ Once a boundary determined by the loss curvature and the algorithm is crossed, a larger $\eta$ causes oscillation or divergence. In the example, SGD at $\eta=0.1$ **diverges exponentially**, while at $\eta=0.06$ it still converges despite oscillation.

❌ Adam guarantees better generalization (or even just a better result) than SGD.

✅ In the geometry experiment, well-tuned SGD and momentum match or slightly outperform Adam when $\eta$ is selected separately. Adam's advantage at a shared $\eta$ on the MLP mainly reflected robustness to a poor value, not a guarantee of a better minimum or generalization. The result concerns one specific problem and is not a general rule.

❌ `optimizer.step()` computes the gradients.

✅ `backward()` computes the gradients and stores them in `.grad`, as described in lecture 3. `optimizer.step()` **reads** `.grad`, updates optimizer state such as $v_t$ and $m_t$, and then modifies the parameters. These are separate operations with different responsibilities.

❌ Using mini-batches means every step uses the same gradient as the full dataset.

✅ The mini-batch gradient $\hat g_t$ is a **random, unbiased estimator** of the full gradient $\nabla L(\theta_t)$. It points in the correct direction on average, while an individual step has variance that depends on batch size. A larger batch means less noise, not a different direction.

❌ A drop in training loss guarantees a drop in validation loss.

✅ These are separate quantities measured on different data. In the MLP experiment they move together because this short training run does not overfit. This is not a general guarantee. The mechanisms that make the curves diverge are discussed in lecture 5.

## Engineering implications

- **Memory cost of optimizer state.** SGD without momentum stores no additional state. SGD with momentum stores one tensor per parameter, $v_t$, which doubles the memory occupied by model parameters alone. Adam stores two tensors per parameter, $m_t$ and $v_t$, so this memory triples. For large models, this is a meaningful engineering cost.
- **Compute cost per step.** Adam performs more operations per step than SGD because it maintains two moving averages and computes a division and square root. The difference accumulates over millions of parameters and steps. The forward and backward passes usually remain more expensive than `optimizer.step()` itself.
- **Stability as a design cost.** A narrow safe range of $\eta$ for SGD on an ill-conditioned problem requires more tuning before the main training run. The same effect appears in the lab. Adam is sometimes selected for this practical reason rather than for a stronger convergence theory.
- **AdamW and learning-rate schedules.** Adam is often used in its AdamW variant, which implements weight decay differently. Lecture 5 returns to this subject during the discussion of regularization. A constant $\eta$ is also uncommon throughout an entire training run. It is typically reduced over time with a learning-rate scheduler. Both topics lie beyond this lecture's scope.

## What to remember

- A gradient provides a direction, while the optimizer determines how to use it. SGD takes a fixed step, momentum adds memory, and Adam adapts the step for each parameter.
- A mini-batch gradient is an unbiased but noisy estimator of the full gradient. It is not a different gradient.
- PyTorch's momentum convention, $v_t=\beta v_{t-1}+\hat g_t$, differs from the moving average used in some textbooks. The definitions should not be combined.
- Adam normalizes the step separately for each parameter using the history of gradient magnitudes. This mechanism differs from momentum's damping of oscillations.
- Momentum changes both the stability region and the rate at which oscillations decay. A learning rate needs to be assessed for a particular algorithm and over a sufficiently long horizon. Poor progress after a few steps does not necessarily indicate instability.
- Comparing optimizers at one shared $\eta$ mainly measures robustness to that choice, not which algorithm is generally better.

## Questions for students

1. Why, in step 3 of the momentum worked example, does the parameter keep moving, even though the gradient at that point is zero?
2. Why does Adam's bias correction have the largest effect during the first few steps and then practically disappear?
3. Why does Adam handle an ill-conditioned problem with very different curvature across directions, while plain SGD requires a carefully selected $\eta$?
4. Which hypothesis should be checked first when the loss grows exponentially from the beginning of training? Why this one?

## References

- Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015 (arXiv:1412.6980).
- Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). *On the importance of initialization and momentum in deep learning*. ICML 2013.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/, chapter 8 (Optimization for Training Deep Models).
- PyTorch Contributors. *torch.optim* documentation. https://pytorch.org/docs/stable/optim.html

Full bibliography entries are available in [`references/references.bib`](../../references/references.bib).
