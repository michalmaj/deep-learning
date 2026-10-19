Language: **English (default)** | [Polski](lecture.md)

# Lecture 05: generalization, overfitting, and regularization

## Learning goals

After this lecture, you should be able to:

1. distinguish empirical risk (what we actually minimize) from expected risk (what we actually care about), with unambiguously defined symbols,
2. recognize underfitting, a good fit, and overfitting on training and validation loss curves -- and justify that diagnosis with specific numbers, not an impression,
3. derive the update rule for L2 regularization in plain SGD and state the exact condition under which it is identical to "weight decay",
4. explain why this identity does **not** have to hold for Adam, and how AdamW differs from it,
5. write inverted dropout mathematically (a Bernoulli mask, scaling) and explain the difference between `model.train()` and `model.eval()`,
6. implement correct early stopping -- one that **restores** the best remembered model state, not just remembers it.

## Why this matters

Since lecture 1, we have been minimizing loss on the training set, silently assuming this is a good goal. This lecture challenges that assumption directly: a model that fits the training data perfectly can be a **worse** model than one that fits it less well. This is the moment in the course where "optimization" (lecture 4: how fast and stably to drive the training loss down) and "machine learning" (does what we learned work on new data) formally diverge -- and where you learn to look at this honestly, methodologically, rather than based on a single favorable run.

## Intuition

### What we actually want

We do not want a model that performs well on data it has already seen -- we want a model that performs well on data it has **not yet seen**, but which comes from the same "world" (the same probability distribution). The loss on the training set is only an **approximation** of what we care about, computed on a finite sample. The better this approximation matches reality, the better the model **generalizes**.

### Underfitting and overfitting

- **Underfitting**: the model is too simple (or trained too briefly) to capture even the structure present in the training data. Symptom: **high loss on both training and validation**, close to each other.
- **Overfitting**: the model has sufficient (or excess) capacity to fit not only the data's true structure, but also the noise of this specific training sample. Symptom: **the training loss keeps dropping, while the validation loss stops dropping and starts rising** -- the model "memorizes" the training examples, including their random noise, at the expense of general structure.

A key caveat we will return to empirically: **the relationship between model capacity and generalization quality is not monotonic** and cannot always be predicted without an experiment. A larger model **can** overfit more -- but need not; small models also overfit given a sufficiently small amount of data, and large models, given enough data and appropriate regularization, can generalize very well. Do not treat "small = safe, large = risky" as a law of nature.

### Why regularization

Regularization is a family of techniques that do **not** change what a model learns directly from the data's signal, but instead discourage behaviors typical of overfitting: excessively large weights (L2/weight decay), excessive reliance on specific, individual neurons (dropout), or training for too long (early stopping). None of them is a guarantee -- each is a **trade-off** that must be verified experimentally on the specific problem, not assumed in advance.

## Formal model

Let $P$ be the (unknown) probability distribution that pairs $(x,y)$ come from. We define:

$$R(\theta) = \mathbb{E}_{(x,y)\sim P}\big[\ell(f(x;\theta), y)\big] \qquad \text{(expected risk)}$$

This is **what we actually care about** -- the model's average error over the *entire* data distribution, not just the data we have. We cannot compute it directly (we do not know $P$, we only have a finite sample).

$$\hat R(\theta) = \frac{1}{N}\sum_{i=1}^N \ell\big(f(x_i;\theta), y_i\big) \qquad \text{(empirical risk)}$$

This is exactly the $L(\theta)$ from lecture 1 -- the average loss on **our specific, finite sample** $\{(x_i,y_i)\}_{i=1}^N$. Training minimizes $\hat R(\theta)$ as an **approximation** of $R(\theta)$ -- this approximation is better the more representative data we have, and worse the more the model "exploits" the fact that $\hat R$ and $R$ are not the same function.

**Generalization gap**: $R(\theta) - \hat R(\theta)$ -- in practice approximated by $L_{\text{val}}(\theta) - L_{\text{train}}(\theta)$ (the loss on a validation set the model did not see during training, minus the training loss). A growing gap alongside a decreasing training loss is the operational definition of overfitting we use in this lecture.

**Data splits and their roles** (a reminder and a refinement): **training** -- fitting the parameters; **validation** -- comparing models/hyperparameters and (in this lecture) deciding when to stop training; **test** -- a **single**, final evaluation of the already-chosen model. Using the test set to make any decision (which model, which hyperparameter, which epoch) makes it part of the selection process -- and invalidates it as a measure of generalization, for exactly the same reason the training loss is not such a measure.

**Data leakage**: any situation where information from the validation/test set (or from the future, in time-series data) influences training or model selection -- e.g. computing the mean/standard deviation for normalization over the **entire** dataset before splitting, instead of only on the training set. The effect: an inflated, dishonest estimate of generalization -- the model "saw" (indirectly, through statistics) the data we are evaluating it on. In the lab you will measure this, not just read about it.

## Mathematics

### L2 regularization and its effect on the gradient (SGD)

We extend the objective function with a penalty for large weights:

$$L_{\text{reg}}(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|_2^2 = L(\theta) + \frac{\lambda}{2}\sum_j \theta_j^2$$

where $\lambda \geq 0$ is the regularization strength (a hyperparameter). The gradient:

$$\nabla L_{\text{reg}}(\theta) = \nabla L(\theta) + \lambda\theta$$

Plain SGD's step with this gradient:

$$\theta \leftarrow \theta - \eta\big(\nabla L(\theta) + \lambda\theta\big) = \underbrace{(1-\eta\lambda)\theta}_{\text{weight "shrinkage"}} - \eta\nabla L(\theta)$$

This is exactly what we call **weight decay**: every step first multiplies the weights by $(1-\eta\lambda) < 1$ ("shrinking" them toward zero), then performs the ordinary gradient step. **For plain SGD, these two formulations -- "add $\frac{\lambda}{2}\|\theta\|^2$ to the loss" and "multiply the weights by $(1-\eta\lambda)$ at every step" -- are exactly the same formula**, not two similar techniques (verified within numerical tolerance in the notebook, `atol=1e-3`, against `torch.optim.SGD(weight_decay=λ)`). Note that this must be **one, single step**: doing "first a full gradient step $\theta'=\theta-\eta\nabla L(\theta)$, and then a separate shrinkage $(1-\eta\lambda)\theta'$" gives
$$(1-\eta\lambda)\theta'= (1-\eta\lambda)\theta - \eta(1-\eta\lambda)\nabla L(\theta),$$
which differs from the formula above by an extra factor $(1-\eta\lambda)$ on the gradient term -- the difference is exactly $\eta^2\lambda\nabla L(\theta)$, so for $\lambda\neq0$ this is in general a **different** result than the one, combined step, and whether it is larger or smaller depends on the sign of $\nabla L(\theta)$ (it is not universally "a smaller result" in either direction).

### L2 vs. weight decay for adaptive optimizers -- WARNING, these are not the same

For Adam **with the `weight_decay` parameter** (as `torch.optim.Adam` implements it by default), the L2 penalty enters the gradient **before** the adaptive normalization:

$$\hat g_t = \nabla \ell(\theta_t) + \lambda\theta_t, \qquad m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1}+(1-\beta_2)\hat g_t^2, \qquad \dots$$

(the rest exactly as in lecture 4). The problem: $\hat g_t$ then passes through division by $\sqrt{\hat v_t}$ -- **the same mechanism that normalizes the task gradient's scale also normalizes the L2 penalty's contribution**. Consequence, verified in the notebook: if the task gradient is already approximately proportional to $\theta$ (which happens sometimes, and which the worked example below demonstrates exactly), adding $\lambda\theta$ to it **does not change at all** Adam's normalized direction/step -- in this convention, the L2 penalty can in practice be **ineffective**, despite a nonzero $\lambda$.

**AdamW** (Loshchilov and Hutter, 2019) fixes this by **decoupling** the penalty from the adaptive normalization -- the moments $m_t, v_t$ are computed from the **pure** task gradient, and the weight shrinkage is applied separately, after normalization:

$$m_t, v_t \ \text{computed from } \nabla\ell(\theta_t) \text{ (without } \lambda\theta_t\text{)}, \qquad \theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\theta_t\right)$$

Now $\lambda\theta_t$ acts exactly the same way, regardless of the task gradient's history/scale -- predictably, as in SGD. This is the reason why in practice (and in this lecture's experiment), `AdamW`, not `Adam(weight_decay=...)`, is used to regularize weights with Adam.

### Dropout: the Bernoulli mask and *inverted dropout*

For a layer with activations $a\in\mathbb R^m$ and a drop probability $p\in(0,1)$, in **training** mode:

$$r_i \sim \text{Bernoulli}(1-p) \ \text{independently for each } i, \qquad \tilde a_i = \frac{r_i}{1-p}\, a_i$$

$r_i=0$ with probability $p$ (the neuron is "switched off" at this step -- zeroed out, **not** removed from the architecture: at the next step it may be active again, the mask is redrawn on every forward pass), $r_i=1$ with probability $1-p$. Scaling by $\frac{1}{1-p}$ (*inverted dropout* -- the version `nn.Dropout` implements) ensures $\mathbb E[\tilde a_i] = a_i$: the output's expected value is preserved, despite the random zeroing (verified in the notebook over 10,000 samples).

In **evaluation** mode (`model.eval()`): $\tilde a_i = a_i$ -- no random drawing, no scaling. Because the scaling was already performed during training (hence *inverted*), inference is an ordinary, deterministic forward pass -- nothing extra needs to be done.

`model.train()` / `model.eval()` is a switch for **the behavior of specific layers** (here: dropout), **not** a switch for computing gradients -- that is a separate mechanism (`torch.no_grad()` / `requires_grad`, lecture 3). `model.eval()` does not automatically disable gradients (verified in the notebook: `.grad` is computed normally in `eval()` mode).

### Early stopping as a procedure

After every epoch $e$ with validation loss $V_e$:

1. if $V_e < V_{\text{best}} - \delta$ (a minimal meaningful improvement $\delta$, *min\_delta*): remember $V_{\text{best}}\leftarrow V_e$, remember the **full parameter state** of the model (not just the epoch number!), reset the counter of epochs without improvement;
2. otherwise: increment the counter of epochs without improvement; if the counter reaches `patience`: **stop training**;
3. after stopping (or after exhausting the epoch budget): **restore** the remembered best parameter state -- the model you go on to use comes from epoch $e_{\text{best}}$, not from the last trained epoch.

Step 3 is the one most often skipped (hence the misconception in the section below) -- without it, "early stopping" is only an earlier stop to the loop, with no benefit at all, because the model stays in the (worse) state from the last epoch anyway.

## Worked example

$\theta_0=2$, $\eta=0.1$, $\lambda=0.5$ (large, for clarity), the task $\ell(\theta)=\frac12\theta^2$ (so $\nabla\ell(\theta)=\theta$) -- the same task-gradient formula as in modules 1/4.

**SGD, without regularization:** $\theta_1 = 2 - 0.1\cdot 2 = 1.8$.

**SGD with L2/weight decay:** $\nabla L_{\text{reg}} = 2 + 0.5\cdot2 = 3.0$; $\theta_1 = 2 - 0.1\cdot3.0 = 1.7$. (Equivalently: $(1-\eta\lambda)\theta_0 - \eta\nabla\ell(\theta_0) = 0.95\cdot2 - 0.2 = 1.7$ -- the same number, from the second formula).

**Adam, `weight_decay=0.5` (the L2-in-the-gradient convention):**

$$\hat g_1 = \theta_0 + \lambda\theta_0 = 2+1=3.0 \quad\Rightarrow\quad m_1=0.1\cdot3.0=0.3,\ v_1=0.001\cdot9.0=0.009$$
$$\hat m_1 = 0.3/0.1=3.0,\quad \hat v_1=0.009/0.001=9.0 \quad\Rightarrow\quad \theta_1 = 2 - 0.1\cdot\frac{3.0}{\sqrt{9.0}} = 2-0.1 = 1.9$$

**Surprising, but verified: for the same $\theta_0$, Adam *without* regularization ($\lambda=0$) gives exactly the same $\theta_1=1.9$** -- because $\hat g_1=\theta_0$ (without $\lambda$), $\hat m_1=2.0$, $\hat v_1=4.0$, $\theta_1=2-0.1\cdot\frac{2.0}{\sqrt{4.0}}=2-0.1=1.9$. When the task gradient is (as here) proportional to $\theta$, adding $\lambda\theta$ only rescales it -- and Adam's normalized direction ($\hat m/\sqrt{\hat v}$) is insensitive to a uniform rescaling of the gradient. **The L2 penalty in this convention, for this step, did literally nothing.**

**AdamW, `weight_decay=0.5` (the decoupled convention):** the moment computed from the pure $\nabla\ell(\theta_0)=2.0$ (without $\lambda$): $\hat m_1=2.0$, $\hat v_1=4.0$, normalized step $=\hat m_1/\sqrt{\hat v_1}=1.0$.

$$\theta_1 = \theta_0 - \eta\left(1.0 + \lambda\theta_0\right) = 2 - 0.1\cdot(1.0+0.5\cdot2) = 2-0.1\cdot2.0 = 1.8$$

AdamW **sees** the regularization ($1.8 \neq$ AdamW without regularization, which would give $\theta_1=1.9$ -- the same arithmetic as Adam without $\lambda$, since without the penalty the two formulas coincide) -- Adam with `weight_decay` does not see it at all in this step. All four numbers ($1.8$; $1.7$; $1.9$; $1.8$) are verified in the notebook against the corresponding `torch.optim` classes.

## Experiment

The notebook [`examples.ipynb`](examples.ipynb):

1. **Three capacities, the same problem** -- regression $y=\sin(3x)+0.3x+\varepsilon$, $N=30$ noisy training points, a dense validation set ($N=200$) from the same distribution. A small MLP (underfitting), a medium MLP (a good fit), a large MLP `[128,128,128]` (overfitting) -- identical data, identical training procedure.
2. **Regularization on the model from (1) that overfit** -- without regularization, with `AdamW(weight_decay=...)`, with `nn.Dropout`, with early stopping -- plus an example of too-strong dropout leading to underfitting.
3. **(Optional, if time permits) The effect of the number of training examples** -- the same model, $N\in\{15,30,60,120,300\}$.

Each of sections 1-2 is **verified across several random seeds** before writing the conclusions below -- not just on a single, chosen run.

## What we observe

- **Underfitting**: the small model has a high training **and** validation loss, close to each other (e.g. $\approx 0.5$ and $\approx 0.6$) -- the model does not have the means to fit even the training data.
- **A good fit**: the medium model has a low training loss, a moderate validation loss, and the validation loss is still close to its minimum toward the end of training -- there is no clear upward bounce.
- **Overfitting, measured, not assumed**: the large model ($33409$ parameters) achieves the lowest training loss ($\approx 0.01$-$0.05$) of all three -- but the validation loss **first drops, reaches a minimum clearly before the end of training, and then rises**, even though the training loss keeps decreasing. The exact scale of this bounce depends on the random seed (checked: in our tests the final-loss/minimum-loss ratio ranged from $\approx 1.1\times$ to over $8\times$ depending on the specific data sample) -- **the direction of the effect is stable, its exact magnitude is not**, and that is how we present it.
- **Weight decay (AdamW) helps, moderately**: in our tests, with a well-chosen $\lambda$ (on the order of $10^{-2}$), the final validation loss dropped noticeably relative to the model without regularization, though the size of the improvement depended on the seed.
- **Dropout helped more strongly and more consistently than weight decay in our tests** (at $p\approx0.1$, the final validation loss dropped to a level close to, and sometimes below, the unregularized model's best moment) -- this is a specific observation *for this problem and this architecture*, not a general law "dropout > weight decay".
- **Too strong dropout hurts**: at $p=0.5$ the training loss rose noticeably (the model stopped being able to fit even the training data) -- regularization turned overfitting into underfitting.
- **Early stopping works, as long as it actually restores the state** -- stopping by itself gives nothing (the loss at the moment of stopping was sometimes noticeably worse than the loss at the best remembered epoch); only restoring the remembered weights gives the benefit.
- **An interaction we did not assume in advance**: combining dropout + early stopping with the same `patience` as for the model without dropout stopped training **prematurely** -- dropout introduces extra epoch-to-epoch noise into the validation curve, so the same patience threshold can end up too low. This is not an implementation bug -- it is a real interaction between two techniques that must be accounted for in practice.

## Common misconceptions

❌ A low training loss means good generalization.

✅ A low training loss only means the model fit **this specific sample** well -- including its noise. In our experiment, the model with the lowest training loss had one of the worse validation losses.

❌ A high validation loss always means overfitting.

✅ A high validation loss can also mean underfitting (the model too simple, the training loss also high) -- what distinguishes them is exactly the **relationship** between training and validation loss (close to each other and both high = underfitting; diverging, training low = overfitting), not the validation loss value alone, in isolation from the training loss.

❌ Dropout permanently removes neurons from the architecture.

✅ Dropout draws a **new** zero mask at every forward pass in training mode -- no neuron is removed permanently; at the next step (and always in `eval()` mode) it is fully active again.

❌ `model.eval()` automatically disables gradient computation.

✅ `model.eval()` only changes the behavior of specific layers (dropout, and others not yet covered in this course) -- gradient computation is controlled by a separate mechanism (`requires_grad`, `torch.no_grad()`, lecture 3), independent of `train`/`eval` mode (verified: in `eval()`, `.grad` is still computed).

❌ Early stopping is simply picking the last training epoch.

✅ Early stopping is stopping training **after** detecting no improvement for `patience` epochs, **and restoring the weights from the best, earlier epoch** -- the last trained epoch is by definition worse (or at best equal to) the one remembered as best.

❌ The test set can be used to choose hyperparameters or the number of epochs, as long as the final result is still reported honestly.

✅ Any use of the test set to make a decision (which model, which $\lambda$, which epoch) makes it part of the model-selection process -- it stops being an independent measure of generalization, for exactly the same reason the training loss is not one.

❌ Stronger regularization always gives a better result.

✅ Measured directly: at $p=0.5$, dropout made the result worse than at $p=0.1$ (and worse than the training loss itself, which stopped dropping) -- regularization has an optimum, it is not "the more, the better".

## Engineering implications

- **The cost of extra experiments**: comparing several values of $\lambda$/$p$/`patience` is a multiple of the cost of a single training run -- in practice this is often the project's largest time cost, larger than the final training run itself.
- **Storing checkpoints**: correct early stopping requires keeping the full state of the best model in memory (or on disk) during training, not just a metric -- for large models this is a real memory/disk cost that must be decided on deliberately (how often to overwrite the checkpoint, whether to keep more than one).
- **Training vs. inference**: dropout (and, beyond the scope of this lecture, batch normalization) behave **differently** in `train()` and `eval()` -- forgetting to switch mode before evaluation/deployment is a common, silent bug (the model keeps "working", just worse, and in a way that is not immediately easy to diagnose).
- **The risk of data leakage in a pipeline**: normalization, feature selection, or augmentation fitted to statistics from data outside the training set is one of the most common, hardest-to-detect bugs in practical ML projects -- because the model "works" and produces numbers, only those numbers are dishonestly optimistic.
- **Evaluation instability with small datasets**: the sensitivity to the random seed measured in this lecture (a difference between $1.1\times$ and $8\times$ for the same configuration, a different data sample) is not an artifact of our experiment -- it is a real property of small datasets that must be accounted for when interpreting a single result in practice.
- **Distribution shift**: all the generalization mathematics in this lecture assumes production data comes from the same distribution $P$ as the training/validation data. If that assumption stops holding (data changes over time, new users, a different context), none of these guarantees -- even with ideal regularization -- necessarily carry over. This is a separate problem from overfitting, often confused with it.

## What to remember

- We minimize empirical risk as an approximation of expected risk -- the gap between them is overfitting.
- Underfitting: high training and validation loss, close to each other. Overfitting: diverging, training low.
- L2 and weight decay are the same formula for SGD -- but **not** for Adam; AdamW fixes this by decoupling the penalty from the adaptive normalization.
- Inverted dropout scales activations by $1/(1-p)$ during training, so `eval()` can be an ordinary pass with no changes.
- `train()`/`eval()` and gradient computation are two independent mechanisms.
- Early stopping without restoring the best state gives no benefit at all.
- We touch the test set once, after all decisions -- otherwise it stops measuring what it is supposed to measure.
- No regularization method is a guarantee -- each must be verified on the specific problem, with more than one random seed.

## Questions for students

1. You have a model with a low training loss and a high validation loss. How would you distinguish, from these two numbers alone, whether this is overfitting or, say, a bug in the code computing the validation loss (e.g. the validation data is from a different distribution)?
2. Why can the L2 penalty in Adam (`weight_decay`) fail to have an effect, even though it is mathematically "present" in the gradient? Answer by referring to the adaptive-normalization mechanism from lecture 4.
3. You implemented early stopping, but after training the model behaves exactly as if it were not there. What is the most likely bug?
4. Why can you not choose the best $\lambda$ (weight decay) based on the test-set result, even if you ultimately still report the test result in your write-up?

## References

- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019 (arXiv:1711.05101).
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(56), 1929-1958.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/ -- chapter 5.2 (Capacity, Overfitting and Underfitting) and chapter 7 (Regularization for Deep Learning).
- PyTorch Contributors. *torch.nn.Dropout* documentation. https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- PyTorch Contributors. *torch.optim.AdamW* documentation. https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

Full bibliography entries: [`references/references.bib`](../../references/references.bib).
