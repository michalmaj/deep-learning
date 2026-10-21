Language: **English (default)** | [Polski](lecture.md)

# Lecture 05. Generalization, overfitting, and regularization

## Learning goals

1. distinguish empirical risk, minimized during training, from expected risk, which represents the actual objective, with unambiguously defined symbols,
2. recognize underfitting, a good fit, and overfitting on training and validation loss curves and justify the diagnosis with specific numbers,
3. derive the update rule for L2 regularization in plain SGD and state the exact condition under which it is identical to "weight decay",
4. explain why this identity does **not** have to hold for Adam, and how AdamW differs from it,
5. write inverted dropout mathematically (a Bernoulli mask, scaling) and explain the difference between `model.train()` and `model.eval()`,
6. implement early stopping that **restores** the best recorded model state.

## Why this matters

Since lecture 1, training has focused on reducing loss on the training set. A model that fits this data perfectly can still perform worse on new examples than one with a less exact fit. Optimization and generalization therefore become separate concerns. The first concerns reducing training loss quickly and stably, while the second concerns performance beyond the data used for learning. Proper evaluation requires a sound methodology rather than a single favorable run.

## Intuition

### The actual training objective

The goal is a model that performs well on new data from the same distribution as the training data. Training loss is only an **approximation** of this objective, computed from a finite sample. Better representation of the full distribution can lead to better generalization.

### Underfitting and overfitting

- **Underfitting.** The model is too simple or trained too briefly to capture the structure in the training data. Training and validation losses remain high and close to each other.
- **Overfitting.** The model fits not only the data's structure but also noise specific to the training sample. Training loss continues to decrease while validation loss stops improving and begins to rise.

**The relationship between model capacity and generalization quality is not monotonic** and often requires experimental verification. A larger model can overfit more, but this is not guaranteed. Small models also overfit on limited data, while large models can generalize well with sufficient data and suitable regularization. Model size alone does not determine the result.

### Why regularization

Regularization includes techniques that discourage behavior associated with overfitting. L2 and weight decay discourage large weights, dropout limits reliance on individual neurons, and early stopping prevents excessively long training. Each method introduces a **trade-off** that needs to be evaluated experimentally for the problem at hand.

## Formal model

Let $P$ be the unknown probability distribution that generates pairs $(x,y)$. Define

$$R(\theta) = \mathbb{E}_{(x,y)\sim P}\big[\ell(f(x;\theta), y)\big] \qquad \text{(expected risk)}$$

This is the model's average error over the *entire* data distribution. It cannot be computed directly because $P$ is unknown and only a finite sample is available.

$$\hat R(\theta) = \frac{1}{N}\sum_{i=1}^N \ell\big(f(x_i;\theta), y_i\big) \qquad \text{(empirical risk)}$$

This is $L(\theta)$ from lecture 1, the average loss on the finite sample $\{(x_i,y_i)\}_{i=1}^N$. Training minimizes $\hat R(\theta)$ as an **approximation** of $R(\theta)$. The approximation improves with more representative data and deteriorates when the model fits properties unique to the sample.

The **generalization gap** is $R(\theta) - \hat R(\theta)$. In practice, it is approximated by $L_{\text{val}}(\theta) - L_{\text{train}}(\theta)$. A growing gap alongside a decreasing training loss serves as the operational definition of overfitting in this lecture.

**Data splits** assign a different role to each subset. Training data fits the parameters, validation data compares models and hyperparameters and determines when to stop, and test data provides a **single** final evaluation of the selected model. Any use of the test set during decision-making makes it part of model selection and removes its role as an independent measure of generalization.

**Data leakage** occurs when information from validation, test, or future data influences training or model selection. One example is computing normalization statistics over the **entire** dataset before splitting rather than using only the training set. This inflates the estimate because the model indirectly receives information about evaluation data. The lab measures this effect directly.

## Mathematics

### L2 regularization and its effect on the gradient (SGD)

The objective function is extended with a penalty for large weights.

$$L_{\text{reg}}(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|_2^2 = L(\theta) + \frac{\lambda}{2}\sum_j \theta_j^2$$

where $\lambda \geq 0$ is the regularization strength (a hyperparameter). The gradient:

$$\nabla L_{\text{reg}}(\theta) = \nabla L(\theta) + \lambda\theta$$

Plain SGD's step with this gradient:

$$\theta \leftarrow \theta - \eta\big(\nabla L(\theta) + \lambda\theta\big) = \underbrace{(1-\eta\lambda)\theta}_{\text{weight "shrinkage"}} - \eta\nabla L(\theta)$$

This is the definition of **weight decay**. Each step multiplies the weights by $(1-\eta\lambda) < 1$, moving them toward zero, and then applies the ordinary gradient update. **For plain SGD, adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss and multiplying the weights by $(1-\eta\lambda)$ within the same step are exactly equivalent.** The notebook confirms agreement with `torch.optim.SGD(weight_decay=λ)` at `atol=1e-3`.

The equivalence requires one combined step. Applying a full gradient update $\theta'=\theta-\eta\nabla L(\theta)$ first and separate shrinkage $(1-\eta\lambda)\theta'$ afterward gives
$$(1-\eta\lambda)\theta'= (1-\eta\lambda)\theta - \eta(1-\eta\lambda)\nabla L(\theta),$$
which adds a factor $(1-\eta\lambda)$ to the gradient term. The difference is exactly $\eta^2\lambda\nabla L(\theta)$, so for $\lambda\neq0$ this is generally a **different** update. Its direction depends on the sign of $\nabla L(\theta)$.

### L2 and weight decay for adaptive optimizers

In Adam **with the `weight_decay` parameter**, as implemented by `torch.optim.Adam` by default, the L2 penalty enters the gradient **before** adaptive normalization.

$$\hat g_t = \nabla \ell(\theta_t) + \lambda\theta_t, \qquad m_t = \beta_1 m_{t-1} + (1-\beta_1)\hat g_t, \qquad v_t = \beta_2 v_{t-1}+(1-\beta_2)\hat g_t^2, \qquad \dots$$

The remaining operations follow lecture 4. Since $\hat g_t$ is divided by $\sqrt{\hat v_t}$, the mechanism that normalizes the task gradient also normalizes the L2 contribution. If the task gradient is approximately proportional to $\theta$, adding $\lambda\theta$ may leave Adam's normalized direction and step unchanged. Under this convention, the L2 penalty can therefore be **ineffective** despite a nonzero $\lambda$. The notebook demonstrates this case.

**AdamW** (Loshchilov and Hutter, 2019) **decouples** the penalty from adaptive normalization. The moments $m_t$ and $v_t$ are computed from the pure task gradient, while weight shrinkage is applied separately after normalization.

$$m_t, v_t \ \text{computed from } \nabla\ell(\theta_t) \text{ (without } \lambda\theta_t\text{)}, \qquad \theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\theta_t\right)$$

The contribution $\lambda\theta_t$ no longer depends on the task gradient's history or scale, as in SGD. For this reason, weight regularization with Adam typically uses `AdamW`, including in this lecture's experiment.

### Dropout, the Bernoulli mask, and *inverted dropout*

For a layer with activations $a\in\mathbb R^m$ and a drop probability $p\in(0,1)$, in **training** mode:

$$r_i \sim \text{Bernoulli}(1-p) \ \text{independently for each } i, \qquad \tilde a_i = \frac{r_i}{1-p}\, a_i$$

$r_i=0$ with probability $p$, which zeros the neuron for the current step without removing it from the architecture. A new mask is drawn on every forward pass. The probability of $r_i=1$ is $1-p$. Scaling by $\frac{1}{1-p}$, known as *inverted dropout* and implemented by `nn.Dropout`, ensures $\mathbb E[\tilde a_i] = a_i$. The expected output is therefore preserved despite random zeroing. The notebook confirms this over 10,000 samples.

In **evaluation** mode, `model.eval()`, $\tilde a_i = a_i$. No random drawing or scaling occurs. Since scaling is applied during training, inference remains an ordinary deterministic forward pass.

`model.train()` and `model.eval()` switch **the behavior of specific layers**, including dropout. They do not control gradient computation, which is handled separately through `torch.no_grad()` and `requires_grad`. `model.eval()` does not disable gradients automatically, as confirmed by the successful computation of `.grad` in the notebook.

### Early stopping as a procedure

After every epoch $e$ with validation loss $V_e$:

1. If $V_e < V_{\text{best}} - \delta$, where $\delta$ is the minimum meaningful improvement (*min_delta*), record $V_{\text{best}}\leftarrow V_e$, save the model's **full parameter state**, and reset the counter of epochs without improvement.
2. Otherwise, the counter is incremented. Training stops when it reaches `patience`.
3. After stopping or exhausting the epoch budget, the best saved state is **restored**. The model used afterward comes from epoch $e_{\text{best}}$, not the final training epoch.

Omitting the third step leaves the model in the usually worse state from the final epoch. Ending the loop early is not sufficient for correct early stopping.

## Worked example

Let $\theta_0=2$, $\eta=0.1$, and the deliberately large $\lambda=0.5$. For the task $\ell(\theta)=\frac12\theta^2$, $\nabla\ell(\theta)=\theta$, as in modules 1 and 4.

**SGD, without regularization:** $\theta_1 = 2 - 0.1\cdot 2 = 1.8$.

**SGD with L2/weight decay:** $\nabla L_{\text{reg}} = 2 + 0.5\cdot2 = 3.0$; $\theta_1 = 2 - 0.1\cdot3.0 = 1.7$. The equivalent expression $(1-\eta\lambda)\theta_0 - \eta\nabla\ell(\theta_0) = 0.95\cdot2 - 0.2 = 1.7$ gives the same value.

**Adam, `weight_decay=0.5` (the L2-in-the-gradient convention):**

$$\hat g_1 = \theta_0 + \lambda\theta_0 = 2+1=3.0 \quad\Rightarrow\quad m_1=0.1\cdot3.0=0.3,\ v_1=0.001\cdot9.0=0.009$$
$$\hat m_1 = 0.3/0.1=3.0,\quad \hat v_1=0.009/0.001=9.0 \quad\Rightarrow\quad \theta_1 = 2 - 0.1\cdot\frac{3.0}{\sqrt{9.0}} = 2-0.1 = 1.9$$

**For the same $\theta_0$, Adam without regularization, with $\lambda=0$, also gives $\theta_1=1.9$.** In that case, $\hat g_1=\theta_0$, $\hat m_1=2.0$, $\hat v_1=4.0$, and $\theta_1=2-0.1\cdot\frac{2.0}{\sqrt{4.0}}=1.9$. Since the task gradient is proportional to $\theta$, adding $\lambda\theta$ only rescales it. Adam's normalized direction $\hat m/\sqrt{\hat v}$ is insensitive to this uniform scaling. **The L2 penalty does not change this step.**

**AdamW, `weight_decay=0.5` (the decoupled convention):** the moment is computed from the pure $\nabla\ell(\theta_0)=2.0$, without $\lambda$. This gives $\hat m_1=2.0$, $\hat v_1=4.0$, and a normalized step $\hat m_1/\sqrt{\hat v_1}=1.0$.

$$\theta_1 = \theta_0 - \eta\left(1.0 + \lambda\theta_0\right) = 2 - 0.1\cdot(1.0+0.5\cdot2) = 2-0.1\cdot2.0 = 1.8$$

AdamW incorporates the regularization because it gives $1.8$ instead of the $1.9$ obtained without the penalty. Adam with `weight_decay` does not change the result in this step. The notebook verifies all four values, $1.8$, $1.7$, $1.9$, and $1.8$, against the corresponding `torch.optim` classes.

## Experiment

The notebook [`examples.ipynb`](examples.ipynb) contains three parts.

1. **Three capacities for the same problem.** Regression $y=\sin(3x)+0.3x+\varepsilon$ uses $N=30$ noisy training points and a dense validation set with $N=200$. The comparison includes a small MLP with underfitting, a medium MLP with a good fit, and a large `[128,128,128]` MLP with overfitting. The data and training procedure are identical.
2. **Regularization of the model prone to overfitting.** The comparison includes no regularization, `AdamW(weight_decay=...)`, `nn.Dropout`, early stopping, and excessively strong dropout that causes underfitting.
3. **Optional effect of the number of training examples.** The same model is trained with $N\in\{15,30,60,120,300\}$.

The first two parts were **checked across several random seeds**, not only on a single selected run.

## Observations

- **Underfitting.** The small model has high, similar training and validation losses, such as $\approx 0.5$ and $\approx 0.6$. It cannot fit even the training data.
- **A good fit.** The medium model reaches a low training loss and moderate validation loss. Near the end of training, validation loss remains close to its minimum.
- **Measured overfitting.** The large model with $33409$ parameters achieves the lowest training loss of the three, around $0.01$-$0.05$. Validation loss first decreases, reaches its minimum well before training ends, and then rises while training loss continues to fall. The size of this increase depends on the seed. In the tests, the final-loss to minimum-loss ratio ranged from about $1.1\times$ to over $8\times$. The direction of the effect remained stable, while its magnitude did not.
- **Weight decay with AdamW helps moderately.** With a suitable $\lambda$ on the order of $10^{-2}$, final validation loss decreased noticeably relative to the unregularized model. The size of the improvement depended on the seed.
- **Dropout was stronger and more consistent than weight decay.** At $p\approx0.1$, final validation loss approached or sometimes improved on the best result of the unregularized model. This observation applies to this problem and architecture, not to dropout in general.
- **Excessive dropout hurts.** At $p=0.5$, training loss increased because the model could no longer fit the training data. Regularization turned overfitting into underfitting.
- **Early stopping requires state restoration.** Loss at the stopping point was sometimes noticeably worse than at the best recorded epoch. The benefit appeared only after restoring the saved weights.
- **Dropout affects early-stopping settings.** Combining dropout with the same `patience` used for a model without dropout stopped training prematurely. Additional noise in the validation curve can make the same patience threshold too low. This is a real interaction between the methods, not an implementation bug.

## Common misconceptions

❌ A low training loss means good generalization.

✅ A low training loss only indicates a good fit to **this specific sample**, including its noise. In the experiment, the model with the lowest training loss had one of the higher validation losses.

❌ A high validation loss always means overfitting.

✅ A high validation loss can also result from underfitting, accompanied by a high training loss. The **relationship** between the two values determines the diagnosis. Two high and similar losses indicate underfitting, while a growing gap with low training loss indicates overfitting.

❌ Dropout permanently removes neurons from the architecture.

✅ Dropout draws a **new** zero mask at every forward pass in training mode. No neuron is removed permanently. It may be active again on the next step, and all neurons remain active in `eval()` mode.

❌ `model.eval()` automatically disables gradient computation.

✅ `model.eval()` changes the behavior of selected layers, including dropout. Gradient computation is controlled separately through `requires_grad` and `torch.no_grad()`. The notebook confirms that `.grad` can still be computed in `eval()` mode.

❌ Early stopping is simply picking the last training epoch.

✅ Early stopping ends training after no improvement for the number of epochs set by `patience` and **restores the weights from the best earlier epoch**. The final trained state is worse than or equal to the best saved state.

❌ The test set can be used to choose hyperparameters or the number of epochs, as long as the final result is still reported honestly.

✅ Using the test set to choose a model, $\lambda$, or epoch count makes it part of model selection. It then stops being an independent measure of generalization.

❌ Stronger regularization always gives a better result.

✅ At $p=0.5$, dropout produced a worse result than at $p=0.1$, and training loss stopped decreasing. Regularization strength has an optimum, so a larger value does not necessarily improve the result.

## Engineering implications

- **Cost of additional experiments.** Comparing several values of $\lambda$, $p$, or `patience` multiplies the cost of a single training run. In practice, tuning can take more time than final training.
- **Checkpoint storage.** Correct early stopping requires keeping the full state of the best model in memory or on disk. This is a meaningful cost for large models. The overwrite frequency and number of retained checkpoints also require explicit decisions.
- **Training and inference.** Dropout and batch normalization behave differently in `train()` and `eval()`. Failing to switch modes before evaluation or deployment can reduce quality without producing an explicit program error.
- **Data-leakage risk in a pipeline.** Normalization, feature selection, or augmentation fitted to statistics outside the training set produces unfairly optimistic results. The error is difficult to detect because the pipeline still runs normally.
- **Evaluation instability on small datasets.** The difference between $1.1\times$ and $8\times$ for the same configuration on different samples is not an artifact of the experiment. Small datasets introduce real variation that needs to be considered when interpreting a single result.
- **Distribution shift.** The mathematics presented here assumes that production data comes from the same distribution $P$ as training and validation data. Changes over time, new users, or a different context can violate this assumption. Even a well-regularized model may then lose quality. This is distinct from overfitting.

## Key points

- Empirical risk approximates expected risk. A growing gap between them indicates overfitting.
- Underfitting produces high and similar training and validation losses. With overfitting, low training loss separates from rising validation loss.
- L2 and weight decay are equivalent for SGD, but **not** for Adam. AdamW decouples the penalty from adaptive normalization.
- Inverted dropout scales activations by $1/(1-p)$ during training, allowing `eval()` to use an ordinary forward pass.
- `train()`/`eval()` and gradient computation are two independent mechanisms.
- Early stopping without restoring the best state gives no benefit at all.
- The test set is used once after all decisions have been made.
- No regularization method guarantees improvement. Each needs verification on the specific problem and with more than one random seed.

## Questions for students

1. Can low training loss and high validation loss alone distinguish overfitting from an error in validation, such as data from another distribution? What additional information is needed?
2. Why can the L2 penalty in Adam (`weight_decay`) fail to have an effect despite being present in the gradient? How does this relate to adaptive normalization from lecture 4?
3. What is the most likely error when an early-stopping implementation does not change the model's behavior after training?
4. Why should the best $\lambda$ not be selected from test-set results, even if the final report includes a test result?

## References

- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019 (arXiv:1711.05101).
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(56), 1929-1958.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/, chapter 5.2 (Capacity, Overfitting and Underfitting) and chapter 7 (Regularization for Deep Learning).
- PyTorch Contributors. *torch.nn.Dropout* documentation. https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- PyTorch Contributors. *torch.optim.AdamW* documentation. https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

Full bibliography entries are available in [`references/references.bib`](../../references/references.bib).
