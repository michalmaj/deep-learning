Language: **English (default)** | [Polski](lecture.md)

# Lecture 07 — Transfer learning: reusing previously learned representations

## Learning goals

1. Understand why representations learned by a CNN on a large dataset (ImageNet) are often useful on a completely different image-classification task.
2. Distinguish three strategies for adapting a pretrained model: a frozen backbone (*linear probing*), partial fine-tuning, full fine-tuning — and know when each makes sense.
3. Correctly manipulate `requires_grad`, optimizer parameter groups, and `train()`/`eval()` mode — and understand that these are **three different, independent** mechanisms.
4. Understand the particular BatchNorm risk during freezing and fine-tuning.
5. Avoid treating transfer learning as a universal way to get higher accuracy — be able to judge when it actually helps, and when it does not.

## Why this matters

Training a CNN from scratch (lecture 6) requires a lot of data and time — ImageNet has 1.2 million images and 1000 classes; a model like ResNet18 trained from scratch reaches good quality after days of training on a GPU. In engineering practice we rarely have that much data for our own, narrow task (e.g. 600 photos across 10 object categories). Transfer learning is the answer to this constraint: instead of training from scratch, we **start from weights trained on a large, general dataset** and adapt them (or only part of them) to the new task. This is one of the most commonly used patterns in practical computer vision — and one of the easiest to break through subtle implementation mistakes, as we will see in the BatchNorm section.

## Intuition

A CNN trained on ImageNet (lecture 6: convolutional layers, pooling, a growing receptive field) learns a hierarchy of features: early layers detect edges, contours, and simple texture patterns — **general** features, useful for practically any image-related task. Deeper layers detect increasingly complex patterns that are increasingly **specific to the training set** (shapes characteristic of specific ImageNet categories). The intuition behind transfer learning: early/general features can be **reused as-is**; what may need adapting are the deeper layers and (always) the final classification head, since the number and identity of classes in the new task differs from ImageNet.

❌ "A pretrained model will always improve the result, because it 'knows more' than a model trained from scratch."

✅ A pretrained model helps when the features learned on the source dataset (ImageNet: ground-level photos of objects, everyday scenes) are **genuinely useful** for the new domain. The more visually distant the new domain is from ImageNet (we will see this literally in the lab — satellite imagery), the smaller the guaranteed benefit. We check this experimentally, we do not assume it in advance.

## Formal model: backbone and head

We treat a pretrained model as a composition of two functions:

$$f(x;\theta) = f_{\text{head}}\Big(f_{\text{backbone}}(x;\theta_{\text{backbone}});\ \theta_{\text{head}}\Big), \qquad \theta = (\theta_{\text{backbone}}, \theta_{\text{head}})$$

- $f_{\text{backbone}}$ (the **backbone**, *feature extractor*) — the convolutional part of the network (for ResNet18: `conv1, bn1, layer1..layer4, avgpool`), maps an image $x\in\mathbb R^{3\times224\times224}$ to a feature vector $z = f_{\text{backbone}}(x) \in \mathbb R^{512}$.
- $f_{\text{head}}$ (the **classification head**) — the last layer (or a few layers), maps the feature vector to $C$ logits of the new task: $f_{\text{head}}: \mathbb R^{512}\to\mathbb R^{C}$. In ResNet18 this is by default a single `nn.Linear(512, 1000)` (for the original ImageNet) — **the first step of transfer learning is always to replace this layer** with `nn.Linear(512, C)` with a random initialization, since $C$ (the number of classes in your task) differs from 1000.

**A note on notation.** You might expect the head's parameters to be denoted $\varphi$ (seen in the literature). In this course $\varphi$ is already reserved for the **activation function** (lectures 2–4) — to avoid a collision, instead of a new Greek symbol we extend the existing $\theta$ = model parameters convention (lecture 1) with explicit indices: $\theta_{\text{backbone}}$, $\theta_{\text{head}}$. Similarly, $C$ (the number of classes in the new task) collides with $C$ = number of channels from the NCHW convention (lecture 6) — the context (the number of outputs of the classification head vs. the number of channels of an image tensor) unambiguously settles which $C$ is meant at a given point, so we keep both meanings rather than introducing a new symbol.

## Mathematics: three adaptation strategies

As always, training minimizes the loss $L(\theta) = \frac1N\sum_{i=1}^N\ell(f(x_i;\theta), y_i)$ (lecture 1). The difference between the strategies is **with respect to which subset of $\theta$ we compute and apply the gradient**.

**1. Frozen backbone (linear probing).** $\theta_{\text{backbone}}$ set to its pretrained value and **excluded** from optimization:

$$\min_{\theta_{\text{head}}} L(\theta_{\text{backbone}}^{\text{pretrained}}, \theta_{\text{head}}), \qquad \theta_{\text{backbone}} \text{ fixed, does not enter } \nabla$$

In PyTorch: `requires_grad=False` for all backbone parameters, the optimizer only receives `model.fc.parameters()`.

**2. Partial fine-tuning.** We split the backbone into a frozen part and an unfrozen part, $\theta_{\text{backbone}} = (\theta_{\text{backbone}}^{\text{frozen}}, \theta_{\text{backbone}}^{\text{unfrozen}})$ (e.g. `layer1..layer3` frozen, `layer4` unfrozen):

$$\min_{\theta_{\text{backbone}}^{\text{unfrozen}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{\text{frozen}} = \theta_{\text{backbone}}^{\text{frozen,pretrained}} \text{ fixed}$$

**3. Full fine-tuning.** Everything trainable, but the backbone's **initialization** is still pretrained (not random!):

$$\min_{\theta_{\text{backbone}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{(0)} = \theta_{\text{backbone}}^{\text{pretrained}} \text{ (a starting point, not a constant)}$$

**Which parameters get a gradient?** `loss.backward()` (lecture 3) computes $\partial L/\partial\theta_j$ for **every** $\theta_j$ with `requires_grad=True` — regardless of strategy, the forward pass always goes through the **entire** model (otherwise we could not compute the loss). What differs is which of the computed partial derivatives end up in `.grad` (PyTorch skips accumulation for `requires_grad=False`) and which parameters `optimizer.step()` receives. Freezing **does not automatically speed up the forward pass** — we will see this numerically in the experiment.

**Different learning rates for different parameter groups** — `torch.optim` supports this natively via *parameter groups*:

```python
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": eta_backbone},
    {"params": model.fc.parameters(), "lr": eta_head},
])
```

where typically $\eta_{\text{backbone}} \ll \eta_{\text{head}}$ (e.g. $10$–$100\times$ smaller).

**Why small steps on the backbone, not none or large ones?** From module 4: the loss is a landscape, the gradient points in a local direction of descent. The pretrained $\theta_{\text{backbone}}$ lies in some good region *for its original task* — for the new task this point is usually not optimal, but it is a **much better starting point** than random initialization (because many low/mid-level features — edges, textures, simple shapes — are useful regardless of the specific task). A large step, before the randomly initialized head has learned anything sensible, means propagating large, chaotic gradients deep into the backbone — a risk of moving away from this good point before anything has had a chance to "adjust" to it (informally: destroying previously learned features before the new head can benefit from them). Hence the typical order in practice (and in our experiment): first a frozen backbone + training the head alone, only then, carefully, a partial unfreezing with a small $\eta_{\text{backbone}}$. We do not repeat the full derivation of backpropagation here (lecture 3) — we use it as an already-established mechanism.

❌ "Fine-tuning the whole model will always give a better result than training the head alone, because the model has more to work with."

✅ Full fine-tuning has **more degrees of freedom to fit the small dataset** — which also means a higher risk of overfitting (module 5) with few examples. We check this experimentally, we do not assume it.

## Worked example: how many parameters are actually trained?

ResNet18 with its head replaced for $C=10$ classes:

| Strategy | Trainable parameters | % of total |
|---|---:|---:|
| Full model (reference) | 11,181,642 | 100.00% |
| Frozen backbone (only `fc`) | 5,130 | 0.05% |
| Partial fine-tuning (`layer4` + `fc`) | 8,398,858 | 75.11% |
| Full fine-tuning | 11,181,642 | 100.00% |

(The number 5,130 = $512\times10 + 10$ — the weights and biases of `nn.Linear(512, 10)`, exactly the formula for dense layers from module 2. The number 8,398,858 is ResNet18's `layer4` [8,393,728 — two `BasicBlock`s] plus the head.)

**It is not enough to count this by simply reading off `requires_grad=True` — you have to verify that the frozen parameters actually do not change, and the trainable ones actually do change, after a real optimizer step:**

```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
# ... one step: zero_grad -> forward -> backward -> step ...
for n, p in model.named_parameters():
    changed = not torch.equal(before[n], p.detach())
    assert changed == p.requires_grad, f"{n}: unexpected state"
```

Verified programmatically (the notebook repeats this on real training): **0 frozen tensors change** under each of the three strategies, **all trainable tensors do change** after a few `optimizer.step()` calls.

**A surprising discovery from this verification — BatchNorm.** ResNet18 has **20** `BatchNorm2d` layers. When the model is in `model.train()` mode, **all 20** of them update their `running_mean`/`running_var` buffers **regardless of `requires_grad`** — including layers deep inside the "frozen" `layer1`. Only `model.eval()` stops this. We unpack this in the *Common misconceptions* section below — this is one of the most common, silent bugs in transfer learning.

## Experiment

The notebook (`examples.en.ipynb`) compares, on the same task (Caltech101, 10 balanced classes with good ImageNet coverage: `airplanes, Motorbikes, elephant, kangaroo, grand_piano, car_side, dalmatian, butterfly, starfish, dolphin`, 600 images, split 360/120/120):

1. **From scratch** — ResNet18 with random initialization, full training.
2. **Frozen backbone** (linear probing) — two variants: naive (a full forward pass through the backbone every epoch) and with caching of the extracted features (extraction once, training the head on the saved $512$-dimensional vectors).
3. **Partial fine-tuning** (`layer4` + `fc`, a separate, lower $\eta_{\text{backbone}}$).

For each strategy: total/trainable parameters, training/validation loss, validation accuracy, measured training and inference time. A comparison **at the same number of epochs** is explicitly distinguished from a comparison **at the same compute time** — these are not equivalent conditions.

## What we observe

Actual results from the notebook (10 Caltech101 classes, 360 training images):

| Strategy | Trainable params | Validation accuracy | Training time | Inference/image |
|---|---:|---:|---:|---:|
| From scratch (8 epochs) | 11,181,642 (100%) | 0.65 | 86.5s | 13.37ms |
| Frozen backbone, naive (5 epochs) | 5,130 (0.05%) | 1.00 | 26.6s | -- |
| Frozen backbone, cached (5 epochs, same result) | 5,130 | 1.00 | **7.75s** | 11.63ms |
| Frozen backbone, final (15 epochs on cache) | 5,130 | 1.00 | 7.8s | -- |
| Partial fine-tuning, `layer4`+`fc` (8 epochs) | 8,398,858 (75%) | 1.00 | 51.0s | 12.41ms |

**Transfer learning gave a huge, unambiguous advantage here** — the frozen backbone reaches a validation accuracy of $1.00$ with $0.05\%$ trainable parameters, while full training from scratch on the same $360$ images reaches only $0.65$. This is not a one-run fluke: repeated on $2$ seeds, the frozen backbone (cache) gives $1.00/1.00$ validation accuracy and $0.983/1.00$ test accuracy, partial fine-tuning gives $1.00/1.00$ validation and $0.958/0.992$ test — stably close to the ceiling in both cases.

**Did partial fine-tuning give anything beyond the frozen backbone?** In this specific experiment — **no, within measurement error**: both strategies reach a validation accuracy of $1.00$ and very close test accuracy ($0.98$–$1.00$ for cache, $0.96$–$0.99$ for partial). Partial fine-tuning took $51.0$s (8 epochs) versus $7.8$s for the final cache version (15 epochs) — **this is a comparison at different epoch budgets and different numbers of updates, not at the same compute cost**, so the time difference alone is not a reliable conclusion about the relative computational efficiency of these two strategies per se (a fair "same epoch budget" comparison is below: naive counting vs. cache, both at 5 epochs). What *is* reliable here — because it does not depend on the epoch budget — is the lack of a measurable quality benefit: both strategies reach the same accuracy ceiling. This is **not** proof that partial fine-tuning is useless in general — it is a consequence of the chosen classes having very strong ImageNet coverage, so the frozen features already separate the problem almost perfectly (a ceiling effect). In the lab (EuroSAT satellite imagery, a domain visually farther from ImageNet) a different result was measured — partial fine-tuning gave a measurable improvement there. These two experiments (Caltech101 and EuroSAT) **are not enough to establish a general relationship** of "the bigger the domain shift, the bigger the benefit of fine-tuning" — they only show that in these two specific cases the result was different, which by itself disproves the claim "partial fine-tuning never/always helps."

**Cache vs. naive counting, at an identical number of epochs (5) and an identical result ($1.00$ accuracy)**: $26.6$s naive vs. $7.75$s with cache — **$3.4\times$ faster** (exactly what the notebook itself computes and prints), with no difference in quality (because the backbone really does not change).

**Effect of data size** (an additional experiment, $6/18/36$ images per class): the frozen backbone (cache) reaches $0.992 \to 1.00 \to 1.00$ — almost immediately close to the ceiling, even with only $6$ images per class. Training from scratch: $0.250 \to 0.517 \to 0.575$ — grows with the amount of data, but stays far behind even at the full $36$/class. The advantage of transfer learning is **largest precisely when data is scarce** — consistent with the intuition from the *Why this matters* section. **This specific experiment, unlike the strategy comparison above, uses a single random seed** — the large observed effect has not been checked for stability across multiple runs, as above.

## Common misconceptions

❌ "Freezing parameters (`requires_grad=False`) is the same as `model.eval()`."

✅ These are **two independent mechanisms**. `requires_grad=False` tells autograd: "do not compute or accumulate a gradient for this tensor" — it affects what happens **backward**. `model.eval()`/`model.train()` changes **the forward behavior of specific layers** (BatchNorm: whether to update running statistics or use the saved ones; Dropout: whether to randomly zero activations). You can freeze weights and leave the model in `train()` — the weights will not change, but **BatchNorm statistics will still drift** (verified above: 20/20 layers). You can also leave `requires_grad=True` and call the model under `torch.no_grad()` — then no gradients get built at all, regardless of what `requires_grad` says.

❌ "`torch.no_grad()` and `torch.inference_mode()` are the same as freezing parameters."

✅ These control **whether a computational graph for backpropagation gets built at all** in a given piece of code — regardless of the `requires_grad` of individual parameters. We use them for inference (we do not need gradients, saving memory and time) and for the one-off extraction of features into a cache (see below) — not during the training of the head itself, since there we actually **need** the gradient with respect to $\theta_{\text{head}}$. `torch.inference_mode()` is a stricter, newer version of `no_grad()` — tensors coming out of it are not suitable for further differentiation, so we use it only where we are sure nothing further will need to be differentiated through it (extracting features into a cache).

❌ "A frozen backbone means the whole model has stopped learning."

✅ The backbone has stopped learning — the head **is still learning**, and it carries the entire burden of adapting to the new task. A "frozen model" in practice always means "a frozen **part** of the model" — there is always something trainable (at least the new head, otherwise there would be nothing to optimize).

❌ "Fine-tuning all layers must give a better result than training the head alone."

✅ Checked experimentally in the notebook — with a small dataset, more degrees of freedom also means a higher risk of overfitting (module 5). The strategy that wins is the one with the best actual **validation** loss/accuracy in a given run, not the one with the most trainable parameters.

❌ "A bigger pretrained model is always a better choice."

✅ A bigger model (more layers/channels) usually means higher accuracy on ImageNet, but also a bigger cost in memory, inference, and training — in a constrained environment (CPU, time limit) this is a real trade-off, not an obvious choice. See also: a smaller model need not be faster (*Engineering implications* section).

❌ "A lower training loss after fine-tuning means better generalization."

✅ Just as in module 5: training loss/accuracy measures the fit to data the model has already seen. The only reliable signal of generalization is validation/test.

❌ "Since I am using pretrained weights, I can skip proper validation and error analysis."

✅ Pretrained weights change the optimization's starting point, they do not exempt you from anything related to evaluating the result — the confusion matrix, error analysis, and an honest train/val/test split (module 5) apply exactly the same way.

❌ "Representations learned on ImageNet transfer well to any image domain."

✅ ImageNet is mostly ground-level photos of objects and scenes. Visually distant domains (satellite imagery, medical images, microscopy) may benefit much less — this is called **domain shift**, and in an extreme case transfer can even **hurt** relative to a well-chosen model trained from scratch (*negative transfer*). We examine this directly in the lab (EuroSAT satellite imagery).

## Engineering implications

- **Memory.** Even with a frozen backbone you need to keep its activations in memory during the forward pass (needed for the backward pass through the head) — freezing saves memory on the backbone's **gradients and optimizer state** (Adam: an extra $2\times$ as much memory as the weights themselves, module 4), not on the activations themselves.
- **Compute cost vs. parameter count — again, not the same thing.** Measured in this module: MobileNetV3-Small has $4.6\times$ fewer parameters and $32\times$ fewer FLOPs than ResNet18, and yet on this CPU it is **$4\times$ slower** in practice (depthwise-separable convolutions make poor use of standard CPU kernels without special optimization). The same pattern as in modules 5 and 6: do not trust parameter count as a proxy for time.
- **A frozen backbone does not speed up training the way you might expect** — done naively, it still requires a full forward pass through the whole model every epoch. The speedup comes from **caching features** (extraction once, training the head on ready-made vectors) — but this only works **when the backbone does not change and preprocessing is deterministic** (the same transformation every time, with no random augmentation). Under partial/full fine-tuning the cache becomes stale after the very first optimizer step, because the backbone (and hence the features it produces) changes — the cache **does not replace** fine-tuning, it only speeds up a very specific, narrow case (a genuinely frozen backbone).
- **The size of downloaded weights and reproducibility.** Pretrained weights are an external binary file (ResNet18: ~45MB) downloaded from a specific source with a specific version — it is worth pinning it explicitly (`ResNet18_Weights.IMAGENET1K_V1`, not `DEFAULT`), since the `DEFAULT` alias may point to different weights in the future.
- **The quality/size/cost trade-off** is three-dimensional, not one-dimensional — the choice of strategy and architecture depends on what actually constrains you: training time, memory, or inference latency in production.

## What to remember

- Transfer learning = backbone + head, $\theta = (\theta_{\text{backbone}}, \theta_{\text{head}})$; the three strategies differ in which subset of $\theta$ gets a gradient and is updated.
- `requires_grad`, `model.train()`/`eval()`, `torch.no_grad()`/`inference_mode()` are **three independent mechanisms** — each controls something different.
- Freezing the backbone via `requires_grad=False` does **not** automatically stop BatchNorm statistics from updating in `train()` mode — `model.eval()` is needed for that.
- Transfer learning is not a guarantee of a better result — it depends on how similar the new task's domain is to the domain the model was trained on.

## Questions for students

1. You have a ResNet18 with a frozen backbone (`requires_grad=False` on everything but `fc`). The model is in `model.train()` mode. After 5 `optimizer.step()` calls, will `model.layer1[0].conv1.weight` change? Will `model.layer1[0].bn1.running_mean`? Justify your answer.
2. You have 50 images of a new class and a pretrained ResNet18. Would you choose a frozen backbone, partial, or full fine-tuning? What two arguments (not just "more parameters = better") justify your choice?
3. Why is a cache of extracted features correct for a frozen backbone, but becomes stale after the first fine-tuning step?
4. Compute the number of trainable parameters for a ResNet18 with an `nn.Linear(512, 37)` head (37 classes), when `layer3` and `layer4` are unfrozen together with the head. (Hint: `layer3` has 2,099,712 parameters.)

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. (the ResNet architecture)
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR. (the pretraining dataset)
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS. (generality vs. specificity of features by depth — the source of the early/deep layer intuition)
- PyTorch documentation: *torchvision.models.resnet18*, `https://pytorch.org/vision/stable/models/resnet.html`.
- PyTorch documentation: *Autograd mechanics — Locally disabling gradient computation*, `https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation`.
