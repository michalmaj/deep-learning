Language: **English (default)** | [Polski](lecture.md)

# Lecture 07. Transfer learning and reusable representations

## Learning goals

1. explain why features learned by a CNN on ImageNet can help with another image-classification task,
2. distinguish a frozen backbone, partial fine-tuning, and full fine-tuning,
3. use `requires_grad`, optimizer parameter groups, and `train()` and `eval()` modes correctly,
4. recognize the BatchNorm risk when freezing a model,
5. evaluate experimentally whether transfer learning helps with a given task.

## Why this matters

Training a CNN from scratch requires substantial data and computation. ImageNet contains 1.2 million images from 1000 classes, and training ResNet18 on a GPU takes many hours or days. Practical tasks often provide only a few hundred examples.

Transfer learning starts from weights learned on a large, general dataset and adapts them to a new problem. This approach is common in computer vision, but subtle implementation errors are easy to introduce. Optimizer parameters, model mode, and BatchNorm layers require particular care.

## Intuition

A CNN trained on ImageNet learns a hierarchy of features. Early layers respond to edges, contours, and simple textures that are useful across many image tasks. Deeper layers describe more complex patterns associated with the training dataset.

General features can often be reused without modification. Deeper layers and the classification head may require adaptation because the number and meaning of classes differ from ImageNet.

❌ A pretrained model always improves the result because it knows more than a model trained from scratch.

✅ The benefit depends on domain similarity. ImageNet mainly contains everyday photographs of objects and scenes. Its features may fit satellite, medical, or microscopy images less well. The effect must be measured rather than assumed.

## Backbone and head

A pretrained model can be written as a composition of two functions.

$$f(x;\theta) = f_{\text{head}}\Big(f_{\text{backbone}}(x;\theta_{\text{backbone}});\ \theta_{\text{head}}\Big), \qquad \theta = (\theta_{\text{backbone}}, \theta_{\text{head}})$$

- $f_{\text{backbone}}$ is the convolutional part of the network. In ResNet18 it contains `conv1`, `bn1`, `layer1..layer4`, and `avgpool`. An image $x\in\mathbb R^{3\times224\times224}$ is mapped to a vector $z\in\mathbb R^{512}$.
- $f_{\text{head}}$ is the classification head that maps the feature vector to $C$ logits. The default `nn.Linear(512, 1000)` must be replaced with a randomly initialized `nn.Linear(512, C)`.

The parameters of both parts are denoted by $\theta_{\text{backbone}}$ and $\theta_{\text{head}}$. The symbol $\varphi$ remains reserved for activation functions. Here $C$ denotes the number of classes, although it represents the number of channels in the NCHW convention. The context determines its meaning.

## Three adaptation strategies

Training minimizes the loss

$$L(\theta) = \frac1N\sum_{i=1}^N\ell(f(x_i;\theta), y_i).$$

The strategies differ in the set of parameters for which gradients are calculated and applied.

### 1. Frozen backbone

The backbone parameters retain their pretrained values and are excluded from optimization.

$$\min_{\theta_{\text{head}}} L(\theta_{\text{backbone}}^{\text{pretrained}}, \theta_{\text{head}}), \qquad \theta_{\text{backbone}} \text{ fixed, does not enter } \nabla$$

In PyTorch, backbone parameters receive `requires_grad=False`, and the optimizer receives only `model.fc.parameters()`. This strategy is also known as *linear probing*.

### 2. Partial fine-tuning

The backbone is divided into frozen and trainable parts. For example, `layer1..layer3` can remain frozen while `layer4` is fine-tuned.

$$\min_{\theta_{\text{backbone}}^{\text{unfrozen}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{\text{frozen}} = \theta_{\text{backbone}}^{\text{frozen,pretrained}} \text{ fixed}$$

### 3. Full fine-tuning

All parameters are trainable, but the backbone still begins with pretrained weights.

$$\min_{\theta_{\text{backbone}},\ \theta_{\text{head}}} L(\theta), \qquad \theta_{\text{backbone}}^{(0)} = \theta_{\text{backbone}}^{\text{pretrained}}$$

### Gradients and parameter groups

`loss.backward()` calculates a gradient for every parameter with `requires_grad=True`. The forward pass always traverses the complete model because the head needs the backbone output. Freezing therefore does not remove the forward-pass cost.

Different model sections can use separate learning rates through parameter groups.

```python
optimizer = torch.optim.Adam(
    [
        {"params": model.layer4.parameters(), "lr": eta_backbone},
        {"params": model.fc.parameters(), "lr": eta_head},
    ]
)
```

Typically, $\eta_{\text{backbone}}$ is $10$ to $100\times$ smaller than $\eta_{\text{head}}$. Pretrained weights provide a useful starting point but need not be optimal for the new task. A large update can disrupt them, especially before the randomly initialized head produces a useful signal. A common sequence trains the head first and then carefully unfreezes part of the backbone.

❌ Full fine-tuning always performs better than training only the head.

✅ More trainable parameters increase flexibility but also raise the risk of overfitting a small dataset. Validation should guide the choice.

## How many parameters are trained

ResNet18 with a head for $C=10$ classes has the following trainable parameter counts.

| Strategy | Trainable parameters | % of total |
|---|---:|---:|
| Full model (reference) | 11,181,642 | 100.00% |
| Frozen backbone, only `fc` | 5,130 | 0.05% |
| Partial fine-tuning, `layer4` + `fc` | 8,398,858 | 75.11% |
| Full fine-tuning | 11,181,642 | 100.00% |

The head has $512\times10 + 10 = 5{,}130$ parameters. `layer4` contains 8,393,728 parameters, giving 8,398,858 together with the head.

Inspecting `requires_grad` alone is not enough. Parameters can be compared before and after a real optimizer step.

```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
# ... one step: zero_grad -> forward -> backward -> step ...
for n, p in model.named_parameters():
    changed = not torch.equal(before[n], p.detach())
    assert changed == p.requires_grad, f"{n}: unexpected state"
```

The notebook confirms that no frozen tensor changes and every trainable tensor is updated after several steps.

### The special case of BatchNorm

ResNet18 contains 20 `BatchNorm2d` layers. In `model.train()` mode, all of them update their `running_mean` and `running_var` buffers regardless of `requires_grad`. This includes BatchNorm inside the frozen `layer1`. Only `model.eval()` stops these updates.

Consequently, `requires_grad=False` and `model.eval()` solve different problems. Confusing them is a common and difficult-to-detect error.

## Experiment

Notebook [`examples.en.ipynb`](examples.en.ipynb) compares three strategies on 600 images from 10 balanced Caltech101 classes. The 360/120/120 split uses classes with good ImageNet coverage.

1. ResNet18 trained from scratch with random initialization.
2. A frozen backbone implemented naively and with a feature cache.
3. Partial fine-tuning of `layer4` and `fc` with a lower $\eta_{\text{backbone}}$.

The notebook reports total and trainable parameters, losses, accuracy, and training and inference time. A comparison at the same number of epochs differs from one at the same computational cost.

## Observations

| Strategy | Trainable params | Validation accuracy | Training time | Inference/image |
|---|---:|---:|---:|---:|
| From scratch (8 epochs) | 11,181,642 (100%) | 0.65 | 86.5 s | 13.37 ms |
| Frozen backbone, naive (5 epochs) | 5,130 (0.05%) | 1.00 | 26.6 s | not measured |
| Frozen backbone, cached (5 epochs) | 5,130 | 1.00 | **7.75 s** | 11.63 ms |
| Frozen backbone, final (15 cached epochs) | 5,130 | 1.00 | 7.8 s | not measured |
| Partial fine-tuning, `layer4` + `fc` (8 epochs) | 8,398,858 (75%) | 1.00 | 51.0 s | 12.41 ms |

The frozen backbone reaches $1.00$ validation accuracy with $0.05\%$ trainable parameters. Training from scratch on the same 360 images reaches $0.65$. Across two seeds, the frozen backbone obtains $1.00/1.00$ validation and $0.983/1.00$ test accuracy. Partial fine-tuning obtains $1.00/1.00$ and $0.958/0.992$, respectively.

Partial fine-tuning provides no measurable quality improvement in this experiment. This does not make it generally useless. The selected classes are well represented in ImageNet, so the frozen features already separate them almost perfectly. In the EuroSAT lab, partial fine-tuning improves the result. Two experiments cannot establish a general relationship between domain shift and the benefit of fine-tuning.

The naive and cached variants reach the same result after 5 epochs. Their times are $26.6$ s and $7.75$ s, so caching gives a $3.4\times$ speedup. It is valid only while the backbone and preprocessing remain unchanged.

An additional experiment uses $6/18/36$ images per class. The frozen backbone reaches $0.992 \to 1.00 \to 1.00$, while training from scratch reaches $0.250 \to 0.517 \to 0.575$. The transfer-learning advantage is largest with little data. This measurement uses one seed and therefore does not test stability across runs.

## Common misconceptions

❌ `requires_grad=False` works the same way as `model.eval()`.

✅ `requires_grad` controls gradient calculation. `train()` and `eval()` change the forward behavior of layers such as BatchNorm and Dropout. Frozen weights can remain constant while BatchNorm statistics continue to change.

❌ `torch.no_grad()` and `torch.inference_mode()` are equivalent to freezing parameters.

✅ Both mechanisms disable construction of the computational graph within a section of code. They are suitable for inference and one-time feature extraction, but not for training the head. `inference_mode()` is more restrictive, and tensors created within it cannot be used for later differentiation.

❌ A frozen backbone means that the entire model has stopped learning.

✅ Only the backbone is frozen. The new head continues to learn and adapts the model to the new classes.

❌ A larger pretrained model is always better.

✅ A larger model may have higher ImageNet accuracy, but it also requires more memory and computation. Parameter count is not a reliable predictor of runtime.

❌ Lower training loss after fine-tuning means better generalization.

✅ Training loss measures fit to known data. Generalization is assessed on validation and test sets.

❌ Pretrained weights remove the need for validation and error analysis.

✅ They change the optimization starting point, but an honest data split, confusion matrix, and error analysis remain necessary.

❌ ImageNet representations transfer well to every image domain.

✅ A large domain shift can reduce the benefit and sometimes cause *negative transfer*. Transfer should therefore be compared with a suitable model trained from scratch.

## Engineering implications

- **Memory.** Freezing saves memory for backbone gradients and optimizer state, but the forward pass still creates activations.
- **Parameters and runtime are different measures.** MobileNetV3-Small has $4.6\times$ fewer parameters and $32\times$ fewer FLOPs than ResNet18, yet runs $4\times$ slower on the tested CPU. Depthwise-separable convolutions may use standard CPU kernels inefficiently.
- **Feature caching has a narrow use case.** It speeds up head training only with an unchanged backbone and deterministic preprocessing. Cached features become stale after a fine-tuning step.
- **Weight versions affect reproducibility.** ResNet18 weights occupy about 45 MB. `ResNet18_Weights.IMAGENET1K_V1` should be specified instead of the `DEFAULT` alias, whose meaning may change.
- **Quality, size, and cost require separate evaluation.** The choice depends on memory limits, training time, and inference latency.

## Key points

- Transfer learning combines a backbone with a new head. Strategies differ in which parameters are updated.
- `requires_grad`, `train()` and `eval()`, and `no_grad()` and `inference_mode()` serve different purposes.
- `requires_grad=False` does not stop BatchNorm statistics from updating in `train()` mode.
- Transfer effectiveness depends on domain similarity and must be validated on data.

## Questions for students

1. How do `model.layer1[0].conv1.weight` and `model.layer1[0].bn1.running_mean` behave after 5 `optimizer.step()` calls when the backbone has `requires_grad=False` and the model remains in `train()` mode?
2. Which strategy suits a task with 50 images of a new class? The answer should include at least two arguments beyond parameter count.
3. Why is a feature cache valid for a frozen backbone and stale after the first fine-tuning step?
4. How many trainable parameters does ResNet18 have with an `nn.Linear(512, 37)` head when `layer3`, `layer4`, and the head are unfrozen? `layer3` contains 2,099,712 parameters.

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS.
- PyTorch documentation. *torchvision.models.resnet18*. `https://pytorch.org/vision/stable/models/resnet.html`.
- PyTorch documentation. *Autograd mechanics, Locally disabling gradient computation*. `https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation`.
