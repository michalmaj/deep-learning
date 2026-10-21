Language: **English (default)** | [Polski](lecture.md)

# Lecture 06. From pixels to features in convolutional neural networks

## Learning goals

1. write an image as a tensor in the NCHW convention and explain what each of the four dimensions means,
2. derive and calculate one feature-map value by hand, then verify it against `torch.nn.Conv2d`,
3. calculate the output size of a convolutional layer for a given kernel size, stride, and padding, then compare its parameter count with a fully connected layer,
4. distinguish translation equivariance from translation invariance and identify which of these properties convolution actually provides, and which pooling only partially provides,
5. build a simple CNN architecture with feature extraction, aggregation, and a classification head, and explain the role of each part,
6. interpret learned filters and feature maps without attributing more meaning to them than the data actually supports.

## Why this matters

Earlier models treated each input as a vector without a built-in spatial structure. An MLP does not distinguish adjacent pixels from pixels at opposite corners of an image. This is inefficient because an edge, corner, or texture can occur anywhere and should not require separate learning at every position. A CNN incorporates the spatial neighborhood of pixels directly into the architecture.

## Intuition

### Why flattening an image is often inefficient

A $28\times28$ image flattened into a vector has $784$ elements. An MLP whose first hidden layer contains $128$ neurons needs $784\times128 + 128 = 100{,}480$ parameters for that layer alone. The same pattern in different parts of the image occupies unrelated vector coordinates, so the network must learn it separately for each location. Flattening also removes the direct representation of two-dimensional neighborhoods. Pixels adjacent in a column become `width` positions apart in the vector.

### Locality and weight sharing

A CNN incorporates two assumptions, known as *inductive biases*, about image structure.

- **Locality.** The presence of an edge depends primarily on nearby pixels. A convolutional filter therefore examines a small window, such as $3\times3$, rather than the entire image at once.
- **Weight sharing.** The same filter is applied at every image position. A filter that detects a vertical edge in one corner can detect it elsewhere because each position uses the same weights.

Weight sharing produces **translation equivariance**. Shifting an input pattern shifts the feature map by the same amount. The convolution responds to the local pattern rather than its absolute position.

❌ **This is not translation invariance.** Invariance would keep the output of the entire network unchanged after a pattern moves. Convolution alone produces a shifted feature map instead. Pooling adds some local tolerance by aggregating nearby values, but it does not guarantee invariance to large shifts. Additional pooling layers can increase tolerance without making the network fully invariant.

## Formal model

**An image as an NCHW tensor.** PyTorch uses tensors with shape $(N, C, H, W)$. $N$ is the batch size, $C$ is the number of channels, $H$ is the height, and $W$ is the width. Grayscale images have one channel and RGB images have three. A batch of 64 RGB images sized $32\times32$ has shape $(64, 3, 32, 32)$.

**A convolutional layer** transforms a tensor $(N, C_{in}, H, W)$ into a tensor $(N, C_{out}, H', W')$ using $C_{out}$ filters, each of shape $(C_{in}, K, K)$ (assuming a square $K\times K$ filter) plus a bias. This is exactly `torch.nn.Conv2d(C_in, C_out, kernel_size=K, ...)`.

## Mathematics

### One feature-map value in the single-channel case

For an image $X$ (an $H\times W$ matrix) and a filter $K$ (a $k\times k$ matrix), the feature-map value at position $(i,j)$ is:

$$Y[i,j] = \sum_{p=0}^{k-1}\sum_{q=0}^{k-1} X[i+p,\, j+q]\cdot K[p,q] \;(+\, b)$$

This is the **element-wise sum of products** between the image patch and the filter, plus scalar bias $b$ when used. The filter then moves by `stride` positions before the next value is calculated.

### A note on terminology: convolution vs. cross-correlation

The formula above describes **cross-correlation**, which is the operation implemented by `torch.nn.Conv2d`. The notebook confirms that the manual calculation without flipping the filter agrees with `nn.Conv2d` at `atol=1e-6`. Mathematical **convolution** also flips the filter and can be written as $Y[i,j] = \sum_{p,q} X[i-p,j-q]K[p,q]$. This difference has little practical effect during training because filter weights are learned from scratch. The term "convolutional layer" remains the established name.

### Output size: kernel size, stride, padding

For an input of size $H$ (analogously for $W$), filter size $K$, padding $P$ (number of zeros added on each side) and stride $S$ (the step by which the filter moves), standard convolution (no dilation) gives an output of size:

$$H' = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1$$

Tests in `tests/test_lecture_06_notebook.py` confirm the formula against `nn.Conv2d` for eight combinations of $H,K,S,P$. A larger $K$ reduces the output size. Padding $P>0$ limits this reduction, and $P=\lfloor K/2\rfloor$ with $S=1$ preserves the input size. A larger $S$ skips positions, which reduces both the output size and the amount of computation.

### Multiple input channels, multiple output filters

For $C_{in}>1$ input channels, a filter has shape $(C_{in}, k, k)$ and the sum also covers the channels.

$$Y[i,j] = \sum_{c=0}^{C_{in}-1}\sum_{p=0}^{k-1}\sum_{q=0}^{k-1} X[c, i+p, j+q]\cdot K[c,p,q] \;(+\, b)$$

A single filter produces one output map regardless of the number of input channels. Producing $C_{out}$ maps requires $C_{out}$ independent filters of shape $(C_{in},k,k)$, each with its own bias.

### Number of parameters

$$\#\text{parameters} = \underbrace{(C_{in}\cdot k\cdot k)}_{\text{weights of one filter}} \cdot\, C_{out} \;+\; \underbrace{C_{out}}_{\text{biases}} = (C_{in} k^2 + 1)\, C_{out}$$

Tests in `tests/test_lecture_06_notebook.py` confirm this formula for four configurations, including $k=5$. **The parameter count does not depend on $H$ or $W$** because the same filter moves across the whole image. A `Linear` layer that connects a flattened $C_{in}\times H\times W$ image to $C_{out}$ outputs has $C_{in}\cdot H\cdot W\cdot C_{out} + C_{out}$ parameters, so its size grows with image resolution. For a $28\times28\times1$ image and 32 filters sized $3\times3$, convolution has 320 parameters. A corresponding `Linear` layer has $25{,}120$, which is over $78\times$ more.

❌ **This does not guarantee better generalization.** The effect of a lower parameter count depends on whether locality and weight sharing match the data. These assumptions suit images well but are not justified for tabular features without a spatial structure.

### Intuition for gradient propagation through convolution

Lecture 3 derived the chain rule on a computational graph, so only the intuition is needed here. For a fixed filter, convolution is a linear transformation of the input. The loss gradient with respect to a filter weight sums contributions from every position where that weight was used. Weight sharing in the forward pass therefore means summing gradients in the backward pass. The input gradient uses an operation related to convolution with a flipped or transposed filter, which `autograd` calculates automatically.

## Worked example

A $5\times5$ image, a $3\times3$ filter (a vertical-edge detector), stride $1$, no padding:

$$X = \begin{pmatrix}1&2&3&0&1\\0&1&2&3&1\\1&0&1&2&0\\2&1&0&1&3\\0&2&1&0&1\end{pmatrix}, \qquad K = \begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}$$

The output size is $H'=\lfloor(5+0-3)/1\rfloor+1=3$, so $Y$ is a $3\times3$ matrix. For $Y[0,0]$, the filter covers the top-left patch $X[0{:}3,0{:}3]=\begin{pmatrix}1&2&3\\0&1&2\\1&0&1\end{pmatrix}$.

$$Y[0,0] = (1\cdot1 + 2\cdot0 + 3\cdot(-1)) + (0\cdot1+1\cdot0+2\cdot(-1)) + (1\cdot1+0\cdot0+1\cdot(-1))$$
$$= (1+0-3) + (0+0-2) + (1+0-1) = -2 -2 + 0 = -4$$

Analogously for the remaining eight positions (full computation and verification in the notebook):

$$Y = \begin{pmatrix}-4&-2&4\\0&-4&-1\\1&0&-2\end{pmatrix}$$

The result matches `torch.nn.Conv2d(1,1,kernel_size=3,bias=False)` at `atol=1e-6` after assigning the same filter weights.

## Experiment

Notebook [`examples.en.ipynb`](examples.en.ipynb) contains two experiments.

**A. Filters and feature maps.** A synthetic $12\times12$ image contains a bright square on a dark background. Manually defined vertical- and horizontal-edge filters produce strong responses on the square's borders and zero inside and outside. The experiment also covers the effect of stride and padding on output size, max pooling, and a comparison between a hand-designed filter and one learned during a short training run.

**B. CNN versus MLP on Fashion-MNIST.** A controlled comparison uses the same data split, random seeds, and a comparable training budget. Accuracy, parameter count, and measured training and inference time are reported for a small MLP and CNN. Selected first-layer filters and feature maps are also shown with the limits of their interpretation.

Both experiments run on CPU. `torchvision` downloads Fashion-MNIST on the first run, and the data is not stored in the repository. The download took about 5 seconds in this environment.

## Observations

- A vertical-edge filter gives zero response inside and outside the $6\times6$ square and values up to $\pm3$ on its left and right edges. A horizontal-edge filter behaves similarly on the top and bottom edges. The measured values follow directly from the formula.
- In a Fashion-MNIST comparison with 8000 training images, 10 epochs, and 3 random seeds, the CNN with about 20.5k parameters reaches $\approx0.85$ to $0.86$ test accuracy. The MLP has about 118k parameters, or **5.8× more**, and reaches $\approx0.84$ to $0.85$. The CNN performs slightly but consistently better with fewer parameters.
- The CNN trains on CPU about $15\times$ slower than the MLP, with roughly 12 s versus 0.8 s for 10 epochs. Inference on one test batch is about $15$ to $60\times$ slower. Each of the 320 parameters in the first convolutional layer is used at many filter positions, while each `Linear` parameter is used once per example.
- Some first-layer filters resemble familiar edge and contrast detectors, while others have less readable patterns. This is an observation from one run rather than a general guarantee.

## Common misconceptions

❌ A CNN "recognizes images" solely by applying hand-designed filters, such as the edge detectors from the example.

✅ Hand-designed filters only illustrate the mechanics of convolution. In a trained CNN, all filter weights are learned through gradient descent, just like MLP weights. They do not need to have a simple interpretation.

❌ Weight sharing means that all filters in a given layer are identical.

✅ Weight sharing applies to one filter used at every position. A convolutional layer usually contains many filters, and each has its own independently learned weights.

❌ A larger kernel always gives better results (because it "sees more at once").

✅ A larger kernel covers a wider area but requires more parameters and computation. It can also be less effective at detecting small local patterns. Several $3\times3$ layers are therefore often preferred over one layer with a large kernel.

❌ Pooling is a mandatory component of every CNN.

✅ Pooling is one way to reduce feature-map size and gain some local tolerance to shifts. Convolution with stride $>1$ provides a similar reduction, so some architectures omit separate pooling layers.

❌ A larger number of filters guarantees better quality.

✅ More filters increase model capacity. Depending on the data and architecture, this can help, cause overfitting, or make little difference. The effect must be checked experimentally.

❌ Feature maps clearly show named objects or concepts ("this filter detects cat ears").

✅ Edges and contrasts may be recognizable in the first layer. Deeper maps respond to distributed and increasingly abstract combinations of the signal, so assigning them simple labels requires additional analysis. A familiar-looking shape is not evidence of what the network encodes.

❌ Lower parameter cost automatically means lower runtime.

✅ In the experiment, a CNN with $5.8\times$ fewer parameters than a comparable MLP trained about $15\times$ slower on CPU. Parameter count and floating-point operations measure different properties. Weight sharing reduces the parameter count, but each weight is used many times.

## Engineering implications

- **Activation memory matters alongside parameters.** A CNN may have few parameters while producing large feature maps. With $C_{out}=32$ filters and a $28\times28$ image, each layer stores about $32\times28\times28\approx25{,}000$ numbers per example. Across larger images, deeper networks, and a full batch, activations often use more memory than the weights.
- **Computation grows with image resolution.** Convolution parameters do not depend on $H,W$, but the number of operations grows roughly with $H\cdot W$. Doubling both image dimensions requires about $4\times$ more computation per convolutional layer.
- **Latency and parameter count describe different properties.** The experiment produced a slower CNN despite its lower parameter count. This matters directly when selecting an architecture for CPUs or edge devices.
- **Quality, size, and cost are separate criteria.** A better model can mean higher accuracy, lower memory use, or shorter inference time. Improving all three at once is uncommon, so the deployment constraints determine the choice.
- **Limited hardware affects model design.** The number of filters, network depth, and input resolution require deliberate choices on devices without a GPU or with little memory.

## Receptive field

The **receptive field** of a neuron is the region of the input image that affects its value. A single $3\times3$ layer has a $3\times3$ receptive field. After two consecutive $3\times3$ layers with stride 1, it grows to $5\times5$, not $6\times6$. Each layer adds $k-1$ in both dimensions because neighboring neurons overlap.

The receptive field therefore grows with network depth without requiring larger individual filters. Later layers can combine local edges into patterns that cover a wider image context. This is one reason to use several small layers instead of a single large filter.

## Classic CNN architecture

```mermaid
flowchart LR
    IN["input: image (C,H,W)"] --> F1["convolutional block: Conv2d + activation + pooling"]
    F1 --> F2["convolutional block: Conv2d + activation + pooling"]
    F2 --> F3["... (feature extraction)"]
    F3 --> FLAT["flatten the feature map into a vector"]
    FLAT --> FC["fully connected layers (classification head)"]
    FC --> OUT["class logits"]
```

This structure appeared in the 1998 **LeNet-5** architecture (LeCun et al.), one of the first practical CNNs for handwritten digit recognition. Convolution, activation, and pooling blocks are repeated before flattening and several `Linear` layers. Later architectures, pretrained models, and transfer learning are covered in lecture 7.

## Key points

- Convolution uses locality and weight sharing. It provides translation equivariance, not full invariance.
- `nn.Conv2d` calculates cross-correlation rather than mathematical convolution with a flipped filter. This does not affect training, but the terminology is worth distinguishing.
- The output size is $\lfloor(H+2P-K)/S\rfloor+1$, and the parameter count is $(C_{in}k^2+1)C_{out}$. Unlike `Linear`, the parameter count is independent of $H,W$.
- Fewer parameters do not necessarily mean fewer operations. Shared convolution weights are reused across many image positions.
- The receptive field grows with network depth, allowing small filters to build a wider context.
- Feature maps require careful interpretation. A readable first-layer pattern does not imply an equally simple interpretation of deeper layers.

## Questions for students

1. What output shape and parameter count result from `Conv2d(3, 16, kernel_size=5, stride=2, padding=2)` for a $32\times32\times3$ image? The calculation should be completed without code and then verified.
2. What is the difference between translation equivariance and invariance? Consider the same object placed first in the top-left and then in the bottom-right corner, and identify what changes in the first convolutional layer's feature map.
3. Why did the CNN train more slowly on CPU despite having fewer parameters than the MLP?
4. Why is the receptive field after two $3\times3$ layers $5\times5$, not $6\times6$?

## References

- LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/, chapter 9 (Convolutional Networks).
- Xiao, H., Rasul, K., Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
- PyTorch Contributors. *torch.nn.Conv2d* documentation. https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

Full bibliographic entries are available in [`references/references.bib`](../../references/references.bib).
