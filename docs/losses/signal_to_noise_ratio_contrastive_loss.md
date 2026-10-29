# SignalToNoiseRatioContrastiveLoss

[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python {.signature}
losses.SignalToNoiseRatioContrastiveLoss(pos_margin=0.0, neg_margin=1.0, **options)
```

**Equation**:

$$
L = \left[\operatorname{SNR}(\mathbf{x}_a, \mathbf{x}_p) - m_{pos}\right]_+ + \left[m_{neg} - \operatorname{SNR}(\mathbf{x}_a, \mathbf{x}_n)\right]_+,
\qquad
\operatorname{SNR}(\mathbf{x}_a, \mathbf{x}) = \frac{\operatorname{Var}(\mathbf{x}_a - \mathbf{x})}{\operatorname{Var}(\mathbf{x}_a)}
$$

**Implementation notes**: This is [ContrastiveLoss](contrastive_loss.md) with
the [SNRDistance](../distances.md#snrdistance): the noise-to-signal ratio
replaces the Euclidean distance. The paper pairs it with
[ZeroMeanRegularizer](../regularizers.md#zeromeanregularizer) as the
`embedding_regularizer`.

**Paper settings**: The paper adds the zero-mean regularizer, which is not
enabled by default. Pass
`embedding_regularizer=regularizers.ZeroMeanRegularizer()` for the paper's
formulation.

**Example**:

```python
import torch
from pytorch_metric_learning import losses, regularizers

loss_func = losses.SignalToNoiseRatioContrastiveLoss(
    embedding_regularizer=regularizers.ZeroMeanRegularizer()
)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **pos_margin**: The noise-to-signal ratio over which positive pairs
  contribute to the loss.
- **neg_margin**: The noise-to-signal ratio under which negative pairs
  contribute to the loss.

**Default distance**:

- [`SNRDistance()`](../distances.md#snrdistance). This is the only compatible
  distance; another distance class raises `TypeError`.

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **pos_loss**: One loss per (anchor, positive) pair, indexed by `(anchors, positives)`.
- **neg_loss**: One loss per (anchor, negative) pair, indexed by `(anchors, negatives)`.
