# SignalToNoiseRatioContrastiveLoss

[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}

```python
losses.SignalToNoiseRatioContrastiveLoss(pos_margin=0, neg_margin=1, **options)
```

**Equation**:

$$
L = \left[\operatorname{SNR}(\mathbf{x}_a, \mathbf{x}_p) - m_{pos}\right]_+ + \left[m_{neg} - \operatorname{SNR}(\mathbf{x}_a, \mathbf{x}_n)\right]_+,
\qquad
\operatorname{SNR}(\mathbf{x}_a, \mathbf{x}) = \frac{\operatorname{Var}(\mathbf{x}_a - \mathbf{x})}{\operatorname{Var}(\mathbf{x}_a)}
$$

This is [ContrastiveLoss](contrastive_loss.md#contrastiveloss) with the [SNRDistance](../distances.md#snrdistance): the noise-to-signal ratio replaces the Euclidean distance. The paper pairs it with [ZeroMeanRegularizer](../regularizers.md#zeromeanregularizer) as the `embedding_regularizer`.

**Parameters**:

- **pos_margin**: The noise-to-signal ratio over which positive pairs will contribute to the loss.
- **neg_margin**: The noise-to-signal ratio under which negative pairs will contribute to the loss.

**Default distance**:

- [`SNRDistance()`](../distances.md#snrdistance)
  - This is the only compatible distance.

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **pos_loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
- **neg_loss**: The loss per negative pair in the batch. Reduction type is `"neg_pair"`.
