# CentroidTripletLoss

[On the Unreasonable Effectiveness of Centroids in Image Retrieval](https://arxiv.org/abs/2104.13643)
(Wieczorek et al.) compares each anchor with class centroids instead of with individual
samples, which makes the objective less sensitive to a single unrepresentative image.

```python {.signature}
losses.CentroidTripletLoss(margin=0.05, smooth_loss=False, **options)
```

**Equation**:

$$
\mathcal{L} = \left[ d(f(A), c_P) - d(f(A), c_N) + \alpha_c \right]_+
$$

$A$ is the anchor and $d$ the distance, squared Euclidean by default. $c_P$ is the mean of
the other embeddings in the anchor's class, so the anchor is left out of its own
centroid. $c_N$ is the mean of every embedding in another class.

**Implementation notes**: One loss is computed for each anchor and each other class, so a
batch of $P$ classes with $M$ samples each has $PM(P - 1)$ losses. Classes may have
different sizes.

A class with one embedding in the batch has no positive centroid, so that embedding is not
an anchor. Its centroid is still a negative for the other classes. If no anchor remains,
or the batch has one class, the result is a differentiable zero. Centroids are computed
from the embeddings as passed, before the distance's normalization. float16 and bfloat16
inputs are computed in float32, and the loss is returned in float32, because a squared
distance can exceed the float16 range even when the final loss would fit.

The loss takes `labels` only. `indices_tuple` and `ref_emb` raise `ValueError`.

**Paper settings**: This is equation 2 of the paper, with the centroid of equation 3.
The default distance is the paper's squared Euclidean distance on unnormalized embeddings.
The paper gives no value for $\alpha_c$, so the default margin is unchanged from the
original implementation in this library. The paper uses batches of $P$ classes with $M$
samples each, which [MPerClassSampler](../samplers.md#mperclasssampler) provides, and adds
this loss to a triplet loss, a center loss, and a classification loss.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.CentroidTripletLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.arange(32) // 4
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **margin**: The margin $\alpha_c$.
- **smooth_loss**: Use `softplus` instead of the hinge $[\cdot]_+$.

**Default distance**:

- [`LpDistance(normalize_embeddings=False, p=2, power=2)`](../distances.md#lpdistance)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Credit**: Chan Woo Kim ([cwkeam](https://github.com/cwkeam)) contributed the original
implementation to this library. [pgrosjean](https://github.com/pgrosjean) found why
classes of different sizes failed in
[#451](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/451).
