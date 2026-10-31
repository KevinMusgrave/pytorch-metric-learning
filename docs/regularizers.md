# Regularizers

Regularizers penalize a two-dimensional tensor, such as a batch of embeddings
or a loss's learnable class weights, without labels or tuples. Each row of the
tensor is one embedding or one class weight vector.

Here is a weight regularizer passed to a loss function:

```python
from pytorch_metric_learning import losses, regularizers

regularizer = regularizers.RegularFaceRegularizer()
loss_func = losses.ArcFaceLoss(
    num_classes=100, embedding_size=128, weight_regularizer=regularizer
)
```

## BaseRegularizer

```python {.signature}
regularizers.BaseRegularizer(*, distance=None, reducer=None)
```

An object that extends this class can be passed as the `embedding_regularizer` into any loss function. It can also be passed as the `weight_regularizer` into any loss with [learnable weight arguments](losses.md#learnable-weight-arguments). The loss multiplies the regularizer's reduced value by its `*_reg_weight` and adds it after reducing its own objective terms.

**Parameters**

- **distance**: A [distance](distances.md) object. If `None`, the
  regularizer's default distance is used.
- **reducer**: A [reducer](reducers.md) object. If `None`, the regularizer's
  default reducer is used.

Regularizer signatures end with `**options`, which accepts these two
keyword-only arguments; see `pytorch_metric_learning.types.RegularizerOptions`.
LpRegularizer, SphericalEmbeddingRegularizer, and ZeroMeanRegularizer use no distance and accept only
`reducer`.

Calling `regularizer(x)` computes one loss term with one value per row of `x`,
or per selected pair of rows, and reduces it with the regularizer's reducer. The
reducer receives `labels=None`, so reducers that need labels, such as
[ClassWeightedReducer](reducers.md#classweightedreducer), raise `ValueError`.
Subclasses implement `compute_loss(x)` and return a
[loss-term dictionary](reducers.md#loss-terms).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

## CenterInvariantRegularizer

[Deep Face Recognition with Center Invariant Loss](http://www1.ece.neu.edu/~yuewu/files/2017/twu024.pdf)

This encourages unnormalized embeddings or weights to all have the same Lp norm.

```python {.signature}
regularizers.CenterInvariantRegularizer(**options)
```

**Default distance**:

- [`LpDistance(normalize_embeddings=False, p=2, power=1)`](distances.md#lpdistance)
  - The distance must be `LpDistance(normalize_embeddings=False, power=1)`; `p`
    can be changed. Another distance class raises `TypeError`, and other
    `normalize_embeddings` or `power` settings raise `ValueError`.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

## LpRegularizer

This encourages embeddings or weights to have a small Lp norm. The loss for
row `i` is `torch.norm(x[i], p=p) ** power`.

```python {.signature}
regularizers.LpRegularizer(p=2, power=1, *, reducer=None)
```

**Parameters**

- **p**: The order of the norm. For example, `p=1` is the L1 norm and `p=2` is
  the Euclidean norm.
- **power**: The exponent applied to each row's norm. `power=2` penalizes the
  squared norm.

**Distance**: None. The regularizer takes no `distance` argument.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

## RegularFaceRegularizer

[RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf)

This should be applied as a weight regularizer. It penalizes class vectors that
are very close together: the loss for class `i` is its largest similarity to
another class vector.

```python {.signature}
regularizers.RegularFaceRegularizer(**options)
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)
  - Only similarities (`is_inverted=True`) are compatible. For example,
    [`DotProductSimilarity(normalize_embeddings=True)`](distances.md#dotproductsimilarity)
    also works.
    A distance raises `TypeError`.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

## SparseCentersRegularizer

[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)

This should be applied as a weight regularizer. It encourages multiple class centers to "merge", i.e. group together.

```python {.signature}
regularizers.SparseCentersRegularizer(num_classes, centers_per_class, **options)
```

**Parameters**

- **num_classes**: The number of classes in your training dataset.
- **centers_per_class**: The number of consecutive rows in the weight matrix
  that belong to one class. It must be greater than 1; otherwise construction
  raises `ValueError`.

The regularized tensor must have `num_classes * centers_per_class` rows. The
term carries a divisor equal to twice the number of same-class center pairs, so
it requires [DivisorReducer](reducers.md#divisorreducer), the default.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)
  - This is the only compatible distance. Another distance raises `TypeError`.

**Default reducer**:

- [DivisorReducer](reducers.md#divisorreducer)

## SphericalEmbeddingRegularizer

[Deep Metric Learning with Spherical Embedding](https://arxiv.org/abs/2011.02785) (Zhang
et al.)

This is the spherical embedding constraint, SEC. Losses that normalize embeddings still
update them through their unnormalized form, and an embedding with a large norm then
changes direction less than one with a small norm. SEC pulls every norm toward the
batch's mean norm, without fixing a radius. The loss for row `i` is:

$$
\mathcal{L}_{\mathrm{sec},i} = \left(\lVert f_i \rVert_2 - \mu\right)^2,
\qquad \mu = \frac{1}{N} \sum_{j=1}^{N} \lVert f_j \rVert_2
$$

```python {.signature}
regularizers.SphericalEmbeddingRegularizer(*, reducer=None)
```

Pass it as the `embedding_regularizer`, and set the paper's $\eta$ with
`embedding_reg_weight`. The objective is then $\mathcal{L}_{\mathrm{metric}} + \eta
\mathcal{L}_{\mathrm{sec}}$:

```python
import torch
from pytorch_metric_learning import losses, regularizers

loss_func = losses.TripletMarginLoss(
    embedding_regularizer=regularizers.SphericalEmbeddingRegularizer(),
    embedding_reg_weight=0.5,
)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.arange(32) // 4
loss = loss_func(embeddings, labels)
loss.backward()
```

Do not L2-normalize embeddings before the loss. Every norm would equal 1, so the
penalty and its gradient would be zero. Normalization belongs in the loss's distance,
which is how the default distances work. The paper's $\eta$ study on Cars196 with a
triplet loss reports its best recall at 0.5, among 0, 0.1, 0.5, 1.0, and 1.5.

$\mu$ stays in the graph, which does not change the gradient, because the deviations
from the mean sum to zero. The gradient for row `i` under the default reducer is
equation 18 of the paper, $\frac{2}{N}(\lVert f_i \rVert_2 - \mu) \hat{f}_i$. A zero row
receives a zero gradient. float16 and bfloat16 inputs stay in float32 until the
reducer has finished, so one large deviation cannot overflow the mean.

SEC differs from [CenterInvariantRegularizer](#centerinvariantregularizer), which
penalizes $(\lVert x_i \rVert^2 - \overline{\lVert x \rVert^2})^2 / 4$, the deviation of
squared norms. That places much more weight on large norms, so the two disagree in value
and gradient, and neither is a rescaling of the other. SEC with $\mu = 0$ is the squared
[LpRegularizer](#lpregularizer), `LpRegularizer(power=2)`, which the paper calls L2-reg.

**Distance**: None. The regularizer takes no `distance` argument.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

## ZeroMeanRegularizer

[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python {.signature}
regularizers.ZeroMeanRegularizer(*, reducer=None)
```

**Equation**

In this equation, `N` is the batch size, `M` is the size of each embedding.

$$
L_r = \frac{1}{N} \sum_{i=1}^{N} \left| \sum_{m=1}^{M} h_{im} \right|
$$

Here $h_{im}$ is the $m$-th component of the $i$-th embedding. The loss that applies the regularizer multiplies $L_r$ by its `embedding_reg_weight` or `weight_reg_weight`, which is $\lambda$ in the paper.

**Distance**: None. The regularizer takes no `distance` argument.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)
