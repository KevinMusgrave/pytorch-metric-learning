# DynamicArcFaceLoss

[Google Landmark Recognition 2020 Competition Third Place Solution](https://arxiv.org/abs/2010.05350)

[SubCenterArcFaceLoss](subcenter_arcface_loss.md) with a margin for each class
that grows as the class gets smaller, for long-tailed class distributions.

```python {.signature}
losses.DynamicArcFaceLoss(class_counts, embedding_size, margin_bounds=(0.05, 0.5), power=0.25, scale=64, sub_centers=3,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
m_{y_i} = a \cdot n_{y_i}^{-\lambda} + b
$$

Here $n_{y_i}$ is the size of the class of embedding $i$, and $m_{y_i}$ replaces
the constant margin in the [ArcFaceLoss](arcface_loss.md) equation. The
coefficients $a$ and $b$ follow from `margin_bounds`: the smallest class gets the
upper bound and the largest class gets the lower bound.

**Implementation notes**: `class_counts` is measured on the training set before
the loss is constructed. Its length sets the number of classes. The margins are
in radians and are stored in the `class_margins` buffer, which `.to()` moves
with the module and which is not saved in `state_dict`: reconstruct the loss
with the same arguments before loading a checkpoint. Forward passes do not change
the module. For a target angle above $\pi - m_{y_i}$, the
target logit is $s(\cos\theta_{y_i} - m_{y_i} \sin m_{y_i})$, as in
ArcFaceLoss. Everything else, including `get_logits`, `get_outliers`, and
[miner weights](../losses.md#miner-weights), comes from SubCenterArcFaceLoss.
With `sub_centers=1`, this is ArcFaceLoss with class-dependent margins.

Counts that are not positive integers raise `TypeError` for a non-integer dtype
and `ValueError` otherwise. Identical counts define no margin range and raise
`ValueError`; use ArcFaceLoss for a constant margin.

**Paper settings**: The defaults are the paper's best margin function,
$n^{-1/4}$ with bounds 0.05 and 0.5, with $K = 3$ sub-centers and the scale of
the ArcFace paper. The paper reports bounds rather than values of $a$ and $b$,
so the loss takes bounds; any $a > 0$ and $b$ correspond to one pair of bounds.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

class_counts = torch.tensor([3, 15, 200, 50, 1000])
loss_func = losses.DynamicArcFaceLoss(class_counts, embedding_size=128)
optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 5, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
optimizer.step()
```

**Parameters**:

- **class_counts**: A one-dimensional integer tensor or sequence with the
  number of training examples in each class. Labels must be in
  `0..len(class_counts) - 1`.
- **embedding_size**: The size of the embeddings passed to the loss.
- **margin_bounds**: The `(lower, upper)` margins in radians, with
  `0 <= lower < upper`. Another value raises `ValueError`.
- **power**: The exponent $\lambda$, which must be positive. The paper compares
  1, 1/4, and 1/8.
- **scale**: The scale $s$.
- **sub_centers**: $K$, the number of sub-centers per class.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  layout of `weight` is that of
  [SubCenterArcFaceLoss](subcenter_arcface_loss.md). Include
  `loss_func.parameters()` in an optimizer.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
