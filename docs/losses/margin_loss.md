# MarginLoss

[Sampling Matters in Deep Embedding Learning](https://arxiv.org/abs/1706.07567)

```python
losses.MarginLoss(
    margin=0.2,
    nu=0,
    beta=1.2,
    triplets_per_anchor="all",
    learn_beta=False,
    num_classes=None,
    **options,
)
```

**Equation**:

$$
\operatorname{minimize} \sum_{(i,j)} \ell^{\mathrm{margin}}(i,j) + \nu\left(\beta^{(0)} + \beta^{(\mathrm{class})}_{c(i)} + \beta^{(\mathrm{img})}_i\right)
$$

where

$$
\ell^{\mathrm{margin}}(i,j) = \left(\alpha + y_{ij}(D_{ij} - \beta)\right)_+
$$

Here $y_{ij}$ is $1$ for positive pairs and $-1$ for negative pairs, $D_{ij}$ is the distance between embeddings $i$ and $j$, and $c(i)$ is the class of sample $i$.

**Implementation notes**: The loss is computed over triplets, all of them by
default, so each pair term repeats once per triplet that contains it. Each
triplet's value is its positive-pair term plus its negative-pair term, and the
sum over triplets is divided by the number of nonzero pair terms. $\beta$ is
either a single value or one value per class indexed by the anchor's class; the
per-image $\beta^{(\mathrm{img})}$ is not supported. The $\nu$ penalty is added
only when `learn_beta=True`. `beta` is a parameter when `learn_beta=True` and a
buffer otherwise; either way it follows `loss_func.to(device, dtype)`, is saved
in `state_dict`, and must match the embeddings' placement outside autocast.

**Paper settings**: The paper learns $\beta$, while the default keeps it fixed
at its initial value. Pass `learn_beta=True`, with `num_classes` for the
class-specific $\beta^{(\mathrm{class})}$, and include `loss_func.parameters()`
in the optimizer. The defaults `margin=0.2` and `beta=1.2` are the paper's.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.MarginLoss(nu=0.1, learn_beta=True, num_classes=10)
optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 10, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
optimizer.step()
```

**Parameters**:

- **margin**: The margin $\alpha$. The paper uses 0.2.
- **nu**: The weight $\nu$ of the $\beta$ penalty.
- **beta**: The initial $\beta$. The paper uses 1.2.
- **triplets_per_anchor**: The number of triplets per anchor sampled within a
  batch, an integer or `"all"`. For example, a batch of 128 with
  `triplets_per_anchor=100` samples 12800 triplets. With `"all"`, every triplet
  in the batch is used.
- **learn_beta**: If `True`, `beta` is a `torch.nn.Parameter`; include
  `loss_func.parameters()` in an optimizer.
- **num_classes**: If not `None`, `beta` has one value per class, and labels
  are required.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [DivisorReducer](../reducers.md#divisorreducer). Terms with a divisor must
  reach DivisorReducer, directly or through MultipleReducers; any other reducer
  raises `ValueError`.

**Reducer input**:

- **loss**: One loss per triplet, indexed by
  `TripletIndices(anchors, positives, negatives)`, with `divisor` equal to the
  number of nonzero pair terms.
- **beta_reg_loss**: Present only when `learn_beta=True`. With `num_classes=None`,
  already reduced: $\nu\beta$. Otherwise one value $\nu\beta_{c(a)}$ per triplet,
  indexed by the triplet anchors, with the same `divisor` as `loss`.
