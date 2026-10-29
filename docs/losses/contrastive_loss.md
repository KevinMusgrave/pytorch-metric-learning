# ContrastiveLoss

[Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
(Hadsell, Chopra, and LeCun, 2006)

```python
losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1.0, **options)
```

**Equation**:

With a distance such as [LpDistance](../distances.md#lpdistance), the loss is:

$$
L_{contrastive} = \left[d_p - m_{pos}\right]_+ + \left[m_{neg} - d_n\right]_+
$$

Here $d_p$ and $d_n$ are positive-pair and negative-pair distances, $m_{pos}$ and $m_{neg}$ are `pos_margin` and `neg_margin`, and $[x]_+ = \max(0, x)$.

With a similarity such as [CosineSimilarity](../distances.md#cosinesimilarity), the loss is:

$$
L_{contrastive} = \left[m_{pos} - s_p\right]_+ + \left[s_n - m_{neg}\right]_+
$$

Here $s_p$ and $s_n$ are positive-pair and negative-pair similarities.

**Implementation notes**: The hinges are not squared, unlike the paper's
$\frac{1}{2} d_p^2$ and $\frac{1}{2}[m - d_n]_+^2$. The default reducer averages
the positive-pair losses greater than zero and, separately, the negative-pair
losses greater than zero, then adds the two averages.

**Example**:

```python
import torch
from pytorch_metric_learning import distances, losses

loss_func = losses.ContrastiveLoss(
    pos_margin=1, neg_margin=0, distance=distances.CosineSimilarity()
)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **pos_margin**: The distance (or similarity) over (under) which positive
  pairs contribute to the loss.
- **neg_margin**: The distance (or similarity) under (over) which negative
  pairs contribute to the loss.

The defaults suit a distance such as LpDistance. With a similarity such as
CosineSimilarity, use values such as `pos_margin=1` and `neg_margin=0`.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **pos_loss**: One loss per (anchor, positive) pair, indexed by `(anchors, positives)`.
- **neg_loss**: One loss per (anchor, negative) pair, indexed by `(anchors, negatives)`.
