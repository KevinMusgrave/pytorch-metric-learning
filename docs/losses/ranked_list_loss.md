# RankedListLoss

[Ranked List Loss for Deep Metric Learning](https://arxiv.org/abs/1903.03238)

```python
losses.RankedListLoss(
    margin, neg_temperature, imbalance=0.5, alpha=None, pos_temperature=0, **options
)
```

**Equation**:

$$
L = \frac{1}{N} \sum_{i=1}^{N} \left[(1 - \lambda)\, L_P(\mathbf{x}_i) + \lambda\, L_N(\mathbf{x}_i)\right]
$$

$$
L_P(\mathbf{x}_i) = \frac{\sum_{j \in P^*_i} w_{ij} \left(d_{ij} - (\alpha - m)\right)}{\sum_{j \in P^*_i} w_{ij}},
\qquad
L_N(\mathbf{x}_i) = \frac{\sum_{k \in N^*_i} w_{ik} \left(\alpha - d_{ik}\right)}{\sum_{k \in N^*_i} w_{ik}}
$$

$$
w_{ij} = e^{T_p \left(d_{ij} - (\alpha - m)\right)}, \qquad w_{ik} = e^{T_n \left(\alpha - d_{ik}\right)}
$$

Here $d$ is the Euclidean distance between normalized embeddings, $P^*_i$ holds the positives of $\mathbf{x}_i$ with $d_{ij} > \alpha - m$, $N^*_i$ holds the negatives with $d_{ik} < \alpha$, $\lambda$ is `imbalance`, $m$ is `margin`, and $T_p$ and $T_n$ are `pos_temperature` and `neg_temperature`.

**Implementation notes**: $L_P$ and $L_N$ are exact weighted averages,
computed as softmax-weighted sums over the violating pairs, so large
temperatures do not overflow. An anchor without violating positives or without
violating negatives contributes 0 for that part. Each embedding is an anchor
against the rest of the batch. Labels are required; `indices_tuple` and
`ref_emb` raise `ValueError`.

**Paper settings**: The defaults `pos_temperature=0` and `alpha=None`
correspond to the paper's RLL-Simpler variant, which it recommends for
exploratory analysis: positives are weighted uniformly and
$\alpha = 1 + m/2$.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.RankedListLoss(margin=0.4, neg_temperature=10)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **margin**: The margin $m$ between the positive and negative boundaries.
- **neg_temperature**: The temperature $T_n$ that weights negative pairs.
- **imbalance**: The tradeoff $\lambda$ between the positive and negative
  parts, between 0 and 1; other values raise `ValueError`.
- **alpha**: The negative boundary $\alpha$. `None` sets $\alpha = 1 + m/2$.
- **pos_temperature**: The temperature $T_p$ that weights positive pairs.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per anchor, indexed by anchor position in `embeddings`.
