# LiftedStructureLoss

[Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/abs/1511.06452)

```python
losses.LiftedStructureLoss(neg_margin=1.0, pos_margin=0.0, **options)
```

**Equation**:

$$
\begin{aligned}
\tilde{J}_{i,j} &= \log\left(\sum_{(i,k) \in \mathcal{N}} \exp\lbrace \alpha - D_{i,k} \rbrace + \sum_{(j,l) \in \mathcal{N}} \exp\lbrace \alpha - D_{j,l} \rbrace\right) + D_{i,j} \\
\tilde{J} &= \frac{1}{2\lvert\mathcal{P}\rvert} \sum_{(i,j) \in \mathcal{P}} \max\left(0, \tilde{J}_{i,j}\right)^2
\end{aligned}
$$

Here $\mathcal{P}$ and $\mathcal{N}$ are the positive and negative pairs in the batch, and $D_{i,j}$ is the distance between embeddings $i$ and $j$.

**Implementation notes**: The negatives of each element are combined with a
per-anchor log-sum-exp; float16 and bfloat16 inputs accumulate in float32. A
positive pair whose two elements have no negatives contributes 0: the sum is
empty, its logarithm is $-\infty$, and the hinge is 0, as in the paper.
$\tilde{J}_{i,j}$ uses the negatives of both $i$ and $j$, so anchors and
positives must come from the same set. Passing `ref_emb` raises `ValueError`;
use [GeneralizedLiftedStructureLoss](generalized_lifted_structure_loss.md) for
separate query and reference sets.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.LiftedStructureLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **neg_margin**: The margin $\alpha$. The paper uses 1.
- **pos_margin**: A margin subtracted from $D_{i,j}$ in $\tilde{J}_{i,j}$. The
  paper uses 0, so it does not appear in the equation.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One value $\frac{1}{2}\max(0, \tilde{J}_{i,j})^2$ per (anchor,
  positive) pair, indexed by `(anchors, positives)`.
