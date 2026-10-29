# PNPLoss

[Rethinking the Optimization of Average Precision: Only Penalizing Negative Instances before Positive Ones is Enough](https://arxiv.org/abs/2102.04640)

```python
losses.PNPLoss(
    b=2,
    alpha=1,
    anneal=0.01,
    variant="O",
    *,
    distance=None,
    embedding_regularizer=None,
    embedding_reg_weight=1.0,
)
```

**Equation**:

$$
\begin{aligned}
L_O &= \frac{1}{|S_P|} \sum_{i \in S_P} R(i, S_N) && (5) \\
L_{I_u} &= \frac{1}{|S_P|} \sum_{i \in S_P} \left(1 + R(i, S_N)\right) \log\left(1 + R(i, S_N)\right) && (7) \\
L_{I_b} &= \frac{1}{|S_P|} \sum_{i \in S_P} \frac{1}{b^2} \left(b R(i, S_N) - \log\left(1 + b R(i, S_N)\right)\right) && (9) \\
L_{D_s} &= \frac{1}{|S_P|} \sum_{i \in S_P} \log\left(1 + R(i, S_N)\right) && (12) \\
L_{D_q} &= 1 - \frac{1}{|S_P|} \sum_{i \in S_P} \frac{1}{\left(1 + R(i, S_N)\right)^{\alpha}} \quad (\alpha \geq 1) && (13)
\end{aligned}
$$

Here $R(i, S_N) = \sum_{j \in S_N} \sigma\left((s_j - s_i)/\tau\right)$, where $s_i$ is the cosine similarity between the query and element $i$, $\sigma$ is the sigmoid, and $\tau$ is `anneal`.

**Implementation notes**: Each embedding is a query against the whole batch.
The query is counted in its own $S_P$, as in the authors' implementation. The
per-query values are averaged over the queries that have another positive in
the batch; a query whose only positive is itself is excluded, and a batch
without positive pairs, including an empty batch, gives a differentiable zero.
The authors' batches give every query another positive, so this matches their
average. Labels are required; `indices_tuple` and `ref_emb` raise
`ValueError`. Memory grows with the cube of the batch size.

**Paper settings**: The paper recommends PNP-Dq with $\alpha = 8$:
`PNPLoss(variant="Dq", alpha=8)`. The default `variant="O"` is the plain
PNP loss of equation 5.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.PNPLoss(variant="Dq", alpha=8)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **b**: The boundary $b$ of PNP-Ib (equation 9). The paper uses 2.
- **alpha**: The power $\alpha$ of PNP-Dq (equation 13). The paper uses 8.
- **anneal**: The sigmoid temperature $\tau$ used for $R$. The paper uses 0.01.
- **variant**: One of `"Ds"`, `"Dq"`, `"Iu"`, `"Ib"`, or `"O"`. Any other
  value raises `ValueError`.
- **distance**, **embedding_regularizer**, **embedding_reg_weight**: See
  [loss options](../losses.md#loss-options).

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Reducer**:

- None. The loss reduces its own value, so it has no `reducer` option; passing
  `reducer` raises `TypeError`.

**Loss terms**:

- **loss**: The average of the selected variant over anchors that have another
  positive, with `indices=None`.
