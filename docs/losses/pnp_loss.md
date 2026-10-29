# PNPLoss

[Rethinking the Optimization of Average Precision: Only Penalizing Negative Instances before Positive Ones is Enough](https://arxiv.org/abs/2102.04640)

```python
losses.PNPLoss(b=2, alpha=1, anneal=0.01, variant="O", **options)
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
The query is counted in its own $S_P$, as in the authors' implementation, so
every query has at least one positive. The per-query values are averaged over
the batch. Labels are required; `indices_tuple` and `ref_emb` raise
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

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: Already reduced: the batch average of the selected variant. Every
  reducer returns it unchanged.
