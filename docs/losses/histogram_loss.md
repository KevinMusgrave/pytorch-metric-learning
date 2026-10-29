# HistogramLoss

[Learning Deep Embeddings with Histogram Loss](https://arxiv.org/abs/1611.00822)

```python {.signature}
losses.HistogramLoss(
    num_bins=100,
    *,
    distance=None,
    embedding_regularizer=None,
    embedding_reg_weight=1.0,
)
```

**Equation**:

$$
L = \sum_{r=0}^{B} p^-_r \, \Phi^+_r, \qquad \Phi^+_r = \sum_{q=0}^{r} p^+_q
$$

$$
p^\pm_r = \frac{1}{|\mathcal{S}^\pm|} \sum_{(i,j) \in \mathcal{S}^\pm} \delta_{ij,r},
\qquad
\delta_{ij,r} = \begin{cases} (s_{ij} - t_{r-1}) / \Delta & s_{ij} \in [t_{r-1}, t_r] \\ (t_{r+1} - s_{ij}) / \Delta & s_{ij} \in [t_r, t_{r+1}] \\ 0 & \text{otherwise} \end{cases}
$$

Here $s_{ij}$ is the cosine similarity of pair $(i, j)$, $\mathcal{S}^+$ and $\mathcal{S}^-$ are the sets of positive and negative pairs, $B$ is `num_bins`, $t_r = -1 + r\Delta$ for $r = 0, \ldots, B$ are the nodes of $[-1, 1]$ with spacing $\Delta = 2 / B$, and $p^\pm$ are the linearly interpolated histograms of positive and negative similarities. The loss estimates the probability that a random negative pair is more similar than a random positive pair.

**Implementation notes**: Pairs come from all triplets of the batch, or from
the supplied `indices_tuple`. Each unordered pair is counted once, so duplicate
pairs from triplet enumeration do not change the histograms. Similarities must
lie in $[-1, 1]$. Similarities are clamped to $[-1, 1]$, so identical or opposite embeddings fall in the end bins.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.HistogramLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **num_bins**: $B$, the number of histogram intervals on $[-1, 1]$. A value
  that is not an `int` raises `TypeError`; a value below 1 raises `ValueError`.
- **distance**, **embedding_regularizer**, **embedding_reg_weight**: See
  [loss options](../losses.md#loss-options).

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Reducer**:

- None. The loss reduces its own value, so it has no `reducer` option; passing
  `reducer` raises `TypeError`.

**Loss terms**:

- **loss**: The value of $L$, with `indices=None`.
