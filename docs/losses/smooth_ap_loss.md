# SmoothAPLoss

[Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163)

```python {.signature}
losses.SmoothAPLoss(temperature=0.01, **options)
```

**Equation**:

$$
\mathcal{G}(x; \tau) = \frac{1}{1 + e^{-x / \tau}}
$$

$$
AP_q \approx \frac{1}{|\mathcal{S}_P|} \sum_{i \in \mathcal{S}_P} \frac{1 + \sum_{j \in \mathcal{S}_P, j \neq i} \mathcal{G}(D_{ij}; \tau)}{1 + \sum_{j \in \mathcal{S}_P, j \neq i} \mathcal{G}(D_{ij}; \tau) + \sum_{j \in \mathcal{S}_N} \mathcal{G}(D_{ij}; \tau)}
$$

$$
\mathcal{L}_{AP} = \frac{1}{m} \sum_{k=1}^{m} \left(1 - AP_k\right)
$$

Here $D_{ij} = s_j - s_i$, where $s_i$ is the cosine similarity between query $q$ and element $i$; $\mathcal{S}_P$ and $\mathcal{S}_N$ are the positive and negative sets of $q$; and $m$ is the batch size.

**Implementation notes**: Each embedding is a query against the whole batch.
The query is counted in its own $\mathcal{S}_P$, as in the authors'
implementation. Positive sets come from the labels, so classes may appear in
any order and with different counts. A query without another positive in the
batch has no loss term, as in [FastAPLoss](fast_ap_loss.md); its average
precision would be 1 by construction. With an `indices_tuple`, each query's loss
is multiplied by its [miner weight](../losses.md#miner-weights). Labels are
required, and `ref_emb` raises `ValueError`. Memory grows with the cube of the
batch size.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.SmoothAPLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.arange(8).repeat_interleave(4)
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **temperature**: The sigmoid temperature $\tau$. The paper uses 0.01.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One value $1 - AP_q$ per query that has another positive, indexed
  by query position in `embeddings`.
