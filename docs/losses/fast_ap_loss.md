# FastAPLoss

[Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf)

```python
losses.FastAPLoss(num_bins=10, **options)
```

**Equation**:

$$
\operatorname{FastAP}_q = \sum_{z=1}^{L} \frac{h_q^+(z)\, H_q^+(z)}{H_q(z)},
\qquad
h_q^+(z) = \sum_{j \in \mathcal{S}_q^+} \delta\left(d_{qj}, z\right),
\qquad
H_q^+(z) = \sum_{l \le z} h_q^+(l)
$$

$$
L = \frac{1}{|Q|} \sum_{q \in Q} \left(1 - \frac{\operatorname{FastAP}_q}{|\mathcal{S}_q^+|}\right)
$$

Here $d_{qj}$ is the squared Euclidean distance between normalized embeddings, binned into $L$ bins over $[0, 2^{\text{power}}]$ with the triangular kernel $\delta(d, z) = \left[1 - |d - z| / \Delta\right]_+$; $h_q$ and $H_q$ are the soft histogram and cumulative histogram over all retrieved items, and $h_q^+$, $H_q^+$ over the positives of query $q$. $Q$ is the set of batch elements with at least one positive.

**Implementation notes**: Each embedding is a query against the rest of the
batch; the query itself is excluded from its positives and negatives. Bins
where the product $h_q^+ H_q^+$ or $H_q$ is zero contribute 0. With an
`indices_tuple`, each query's loss is multiplied by its
[miner weight](../losses.md#miner-weights).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.FastAPLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **num_bins**: $L$, the number of soft histogram bins. The paper suggests 10.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](../distances.md#lpdistance).
  The distance must be `LpDistance(normalize_embeddings=True, p=2)` with any
  `power`; another distance class raises `TypeError` and other settings raise
  `ValueError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per query with at least one positive, indexed by those
  queries' positions in `embeddings`.
