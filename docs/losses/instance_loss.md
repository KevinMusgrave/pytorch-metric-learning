# InstanceLoss

[Dual-Path Convolutional Image-Text Embeddings with Instance Loss](https://arxiv.org/abs/1711.05535)

```python {.signature}
losses.InstanceLoss(gamma=64, **options)
```

**Equation**:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left(\gamma \cos(\mathbf{x}_i, \mathbf{x}_{t_i})\right)}{\sum_{j=1}^{N} \exp\left(\gamma \cos(\mathbf{x}_i, \mathbf{x}_j)\right)}
$$

Each embedding is classified against every embedding in the batch, itself included, with target column $t_i$.

**Implementation notes**: $t_i$ is the rank of `labels[i]` among the sorted
distinct labels of the batch, so labels are remapped to `0..K-1` and that rank
is used as a column of the `(N, N)` similarity matrix. With one instance per
label and labels that increase with batch position, such as
`labels = torch.arange(N)`, $t_i = i$ and each embedding is its own class, as in
the paper. With other labels, the target column is not necessarily an
embedding with the same label. `indices_tuple` and `ref_emb` raise
`ValueError`.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.InstanceLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.arange(32)
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **gamma**: $\gamma$, the scale applied to the cosine similarity matrix. The
  paper uses 64 for its two-stream setting.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
