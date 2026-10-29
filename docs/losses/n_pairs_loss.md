# NPairsLoss

[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)

```python
losses.NPairsLoss(**options)
```

**Equation**:

$$
L = \frac{1}{N} \sum_{i=1}^{N} \log\left(1 + \sum_{j \neq i} \exp\left(\mathbf{x}_i^\top \mathbf{x}_j^+ - \mathbf{x}_i^\top \mathbf{x}_i^+\right)\right)
$$

Here the batch provides $N$ pairs $(\mathbf{x}_i, \mathbf{x}_i^+)$ with distinct labels, and the loss is the cross entropy of classifying anchor $i$'s positive among all $N$ positives.

**Implementation notes**: For each label, the first positive pair in index
order, from labels or from `indices_tuple`, forms the
$(\mathbf{x}_i, \mathbf{x}_i^+)$ pair; other samples of that label are ignored.
For batches with more than two samples per label, use
[NTXentLoss](ntxent_loss.md), which uses every positive pair. With a distance
where smaller means closer, the logits are negated distances. Labels are
required, and `ref_emb` raises `ValueError`.

**Paper settings**: The default distance L2-normalizes the embeddings, so
$\mathbf{x}_i^\top \mathbf{x}_j^+$ is a cosine similarity. The paper uses
unnormalized inner products and penalizes the squared embedding norms. Pass
`distance=distances.DotProductSimilarity(normalize_embeddings=False)` and an
`embedding_regularizer` such as `regularizers.LpRegularizer(power=2)` for the
paper's formulation.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.NPairsLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.arange(16).repeat_interleave(2)
loss = loss_func(embeddings, labels)
loss.backward()
```

**Default distance**:

- [`DotProductSimilarity()`](../distances.md#dotproductsimilarity), which
  L2-normalizes the embeddings.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per selected pair, indexed by its anchor's position in
  `embeddings`.
