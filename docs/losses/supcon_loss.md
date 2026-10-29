# SupConLoss

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)

```python
losses.SupConLoss(temperature=0.1, **options)
```

**Equation**:

$$
\mathcal{L}^{sup}_{out} = \sum_{i \in I} \frac{-1}{\left|P(i)\right|} \sum_{p \in P(i)} \log \frac{\exp\left(\mathbf{z}_i \cdot \mathbf{z}_p / \tau\right)}{\sum_{a \in A(i)} \exp\left(\mathbf{z}_i \cdot \mathbf{z}_a / \tau\right)}
$$

Here $I$ indexes the batch, $A(i)$ is every index except $i$, and $P(i) \subseteq A(i)$ holds the indices that share $i$'s label.

**Implementation notes**: $A(i)$ is the set of positives and negatives of
anchor $i$: every other embedding, every reference row with `ref_emb`, or the
pairs of anchor $i$ in a supplied `indices_tuple`. The denominator is a masked
log-sum-exp. An anchor with positives but no negatives has a nonzero loss, as
in the paper, because its denominator still contains its positives. An anchor
without positives has no loss term, so any reducer averages only anchors with
positives. The default reducer averages over the anchors with a positive loss
instead of summing as the paper does. With a distance where smaller means
closer, the logits are negated distances.

**Paper settings**: The paper sums the per-anchor losses; the default reducer
averages the positive ones. Pass `reducer=reducers.SumReducer()` for the
paper's sum. The default `temperature=0.1` is the paper's.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.SupConLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **temperature**: The temperature $\tau$. The paper uses 0.1.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: One loss per anchor that has a positive, indexed by anchor position
  in `embeddings`.
