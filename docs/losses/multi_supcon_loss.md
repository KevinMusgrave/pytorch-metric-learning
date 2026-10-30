# MultiSupConLoss

[Contrastive Learning for Multi-Label Classification](https://sslneurips22.github.io/paper_pdfs/paper_8.pdf)

SupCon for samples that carry a set of labels. Two samples are positives when
their label sets overlap enough, and each positive is weighted by that overlap.

```python {.signature}
losses.MultiSupConLoss(temperature=0.1, threshold=0.5, **options)
```

**Equation**:

$$
\mathcal{L}^{sup}_{ml} = \sum_{i \in I} \frac{-1}{\left|N(i)\right|} \sum_{p \in N(i)} s_{i,p} \log \frac{\exp\left(\mathbf{z}_i \cdot \mathbf{z}_p / \tau\right)}{\sum_{a \in A(i)} \exp\left(\mathbf{z}_i \cdot \mathbf{z}_a / \tau\right)}
$$

Here $A(i)$ is every index except $i$, $N(i) = \{p \in A(i) : s_{i,p} \geq c\}$,
and $s_{i,p}$ is the Jaccard index of the two label sets.

**Label contract**: `labels` is a boolean tensor of shape `(batch, classes)`,
and is required even with `indices_tuple`, because it defines the weights. With
`ref_emb`, `ref_labels` has the same form and the same number of classes.
Another dtype raises `TypeError`, and another shape raises `ValueError`; convert
multi-hot labels with `labels.bool()`.

**Implementation notes**: Two empty label sets are identical, so their
similarity is 1, as the paper specifies for identical sets; Eq. 4 itself is
undefined for them. An empty set has similarity 0 with any other set. Without
`indices_tuple`, a similarity of exactly `threshold` is a positive and a smaller
one is a negative. With `threshold=0`, every other row is a positive, and rows
with no shared labels contribute zero weight but still count in $|N(i)|$.

A supplied `indices_tuple` uses the standard
[pair or triplet indices](../losses.md#index-tuples) and is used as given:
its pairs define $N(i)$ and $A(i)$ for each anchor, `threshold` is not applied,
and the weights still come from the labels. Self-comparisons are excluded
only when `ref_emb` is `None` and `indices_tuple` is `None`.
`loss_func.pair_indices(labels, ref_labels=None)` returns the thresholded
[`PairIndices`](../types.md#pairindices).

An anchor without positives has no loss term. The loss is computed in float32
for float16 and bfloat16 embeddings and returned in the embeddings' dtype.
The weights are computed with one matrix product over the labels, so
references cost no intermediate tensor proportional to the number of classes.

**Supported combinations**: Reducers other than
[ClassWeightedReducer](../reducers.md#classweightedreducer), `indices_tuple`,
`ref_emb`, [MultipleLosses](multiple_losses.md) of label-set losses, and
[CrossBatchMemory](cross_batch_memory.md#label-sets). Miners and
[SelfSupervisedLoss](self_supervised_loss.md) take one label per sample, and
[DistributedLossWrapper](../distributed.md) validates one label per sample, so
these raise `ValueError`.

**Paper settings**: The paper sums the per-anchor losses; the default reducer
averages the positive ones. Pass `reducer=reducers.SumReducer()` for the
paper's sum. The default `threshold=0.5` is the paper's best CelebA setting of
0.1, 0.3, and 0.5. The paper does not state a temperature, and the default of
0.1 follows SupCon. Eq. 4 also defines similarity for label counts, while the
paper's experiments use binary attributes; this loss takes label sets. With
`threshold=1`, it equals [SupConLoss](supcon_loss.md) with one integer label per
distinct set.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.MultiSupConLoss(temperature=0.1, threshold=0.5)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.rand(32, 40) < 0.25
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **temperature**: The temperature $\tau$.
- **threshold**: The similarity $c$ in `[0, 1]` that makes a pair positive.
  Another value raises `ValueError`.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: One loss per anchor that has a positive, indexed by anchor position
  in `embeddings`.
