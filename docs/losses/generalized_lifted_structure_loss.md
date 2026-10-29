# GeneralizedLiftedStructureLoss

[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)

A modification of the original [LiftedStructureLoss](lifted_structure_loss.md)
that supports a separate reference set.

```python
losses.GeneralizedLiftedStructureLoss(pos_margin=0.0, neg_margin=1.0, **options)
```

**Equation**:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{LG}} = \sum_{i=1}^{P} \sum_{a=1}^{K} \Bigg[ & \log \sum_{\substack{p=1 \\ p \neq a}}^{K} e^{D\left(f(x_a^i),\, f(x_p^i)\right)} \\
& + \log \sum_{\substack{j=1 \\ j \neq i}}^{P} \sum_{n=1}^{K} e^{m - D\left(f(x_a^i),\, f(x_n^j)\right)} \Bigg]_+
\end{aligned}
$$

Here the batch has $P$ classes with $K$ samples each, $x_a^i$ is sample $a$ of class $i$, $f$ is the embedding network, and $D$ is the distance. The outer sums run over all anchors, the first inner sum over their positives, and the second over their negatives.

**Implementation notes**: The loss is computed per anchor and averaged over
anchors rather than summed. Both log-sum-exp terms run over the anchor's pairs,
from labels or from `indices_tuple`. An anchor without positives or without
negatives has no loss term: the log-sum-exp of its empty set is $-\infty$, so
the equation's hinge is 0 for it, and the mean runs over the other anchors.

**Paper settings**: The paper sums the per-anchor losses; the default reducer
averages them. Pass `reducer=reducers.SumReducer()` for the paper's sum.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.GeneralizedLiftedStructureLoss(neg_margin=0.5)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
ref_emb = torch.randn(64, 128)
ref_labels = torch.randint(0, 8, (64,))
loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
loss.backward()
```

**Parameters**:

- **pos_margin**: A margin subtracted inside the positive term, $e^{D - \text{pos\_margin}}$.
  The paper uses 0, so it does not appear in the equation.
- **neg_margin**: The margin $m$. The paper uses values between 0.1 and 1.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per anchor that has a positive and a negative, indexed by
  anchor position in `embeddings`.
