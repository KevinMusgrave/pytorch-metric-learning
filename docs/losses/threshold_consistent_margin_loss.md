# ThresholdConsistentMarginLoss

[Threshold-Consistent Margin Loss for Open-World Deep Metric Learning](https://arxiv.org/abs/2307.04047)

```python
losses.ThresholdConsistentMarginLoss(
    lambda_plus=1.0,
    lambda_minus=1.0,
    margin_plus=0.9,
    margin_minus=0.5,
    *,
    distance=None,
    embedding_regularizer=None,
    embedding_reg_weight=1.0,
)
```

**Equation**:

$$
L_{\operatorname{TCM}} = \lambda^+ \cdot \frac{\sum_{s \in S^+} \left(m^+ - s\right) \cdot \mathbf{1}_{s \le m^+}}{\sum_{s \in S^+} \mathbf{1}_{s \le m^+}} + \lambda^- \cdot \frac{\sum_{s \in S^-} \left(s - m^-\right) \cdot \mathbf{1}_{s \ge m^-}}{\sum_{s \in S^-} \mathbf{1}_{s \ge m^-}}
$$

Here $S^+$ and $S^-$ are the similarities of the batch's positive and negative pairs; the paper uses cosine similarity. A term whose denominator is zero contributes 0.

**Implementation notes**: This loss acts as a regularizer and is usually
combined with another metric loss through
[MultipleLosses](multiple_losses.md). The pairs come from labels or from the
supplied `indices_tuple`; with `ref_emb`, positives and negatives come from the
reference set. The similarities are the configured distance's pairwise values,
so `margin_plus` and `margin_minus` are on that similarity's scale.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.MultipleLosses(
    {
        "metric": losses.MultiSimilarityLoss(),
        "tcm": losses.ThresholdConsistentMarginLoss(),
    }
)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **lambda_plus**: The weight $\lambda^+$ of the anchor-positive term.
- **lambda_minus**: The weight $\lambda^-$ of the anchor-negative term.
- **margin_plus**: The threshold $m^+$; positive pairs with similarity at most
  $m^+$ contribute.
- **margin_minus**: The threshold $m^-$; negative pairs with similarity at least
  $m^-$ contribute.
- **distance**, **embedding_regularizer**, **embedding_reg_weight**: See
  [loss options](../losses.md#loss-options).

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). Any similarity
  (`is_inverted=True`) is accepted; a distance raises `TypeError`.

**Reducer**:

- None. The loss reduces its own value, so it has no `reducer` option; passing
  `reducer` raises `TypeError`.

**Loss terms**:

- **loss**: The value of $L_{\operatorname{TCM}}$, with `indices=None`.
