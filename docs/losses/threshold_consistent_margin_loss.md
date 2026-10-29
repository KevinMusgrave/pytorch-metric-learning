# ThresholdConsistentMarginLoss

[Threshold-Consistent Margin Loss for Open-World Deep Metric Learning](https://arxiv.org/abs/2307.04047)

```python
losses.ThresholdConsistentMarginLoss(
    lambda_plus=1.0, lambda_minus=1.0, margin_plus=0.9, margin_minus=0.5, **options
)
```

**Equation**:

$$
L_{\operatorname{TCM}} = \lambda^+ \cdot \frac{\sum_{s \in S^+} \left(m^+ - s\right) \cdot \mathbf{1}_{s \le m^+}}{\sum_{s \in S^+} \mathbf{1}_{s \le m^+}} + \lambda^- \cdot \frac{\sum_{s \in S^-} \left(s - m^-\right) \cdot \mathbf{1}_{s \ge m^-}}{\sum_{s \in S^-} \mathbf{1}_{s \ge m^-}}
$$

Here $S^+$ and $S^-$ are the cosine similarities of the batch's positive and negative pairs. A term whose denominator is zero contributes 0.

**Implementation notes**: This loss acts as a regularizer and is usually
combined with another metric loss through
[MultipleLosses](multiple_losses.md). The pairs come from labels or from the
supplied `indices_tuple`; with `ref_emb`, positives and negatives come from the
reference set. The similarities are computed with
`torch.nn.functional.cosine_similarity`, so the distance's `power` has no
effect.

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

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: Already reduced: the value of $L_{\operatorname{TCM}}$. Every
  reducer returns it unchanged.
