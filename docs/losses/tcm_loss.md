# ThresholdConsistentMarginLoss

[Threshold-Consistent Margin Loss for Open-World Deep Metric Learning](https://arxiv.org/pdf/2307.04047){target=_blank}

This loss acts as a form of regularization and is usually combined with another metric loss function.

```python
losses.ThresholdConsistentMarginLoss(
    lambda_plus=1.0,
    lambda_minus=1.0,
    margin_plus=0.9,
    margin_minus=0.5,
    **kwargs
)
```

**Equation**:

$$
L_{\operatorname{TCM}} = \lambda^+ \cdot \frac{\sum_{s \in S^+} \left(m^+ - s\right) \cdot \mathbf{1}_{s \le m^+}}{\sum_{s \in S^+} \mathbf{1}_{s \le m^+}} + \lambda^- \cdot \frac{\sum_{s \in S^-} \left(s - m^-\right) \cdot \mathbf{1}_{s \ge m^-}}{\sum_{s \in S^-} \mathbf{1}_{s \ge m^-}}
$$

Here $S^+$ and $S^-$ are the cosine similarities of the batch's positive and negative pairs. A term whose denominator is zero contributes 0.

**Parameters**:

- **lambda_plus**: The scaling coefficient for the anchor-positive part of the loss. This is $\lambda^+$ in the above equation.
- **lambda_minus**: The scaling coefficient for the anchor-negative part of the loss. This is $\lambda^-$ in the above equation.
- **margin_plus**: The minimum anchor-positive similarity to be included in the loss. This is $m^+$ in the above equation.
- **margin_minus**: The maximum anchor-negative similarity to be included in the loss. This is $m^-$ in the above equation.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)
  - This is the only compatible distance.
