# RankedListLoss

[Ranked List Loss for Deep Metric Learning](https://arxiv.org/abs/1903.03238)

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.RankedListLoss(margin, Tn, imbalance=0.5, alpha=None, Tp=0, **kwargs)
```

**Parameters**:

The defaults correspond to the paper's RLL-Simpler variant, which it recommends for exploratory analysis: `Tp=0` weights positives uniformly, and `alpha=None` sets $\alpha = 1 + m/2$ for margin $m$.

- **margin** (float): margin between positive and negative set
- **imbalance** (float): tradeoff between positive and negative sets. As the name suggests this takes into account
  the imbalance between positive and negative samples in the dataset
- **alpha** (float): smallest distance between negative points
- **Tp & Tn** (float): temperatures for, respectively, positive and negative pairs weighting.
