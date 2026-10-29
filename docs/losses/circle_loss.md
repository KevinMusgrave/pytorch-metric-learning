# CircleLoss

[Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857)

```python
losses.CircleLoss(margin=0.4, gamma=80, **options)
```

**Equation**:

$$
\mathcal{L}_{circle} = \log \left[ 1 + \sum_{j=1}^{L} \exp\left(\gamma \alpha_n^j \left(s_n^j - \Delta_n\right)\right) \sum_{i=1}^{K} \exp\left(-\gamma \alpha_p^i \left(s_p^i - \Delta_p\right)\right) \right]
$$

For each anchor, $s_p^i$ and $s_n^j$ are its cosine similarities to its $K$ positives and $L$ negatives.

where

$$
\begin{aligned}
\alpha_p^i &= \left[O_p - s_p^i\right]_+ \\
\alpha_n^j &= \left[s_n^j - O_n\right]_+
\end{aligned}
$$

$$
\begin{aligned}
O_p &= 1 + m, & O_n &= -m, \\
\Delta_p &= 1 - m, & \Delta_n &= m
\end{aligned}
$$

**Implementation notes**: $\alpha_p$ and $\alpha_n$ are treated as constants
for gradients. The two sums are computed per anchor with log-sum-exp. An anchor
without a positive or without a negative has no loss term, so any reducer
averages only anchors that have both. The default reducer also skips anchors
whose loss is 0.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.CircleLoss(margin=0.25, gamma=256)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **margin**: The relaxation factor $m$, which controls the radius of the
  decision boundary. The paper uses 0.25 for face recognition and 0.4 for
  fine-grained image retrieval (birds, cars, and online products).
- **gamma**: The scale factor $\gamma$. The paper uses 256 for face recognition
  and 80 for fine-grained image retrieval.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: One loss per anchor that has a positive and a negative, indexed by
  anchor position in `embeddings`.
