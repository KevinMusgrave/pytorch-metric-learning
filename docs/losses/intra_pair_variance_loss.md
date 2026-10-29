# IntraPairVarianceLoss

[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)

```python {.signature}
losses.IntraPairVarianceLoss(pos_eps=0.01, neg_eps=0.01, **options)
```

**Equation**:

$$
\mathcal{L}_{pos} = \mathbb{E}\left[(1 - \epsilon)\mu_{ap} - \cos\theta_{ap}\right]_+^2
$$

$$
\mathcal{L}_{neg} = \mathbb{E}\left[\cos\theta_{an} - (1 + \epsilon)\mu_{an}\right]_+^2
$$

Here $\theta_{ap}$ and $\theta_{an}$ are anchor-positive and anchor-negative angles, $\mu_{ap}$ and $\mu_{an}$ are the batch means of $\cos\theta_{ap}$ and $\cos\theta_{an}$, and $[\cdot]_+ = \max(0, \cdot)$.

**Implementation notes**: The means run over the positive pairs and over the
negative pairs of the batch, or of the supplied `indices_tuple`. Gradients flow
through $\mu_{ap}$ and $\mu_{an}$. With a distance where smaller means closer,
the inequalities are reversed.

The paper combines this loss with [TupletMarginLoss](tuplet_margin_loss.md)
through [MultipleLosses](multiple_losses.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.MultipleLosses(
    {"tuplet": losses.TupletMarginLoss(), "variance": losses.IntraPairVarianceLoss()},
    weights={"tuplet": 1.0, "variance": 0.5},
)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **pos_eps**: $\epsilon$ in $\mathcal{L}_{pos}$. The paper uses 0.01.
- **neg_eps**: $\epsilon$ in $\mathcal{L}_{neg}$. The paper uses 0.01.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **pos_loss**: One loss per (anchor, positive) pair, indexed by `(anchors, positives)`.
- **neg_loss**: One loss per (anchor, negative) pair, indexed by `(anchors, negatives)`.
