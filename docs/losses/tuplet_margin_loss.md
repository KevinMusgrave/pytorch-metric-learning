# TupletMarginLoss

[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)

```python
losses.TupletMarginLoss(margin=5.73, scale=64, **options)
```

**Equation**:

$$
\mathcal{L}_{tuplet} = \log\left(1 + \sum_{i=1}^{k-1} e^{s\left(\cos\theta_{an_i} - \cos\left(\theta_{ap} - \beta\right)\right)}\right)
$$

Here $\theta_{ap}$ and $\theta_{an_i}$ are the angles from the anchor to its positive and to its $i$-th negative. The sum runs over all $k - 1$ negatives of the anchor in the batch, or over its negative pairs in a supplied `indices_tuple`.

**Implementation notes**: Each positive pair contributes one loss term,
averaged by the default reducer. The negatives are combined with a per-anchor
log-sum-exp, and the loss is $\operatorname{softplus}(\mathrm{LSE} - s\cos(\theta_{ap} - \beta))$;
float16 and bfloat16 inputs accumulate in float32. A positive pair whose anchor
has no negatives contributes 0. Cosines are clamped just inside $[-1, 1]$
before $\arccos$, so identical anchor-positive pairs, such as duplicates from
sampling with replacement, have finite values and gradients.

The paper combines this loss with
[IntraPairVarianceLoss](intra_pair_variance_loss.md) through
[MultipleLosses](multiple_losses.md).

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

- **margin**: The angular margin $\beta$ in degrees, applied to positive pairs.
  The paper uses 5.73 degrees (0.1 radians).
- **scale**: The scale $s$.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per (anchor, positive) pair, indexed by `(anchors, positives)`.
