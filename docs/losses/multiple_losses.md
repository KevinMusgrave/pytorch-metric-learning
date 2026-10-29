# MultipleLosses

A weighted sum of several losses. Pass a mapping from names to initialized loss functions; `forward` returns the weighted sum of all of them.

```python
losses.MultipleLosses(losses, miners=None, weights=None)
```

**Parameters**:

- **losses**: A mapping from names to initialized loss functions, for example `{"metric": loss_A, "tcm": loss_B}`.
- **miners**: Optional. A mapping from loss names to miners. Each miner's output is passed to its loss; losses without a miner receive the `indices_tuple` given to `forward`. Supplying both a miner mapping and an `indices_tuple` raises `ValueError`.
- **weights**: Optional. A mapping with exactly the same keys as `losses`, giving the weight of each loss. The default weight is 1.

```python
from pytorch_metric_learning import losses, miners

loss_func = losses.MultipleLosses(
    losses={"tuplet": losses.TupletMarginLoss(), "variance": losses.IntraPairVarianceLoss()},
    weights={"tuplet": 1.0, "variance": 0.5},
    miners={"tuplet": miners.MultiSimilarityMiner()},
)
loss = loss_func(embeddings, labels)
```
