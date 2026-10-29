# MultipleLosses

A weighted sum of several losses.

```python
losses.MultipleLosses(losses, miners=None, weights=None)
```

**Implementation notes**: `MultipleLosses` is a `torch.nn.Module` whose forward
is:

```python
loss_func(embeddings, labels, indices_tuple=None)
```

It returns $\sum_k w_k L_k$ over the named losses. A loss with a configured
miner receives that miner's output for `(embeddings, labels)`; every other loss
receives the `indices_tuple` given to `forward`. Supplying both a miner mapping
and an `indices_tuple` raises `ValueError`. The losses and miners are
registered as submodules, so `loss_func.to(...)` and `loss_func.parameters()`
cover them.

**Parameters**:

- **losses**: A non-empty mapping from names to initialized loss modules, for
  example `{"metric": loss_a, "tcm": loss_b}`. Anything else raises `TypeError`.
- **miners**: Optional. A mapping from loss names to miners. A key that is not
  in `losses` raises `ValueError`.
- **weights**: Optional. A mapping with exactly the same keys as `losses`,
  giving the weight of each loss; other keys raise `ValueError`. The default
  weight is 1.0.

```python
from pytorch_metric_learning import losses, miners

loss_func = losses.MultipleLosses(
    losses={"tuplet": losses.TupletMarginLoss(), "variance": losses.IntraPairVarianceLoss()},
    weights={"tuplet": 1.0, "variance": 0.5},
    miners={"tuplet": miners.MultiSimilarityMiner()},
)
loss = loss_func(embeddings, labels)
```
