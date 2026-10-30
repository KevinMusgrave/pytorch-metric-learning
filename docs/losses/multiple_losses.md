# MultipleLosses

A weighted sum of several losses.

```python {.signature}
losses.MultipleLosses(losses, miners=None, weights=None)
```

**Implementation notes**: `MultipleLosses` is a `torch.nn.Module` whose forward
is:

```python {.signature}
loss_func(embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None)
```

This is the [`BaseMetricLossFunction` forward signature](../losses.md#forward-arguments).
It returns $\sum_k w_k L_k$ over the named losses, and passes `labels`,
`ref_emb`, and `ref_labels` to every loss. A loss with a configured miner
receives that miner's output for
`(embeddings, labels, indices_tuple, ref_emb, ref_labels)`, so the miner
selects from the given `indices_tuple` as
[candidates](../miners.md#candidates) when there is one; every other loss
receives the `indices_tuple` itself. Wrappers that build their own tuple, such
as `CrossBatchMemory` and `DistributedLossWrapper(objective="local_anchor")`,
therefore let per-loss miners refine the wrapper's pairs. The losses and miners are
registered as submodules, so `loss_func.to(...)` and `loss_func.parameters()`
cover them. Because the signature matches, a `MultipleLosses` can be wrapped
by [CrossBatchMemory](cross_batch_memory.md),
[SelfSupervisedLoss](self_supervised_loss.md), and
[DistributedLossWrapper](../distributed.md#distributedlosswrapper), and nested in
another `MultipleLosses`.

**Example**:

```python
import torch
from pytorch_metric_learning import losses, miners

loss_func = losses.MultipleLosses(
    losses={
        "tuplet": losses.TupletMarginLoss(),
        "variance": losses.IntraPairVarianceLoss(),
    },
    miners={"tuplet": miners.MultiSimilarityMiner()},
    weights={"tuplet": 1.0, "variance": 0.5},
)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **losses**: A non-empty mapping from names to `BaseMetricLossFunction` or
  `MultipleLosses` instances, for example `{"metric": loss_a, "tcm": loss_b}`.
  Anything else, including `CrossBatchMemory` and `VICRegLoss`, raises
  `TypeError`.
- **miners**: Optional. A mapping from loss names to miners. A key that is not
  in `losses` raises `ValueError`.
- **weights**: Optional. A mapping with exactly the same keys as `losses`,
  giving the weight of each loss; other keys raise `ValueError`. The default
  weight is 1.0.
