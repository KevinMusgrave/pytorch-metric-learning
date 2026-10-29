# Distributed

Wrap a loss or miner with these classes when training with PyTorch's
[DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
(DDP), so that each rank's anchors are compared with the embeddings of every
rank.

```python
from pytorch_metric_learning.distributed import (
    DistributedLossWrapper,
    DistributedMinerWrapper,
)
```

Without an initialized process group, or with a group of one rank, both wrappers
call the wrapped component directly. Values, gradients, and mining indices are
then those of the wrapped component.

## Objectives

Both wrappers take an `objective`, which must agree across ranks:

- `"global"` (default): each rank computes the loss on the complete gathered
  batch. The returned value equals the single-process loss on the global batch.
  Autograd-aware gathering and DDP's gradient average recover that objective's
  parameter gradients, including those of trainable loss parameters. Each rank
  repeats the global-batch computation.
- `"local_anchor"`: each rank uses its local anchors, local reference gradients,
  and detached remote references. The returned value is world size times that
  rank's local loss. After DDP averaging, gradients equal the sum of these local
  surrogate gradients. This is a different objective, including for uneven
  batches; it is not a faster implementation of `"global"`.

Use `"global"` to match a single-process reference. For a learnable loss, put
the encoder and the wrapped loss in one module and wrap that module in DDP.
Include both sets of parameters in the optimizer. Do not multiply the returned
loss by world size.

## Inputs across ranks

Gathered rows are ordered by rank, then by their original local order. Batch
sizes may differ across ranks, including empty ranks, subject to the wrapped
loss's own size requirements. The embedding width, dtype, gradient requirement,
objective, and presence of each optional argument must agree across ranks;
otherwise the wrapper raises `ValueError`. Labels contain one value per
embedding and must be on the embeddings' device; otherwise the wrapper raises
`ValueError`. Every rank must call forward and backward in the same order.

Explicit reference embeddings, `ref_emb` and `ref_labels`, are gathered the same
way and form a separate set: equal row positions do not identify
self-matches. The wrapper does not add reference-embedding support to a loss
that otherwise rejects it.

## DistributedLossWrapper

```python
distributed.DistributedLossWrapper(loss, objective="global")
```

**Parameters**:

- **loss**: The loss to wrap, a `BaseMetricLossFunction` or a
  [`CrossBatchMemory`](losses/cross_batch_memory.md#crossbatchmemory). Another
  type raises `TypeError`.
- **objective**: `"global"` or `"local_anchor"`; see [objectives](#objectives).
  Another value raises `ValueError`.

```python
wrapper(
    embeddings,
    labels=None,
    indices_tuple=None,
    ref_emb=None,
    ref_labels=None,
    enqueue_mask=None,
)
```

A supplied `indices_tuple` uses these coordinates:

- With `"global"`, query and reference indices refer to the gathered batch, and
  the tuple must be identical on every rank. `DistributedMinerWrapper` with the
  same objective produces these coordinates.
- With `"local_anchor"`, anchor indices refer to the local batch, and positive
  and negative indices refer to the gathered reference batch.

With `"local_anchor"` and no `ref_emb`, the loss receives the gathered batch as
`ref_emb`, so it cannot exclude self-pairs on its own. Without a supplied
`indices_tuple`, the wrapper passes every pair except each anchor's own gathered
row. Losses that take their pairs from `indices_tuple` honor that exclusion.
`NCALoss` uses `indices_tuple` only as miner weights, so under
`"local_anchor"` it counts each anchor as its own zero-distance positive; use
`"global"` with `NCALoss`.

`CrossBatchMemory` supports only the `"global"` objective; `"local_anchor"`
raises `ValueError` at construction. Pass `enqueue_mask` by keyword to select
the rows to enqueue; unselected rows are anchors. Memory replicas must begin
with the same state, and their queues receive the same rank-ordered embeddings,
labels, and masks. Queue embeddings stay detached, while anchor gradients flow
through the gather. `ref_emb` and `ref_labels` are not supported with
`CrossBatchMemory`, and `enqueue_mask` is not supported with other losses; both
combinations raise `ValueError`.

Example usage:

```python
import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.distributed import DistributedLossWrapper

loss_func = DistributedLossWrapper(losses.ContrastiveLoss())

embeddings = torch.randn(32, 16, requires_grad=True)
labels = torch.randint(0, 4, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

## DistributedMinerWrapper

```python
distributed.DistributedMinerWrapper(miner, objective="global")
```

**Parameters**:

- **miner**: The miner to wrap, a `BaseMiner`. Another type raises `TypeError`.
- **objective**: Must match the objective of the `DistributedLossWrapper` that
  receives the mined indices.
  - `"global"`: mines the gathered batch and returns indices in gathered
    coordinates.
  - `"local_anchor"`: mines local anchors against the gathered reference batch.
    Without `ref_emb`, each anchor's own gathered row is excluded from the
    candidates before mining.

```python
wrapper(embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None)
```

Candidate `indices_tuple` inputs are not supported across ranks and raise
`ValueError` when the process group has more than one rank.

Example usage:

```python
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distributed import (
    DistributedLossWrapper,
    DistributedMinerWrapper,
)

miner = DistributedMinerWrapper(miners.MultiSimilarityMiner())
loss_func = DistributedLossWrapper(losses.MultiSimilarityLoss())

embeddings = torch.randn(32, 16, requires_grad=True)
labels = torch.randint(0, 4, (32,))
indices_tuple = miner(embeddings, labels)
loss = loss_func(embeddings, labels, indices_tuple)
loss.backward()
```
