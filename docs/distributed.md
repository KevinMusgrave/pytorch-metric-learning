# Distributed

Wrap a tuple loss or miner with these when using PyTorch's [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (i.e. multiprocessing).

With no initialized process group, or an initialized group containing one rank,
both wrappers behave like the wrapped component. The loss wrapper warns when
there is no process group. Values, gradients, and mining indices are unchanged.

`DistributedLossWrapper` also accepts `CrossBatchMemory` with `efficient=False`.
Pass `enqueue_mask` by keyword to select rows to enqueue; unselected rows are
anchors. It preserves queue updates in single-process use. Explicit `ref_emb`
and `ref_labels` are unsupported for memory losses, and `enqueue_mask` is
unsupported for ordinary losses. These combinations raise `ValueError`.

## DistributedLossWrapper

```python
utils.distributed.DistributedLossWrapper(loss, efficient=False)
```

**Parameters**:

- **loss**: The loss function to wrap
- **efficient**:
  - `False` (default): each rank computes the loss on the complete gathered batch.
    The returned value equals the single-process global-batch loss. Autograd-aware
    gathering and DDP's gradient average recover that objective's parameter
    gradients, including trainable loss parameters. Each rank repeats the
    global-batch computation.
  - `True`: each rank uses local anchors, local reference gradients, and detached
    remote references. The returned value is world size times that rank's local
    loss. After DDP averaging, gradients equal the sum of these local surrogate
    gradients. This is a different objective, including for uneven batches; it
    is not an equivalent faster implementation of the default mode.

Use the default mode when matching a single-process reference. For a learnable
loss, put the encoder and wrapped loss in a module and wrap that module in DDP.
Include both sets of parameters in the optimizer. Do not multiply the returned
loss by world size again.

Gathered rows are ordered by rank, then by their original local order. Batch sizes
may differ across ranks, including empty ranks, subject to the wrapped loss's
own size requirements. The embedding width, dtype, gradient requirements, wrapper
mode, and presence of optional arguments must agree across ranks. Labels contain
one value per embedding; enqueue masks are boolean vectors of the local batch
length. Every rank must call forward and backward in the same order.

In full-batch mode, supplied mining indices use the gathered query/reference
coordinates and must be identical across ranks. The distributed miner produces
these coordinates. In local-anchor mode, anchor indices refer to the local batch
and positive/negative indices refer to the gathered reference batch. Explicit
reference embeddings are a separate set: equal row positions do not identify
self-matches. The wrapper does not add reference-embedding support to a loss
that otherwise rejects it.

Memory replicas must begin with the same state. Their queues receive the same
rank-ordered embeddings, labels, and masks. Queue embeddings stay detached, while
anchor gradients flow through the gather. `CrossBatchMemory` supports only the
full-batch mode.

The full-batch scalar now matches the unscaled global loss. Earlier releases
multiplied it by world size; that corrected encoder gradients with detached
gathering but over-scaled trainable loss-parameter gradients. Code that divides
the wrapper's output for logging should remove that division.

Example usage:

```python
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist

loss_func = losses.ContrastiveLoss()
loss_func = pml_dist.DistributedLossWrapper(loss_func)

loss = loss_func(embeddings, labels)
```

## DistributedMinerWrapper

```python
utils.distributed.DistributedMinerWrapper(miner, efficient=False)
```

**Parameters**:

- **miner**: The miner to wrap
- **efficient**: If your distributed loss function has `efficient=True` then you must also set the distributed miner's `efficient` to True.

Example usage:

```python
from pytorch_metric_learning import miners
from pytorch_metric_learning.utils import distributed as pml_dist

miner = miners.MultiSimilarityMiner()
miner = pml_dist.DistributedMinerWrapper(miner)

indices_tuple = miner(embeddings, labels)
loss = loss_func(embeddings, labels, indices_tuple)
```
