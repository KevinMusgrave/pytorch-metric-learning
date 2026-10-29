# CrossBatchMemory

[Cross-Batch Memory for Embedding Learning](https://arxiv.org/abs/1912.06798)

A wrapper that stores embeddings from previous iterations in a queue and uses
them as references for the current iteration's embeddings, forming more pairs
or triplets than one batch provides.

```python
losses.CrossBatchMemory(loss, embedding_size, memory_size=1024, miner=None, *, label_dtype=torch.long)
```

**Implementation notes**: `CrossBatchMemory` is a `torch.nn.Module` whose
forward is:

```python
loss_fn(embeddings, labels, indices_tuple=None, enqueue_mask=None)
```

Each call builds a snapshot of the occupied queue with the new rows written in,
mines pairs or triplets between `embeddings` and the snapshot, calls
`loss(embeddings, labels, indices_tuple, ref_emb=snapshot, ref_labels=...)`, and
then commits the new rows to the queue. All inputs are validated before any
state changes, and a wrapped loss that raises leaves the queue unchanged.

- Without `enqueue_mask`, every row is enqueued and is also an anchor; pairs of
  an anchor with its own enqueued copy are removed.
- With `enqueue_mask`, the selected rows are enqueued as references and the
  other rows are anchors.
- The number of enqueued rows must not exceed `memory_size`; otherwise the call
  raises `ValueError`. An empty enqueue preserves the queue. An empty batch, or
  a batch with no references, gives a differentiable zero.
- Without a miner, all pairs between anchors and the snapshot are used. A
  supplied `indices_tuple` is added to those pairs or to the miner's tuples; its
  anchors index `embeddings` and its positives and negatives index rows of the
  snapshot. `indices_tuple` and `enqueue_mask` cannot be combined
  (`ValueError`).

The wrapped loss receives the snapshot as `ref_emb`, so it must be a
`BaseMetricLossFunction` that [accepts `ref_emb`](../losses.md#supported-arguments).
Any other module raises `TypeError` at construction; a loss that rejects
references raises `ValueError` on the first call.

The queue buffers `embedding_memory` and `label_memory` belong to the module.
Move it with `loss_fn.to(device, dtype)` so the embedding queue matches the
embeddings; outside autocast a mismatch raises `ValueError`, and forward never
moves or casts the queue. Under autocast, the incoming embeddings are converted
to the queue's dtype for mining and loss computation. Anchor gradients flow
through this conversion; enqueued embeddings are detached, and the queue keeps
its configured precision. Each snapshot copies the occupied queue, so memory
grows with its size, and accumulating several forward losses before one
backward pass does not let later queue writes change an earlier loss's
gradients.

Save the module's `state_dict` to resume the queue. It holds the memory
buffers and a `queue_state` entry with the queue cursor and filled flag, and it
loads with `torch.load(..., weights_only=True)`. Construct the same loss and
queue dimensions before loading, use `load_state_dict(state, assign=True)` to
retain the saved tensor dtypes, and construct optimizers after loading. A state
dict without `queue_state` is reported as a missing key under strict loading.
`loss_fn.reset_queue()` empties the queue.

**Example**:

```python
import torch
from pytorch_metric_learning import losses, miners

loss_func = losses.CrossBatchMemory(
    losses.ContrastiveLoss(),
    embedding_size=128,
    memory_size=1024,
    miner=miners.MultiSimilarityMiner(),
)
for _ in range(3):
    embeddings = torch.randn(32, 128, requires_grad=True)
    labels = torch.randint(0, 8, (32,))
    loss = loss_func(embeddings, labels)
    loss.backward()
loss_func.reset_queue()
```

**Parameters**:

- **loss**: The loss function to wrap, for example `ContrastiveLoss()`.
- **embedding_size**: The size of the embeddings passed to the loss. Other
  sizes raise `ValueError`.
- **memory_size**: The number of rows in the queue.
- **miner**: An optional [miner](../miners.md) that mines pairs or triplets
  between the anchors and the snapshot.
- **label_dtype**: The dtype of the label queue. Incoming labels are converted
  to it only when the conversion round-trips exactly; otherwise the call raises
  `ValueError`. Use a floating dtype for non-integer labels.
- **enqueue_mask** (forward argument): A boolean tensor of shape `(N,)` on the
  embeddings' device, where `enqueue_mask[i]` is `True` if `embeddings[i]`
  should be added to the queue. This supports self-supervision frameworks such
  as [MoCo](https://arxiv.org/abs/1911.05722); see the
  [memory recipe](../recipes/training.md#memory-warmup-contract) for
  caller-controlled queue filling. A non-boolean mask raises `TypeError`; a
  wrong shape or device raises `ValueError`.
