# CrossBatchMemory

This wraps a loss function, and implements [Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf){target=_blank}. It stores embeddings from previous iterations in a queue, and uses them to form more pairs/triplets with the current iteration's embeddings.

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.CrossBatchMemory(loss, embedding_size, memory_size=1024, miner=None, *, label_dtype=torch.long)
```

**Parameters**:

- **loss**: The loss function to be wrapped. For example, you could pass in `ContrastiveLoss()`.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **memory_size**: The size of the memory queue.
- **miner**: An optional [tuple miner](../miners.md), which will be used to mine pairs/triplets from the memory queue.
- **label_dtype**: The dtype of the label queue. Incoming labels are cast to it when the cast is safe under `torch.can_cast`; otherwise the call raises `ValueError`. Use a floating dtype for non-integer labels.

The queue buffers belong to the module. Move it with `loss_fn.to(device, dtype)`
so the embedding queue matches the embeddings; outside autocast a mismatch
raises `ValueError`. Forward never moves or casts the queue.
Under autocast, the incoming embeddings are converted to the queue's dtype for
mining and loss computation. Anchor gradients flow through this conversion;
enqueued embeddings are detached, and the queue keeps its configured precision.

**Forward function**

```python
loss_fn(embeddings, labels, indices_tuple=None, enqueue_mask=None)
```

As shown above, CrossBatchMemory comes with a 4th argument in its `forward` function:

- **enqueue_mask**: A boolean tensor where `enqueue_mask[i]` is True if `embeddings[i]` should be added to the memory queue. This enables CrossBatchMemory to be used in self-supervision frameworks like [MoCo](https://arxiv.org/pdf/1911.05722.pdf). See the [memory recipe](../recipes/training.md#memory-warmup-contract) for caller-controlled queue filling.

With an enqueue mask, selected rows become detached references and unselected
rows are anchors. Without a mask, all rows are enqueued and used as anchors,
with self-comparisons excluded. The number of enqueued rows must not exceed
`memory_size`. An empty enqueue preserves the existing queue; an empty batch
returns a differentiable zero. If no references have been enqueued, the loss is
also zero.

Each forward snapshots the occupied references, so accumulating multiple forward
losses before calling backward does not let later queue writes change an earlier
loss's gradients. This copies the occupied queue and costs memory proportional
to its size; queued embeddings never retain their original computation graphs.

Save the loss module's `state_dict` to resume the queue. It includes the memory
buffers, queue cursor, and filled flag, and supports loading with
`torch.load(..., weights_only=True)`. Construct the same loss and queue dimensions
before loading. Use `load_state_dict(state, assign=True)` to retain the saved
tensor dtypes, and construct optimizers after loading. Older checkpoints that
contain only the buffers lack the `queue_state` entry, which strict loading
reports as a missing key.

**Supported Loss Functions**

The listed types and subclasses that preserve their contracts are supported.
`supported_losses()` returns a tuple of types. ProxyNCA and its subclasses
are excluded because their learned proxies do not accept separate reference
embeddings. Other unsupported losses raise `TypeError` at construction.

- [AngularLoss](angular_loss.md#angularloss)
- [CircleLoss](circle_loss.md#circleloss)
- [ContrastiveLoss](contrastive_loss.md#contrastiveloss)
- [GeneralizedLiftedStructureLoss](generalized_lifted_structure_loss.md#generalizedliftedstructureloss)
- [IntraPairVarianceLoss](intra_pair_variance_loss.md#intrapairvarianceloss)
- [MarginLoss](margin_loss.md#marginloss)
- [MultiSimilarityLoss](multi_similarity_loss.md#multisimilarityloss)
- [NCALoss](nca_loss.md#ncaloss)
- [NTXentLoss](ntxent_loss.md#ntxentloss)
- [SignalToNoiseRatioContrastiveLoss](signal_to_noise_ratio_contrastive_loss.md#signaltonoiseratiocontrastiveloss)
- [SupConLoss](supcon_loss.md#supconloss)
- [TripletMarginLoss](triplet_margin_loss.md#tripletmarginloss)
- [TupletMarginLoss](tuplet_margin_loss.md#tupletmarginloss)

**Reset queue**

The queue can be cleared like this:

```python
loss_fn.reset_queue()
```
