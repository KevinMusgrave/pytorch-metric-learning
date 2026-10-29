# SelfSupervisedLoss

A common use case is to have `embeddings` and `ref_emb` be augmented versions of each other. For most losses, you have to create labels to indicate which `embeddings` correspond with which `ref_emb`.

`SelfSupervisedLoss` is a wrapper that takes care of this by creating labels internally. It assumes that:

- `ref_emb[i]` is an augmented version of `embeddings[i]`.
- `ref_emb[i]` is the only augmented version of `embeddings[i]` in the batch.

```python
losses.SelfSupervisedLoss(loss, symmetric=True)
```

**Parameters**:

- **loss**: The loss function to be wrapped.
- **symmetric**: If `True`, then the embeddings in both `embeddings` and `ref_emb` are used as anchors. If `False`, then only the embeddings in `embeddings` are used as anchors.

Example usage:

```python
loss_fn = losses.SelfSupervisedLoss(losses.TripletMarginLoss())
loss = loss_fn(embeddings, ref_emb)
```

**Supported Loss Functions**

Any `BaseMetricLossFunction` that uses labels. With `symmetric=False`, the loss
also receives `ref_emb`, so it must [accept `ref_emb`](../losses.md#supported-arguments).
Other modules raise `TypeError` at construction; a loss that rejects references
raises `ValueError` on the first asymmetric call.
