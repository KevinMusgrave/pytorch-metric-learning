# SelfSupervisedLoss

A wrapper for the common case where `embeddings` and `ref_emb` are augmented
views of each other. It creates the labels that pair them, so the wrapped loss
can be used without labels.

```python {.signature}
losses.SelfSupervisedLoss(loss, symmetric=True)
```

**Implementation notes**: `SelfSupervisedLoss` is a `torch.nn.Module` whose
forward is:

```python {.signature}
loss_func(embeddings, ref_emb)
```

It assumes that `ref_emb[i]` is an augmented view of `embeddings[i]` and is the
only augmented view of `embeddings[i]` in the batch, so both tensors have
shape `(N, D)`. It assigns label `i` to `embeddings[i]` and `ref_emb[i]`.

- With `symmetric=True`, it concatenates the two tensors into one batch of
  `2N` embeddings with labels `[0..N-1, 0..N-1]` and calls
  `loss(embeddings, labels)`, so every embedding of both views is an anchor.
  Any `BaseMetricLossFunction` that uses labels works, as does a
  [MultipleLosses](multiple_losses.md) of such losses.
- With `symmetric=False`, it calls
  `loss(embeddings, labels, ref_emb=ref_emb, ref_labels=labels)`, so only
  `embeddings` are anchors and the wrapped loss must
  [accept `ref_emb`](../losses.md#supported-arguments). A loss that rejects
  references raises `ValueError` on the first call.

A `loss` that is not a `BaseMetricLossFunction` or `MultipleLosses` raises
`TypeError` at construction.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.SelfSupervisedLoss(losses.TripletMarginLoss())
embeddings = torch.randn(32, 128, requires_grad=True)
ref_emb = torch.randn(32, 128, requires_grad=True)
loss = loss_func(embeddings, ref_emb)
loss.backward()
```

**Parameters**:

- **loss**: The loss function to wrap.
- **symmetric**: If `True`, the embeddings in both `embeddings` and `ref_emb`
  are anchors. If `False`, only the embeddings in `embeddings` are anchors.
