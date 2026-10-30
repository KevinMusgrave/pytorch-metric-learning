# SelfSupervisedLoss

A wrapper for the common case where two or more tensors are augmented views of
the same samples. It creates the labels that pair them, so the wrapped loss can
be used without labels.

```python {.signature}
losses.SelfSupervisedLoss(loss, symmetric=True)
```

**Implementation notes**: `SelfSupervisedLoss` is a `torch.nn.Module` whose
forward is:

```python {.signature}
loss_func(embeddings, ref_emb, *other_views)
```

It assumes that row `i` of every view shows the same sample and that no other
row does, so all `V` views have shape `(N, D)`; other shapes raise `ValueError`.
It assigns label `i` to row `i` of every view, which makes all views of a sample
positives of each other and all rows of other samples negatives.

- With `symmetric=True`, it concatenates the views into one batch of `V * N`
  embeddings with labels `[0..N-1]` repeated `V` times and calls
  `loss(embeddings, labels)`, so every embedding of every view is an anchor.
  Any `BaseMetricLossFunction` that uses labels works, as does a
  [MultipleLosses](multiple_losses.md) of such losses.
- With `symmetric=False`, it concatenates the views after the first into
  references with labels `[0..N-1]` repeated `V - 1` times and calls
  `loss(embeddings, labels, ref_emb=references, ref_labels=ref_labels)`, so only
  `embeddings` are anchors, and references are never compared with each other.
  The wrapped loss must [accept `ref_emb`](../losses.md#supported-arguments). A
  loss that rejects references raises `ValueError` on the first call.

Either way the views reach the wrapped loss in one call, so its reducer sees all
terms together instead of averaging separate per-view losses. A pair loss has
`V * (V - 1) * N` ordered positive pairs when symmetric and `(V - 1) * N`
otherwise. With more than two views, use a loss that uses every positive, such
as [NTXentLoss](ntxent_loss.md) or [SupConLoss](supcon_loss.md).
[NPairsLoss](n_pairs_loss.md) keeps one pair per label and ignores the others.

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

Pass further views as additional positional arguments:

```python
view_1, view_2, view_3 = torch.randn(3, 32, 128, requires_grad=True)
loss = losses.SelfSupervisedLoss(losses.SupConLoss())(view_1, view_2, view_3)
loss.backward()
```

**Parameters**:

- **loss**: The loss function to wrap.
- **symmetric**: If `True`, the embeddings of every view are anchors. If
  `False`, only the embeddings in `embeddings` are anchors.
