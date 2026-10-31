# anchor_batched_backward

Computes the gradient of a tuple loss in chunks of anchors, so large batches need
less memory. The result equals `loss_func(embeddings, labels)` followed by
`backward()`, to floating-point rounding.

```python {.signature}
losses.anchor_batched_backward(
    loss_func,
    embeddings,
    labels,
    batch_size,
    *,
    ref_emb=None,
    ref_labels=None,
)
```

- **loss_func**: A supported loss; see [supported losses](#supported-losses).
- **embeddings**: A `(N, D)` tensor tracking gradients, usually model output.
- **labels**: A `(N,)` tensor of labels.
- **batch_size**: The most anchors in one chunk. It must be positive.
- **ref_emb**, **ref_labels**: Optional reference set of shape `(M, D)` and `(M,)`,
  following the [reference-set rule](../concepts.md#reference-sets). A `ref_emb`
  that tracks gradients receives them; a detached one does not.

It returns the loss as a detached scalar tensor with the dtype of `embeddings`. It
has already called `backward`, so gradients have accumulated in the model the same
way as after `loss.backward()`. Do not call `backward` again.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

model = torch.nn.Linear(16, 8)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_func = losses.NTXentLoss()
inputs = torch.randn(64, 16)
labels = torch.arange(64) // 4

optimizer.zero_grad()
loss = losses.anchor_batched_backward(loss_func, model(inputs), labels, batch_size=16)
optimizer.step()
assert loss.ndim == 0 and not loss.requires_grad
```

## How it works

A loss's graph holds its matrix and every tuple until `backward()`, so its size
follows the whole batch. Here each chunk compares up to `batch_size` anchors with the
whole reference set, backpropagates to a detached copy of the embeddings, and frees
its graph. The summed gradient then passes once through the encoder. Each term keeps
a running sum and count over all chunks, and the division happens at the end, so the
denominators are those of the whole batch and not a mean of chunk means. Float16 and
bfloat16 embeddings accumulate in float32.

Peak memory falls because the matrix and tuples of only one chunk exist at a time.
The encoder's own graph and the `(N, D)` gradient are unchanged. See
[measuring performance](../performance.md) for measurements.

## Supported losses

A loss qualifies only if its objective is a sum of terms that each depend on one
anchor's row:

- `CircleLoss`
- `ContrastiveLoss`
- `GeneralizedLiftedStructureLoss`
- `MultiSimilarityLoss`
- `NTXentLoss`
- `SignalToNoiseRatioContrastiveLoss`
- `SupConLoss`
- `TripletMarginLoss`
- `TupletMarginLoss`

Any other loss raises `ValueError`, because batching it would silently change the
objective. `IntraPairVarianceLoss`, for example, subtracts a mean over every pair.

These cases also raise `ValueError`:

- A reducer other than `MeanReducer` or `AvgNonZeroReducer`.
- An `embedding_regularizer`.
- `triplets_per_anchor` other than `"all"` on `TripletMarginLoss`.
- A non-positive `batch_size`, or inputs that do not track gradients.

There is no `indices_tuple` argument: every pair or triplet from the labels is used,
so this function cannot follow a miner. Use a miner's
[`tuples_per_anchor`](../miners.md#bounding-the-output) to bound mined tuples.
