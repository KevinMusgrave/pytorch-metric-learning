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

It returns the loss as a detached scalar tensor with the dtype of the full loss: that of
`embeddings`, or float32 for float16 and bfloat16 embeddings with an
[unnormalized distance](../distances.md#calling-a-distance). It
has already called `backward`, so gradients have accumulated in the model, and in
any learnable parameters of `loss_func` such as those of a custom distance, the same
way as after `loss.backward()`. Do not call `backward` again.

Gradient state matches as well as values. When every tuple is already satisfied, or
the batch has no tuples, the encoder receives zero gradients rather than `None`, so an
optimizer with weight decay steps as it would after `loss.backward()`. A parameter
the objective never reaches keeps `grad=None`.

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
whole reference set, backpropagates to a detached copy of the embeddings and to the
loss's learnable parameters, and frees its graph. The summed gradient then passes once
through the encoder. Each term keeps
a running sum and count over all chunks, and the division happens at the end, so the
denominators are those of the whole batch and not a mean of chunk means. Float16 and
bfloat16 embeddings and parameters are computed and accumulated in float32, then cast
back when the gradient is delivered.

A chunk keeps what each input's dtype selects in a full loss: the normalization `eps`
and the direct [`LpDistance`](../distances.md#lpdistance) kernel for low-precision rows,
so `embeddings` and `ref_emb` of different dtypes each keep their own. It does not
reproduce float16 rounding inside a normalized loss, so a low-precision result
can differ from the full loss by that rounding, and is then usually closer to float64.

Tensor hooks on `embeddings`, `ref_emb`, and learnable parameters run once, on the
final gradient. Module hooks are different: a forward hook on a loss's `distance`
runs for each chunk. While a chunk runs, `loss_func` holds working copies of its
learnable parameters in place of the originals, and the originals return when the
chunk ends, so do not share `loss_func` with another thread during a call.

Only one chunk's matrix and tuples exist at a time, which is the memory this
function targets. The encoder's own graph is unchanged, and each loss term holds a
`(N, D)` gradient buffer, plus one for each learnable parameter of the loss. On an
NVIDIA A40 with 2048 embeddings of 128 dimensions and chunks of 16, peak memory fell
from 291 MiB to 7.7 MiB for NTXentLoss and from 997 MiB to 11 MiB for
TripletMarginLoss. Time goes up, from 11 ms to 1044 ms and from 32 ms to 971 ms. Use
this function for a batch that does not fit in memory, not for speed.

## Supported losses

A loss qualifies only if its objective is a sum of terms that each depend on one
anchor's row:

- `CircleLoss`
- `ContrastiveLoss`
- `DecoupledContrastiveLoss` without `sigma`
- `GeneralizedLiftedStructureLoss`
- `MultiSimilarityLoss`
- `NTXentLoss`
- `SelectivelyContrastiveTripletLoss`
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
