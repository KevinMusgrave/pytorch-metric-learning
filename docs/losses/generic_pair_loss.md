# GenericPairLoss

A base class for losses over positive and negative pairs.

```python {.signature}
losses.GenericPairLoss(**options)
```

**Implementation notes**: Its `compute_loss` converts labels or a supplied
`indices_tuple` to `types.PairIndices`, returns `{}` when there are no pairs,
computes the distance matrix between `embeddings` and the reference, and calls
`pair_loss`. Labels or an `indices_tuple` are required; with neither, it raises
`ValueError`. Subclasses implement `pair_loss(self, mat, pairs)`.

`mat` is the `(N, N)` distance matrix of `embeddings` with itself when `ref_emb`
is `None`, or the `(N, M)` matrix between `embeddings` and `ref_emb`. Without
`ref_emb`, an embedding is never its own positive. `pairs` is a `PairIndices`
`(a1, p, a2, n)`. Two static helpers extract the common views:
`self.pair_values(mat, pairs)` returns the positive-pair and negative-pair
distances, and `self.pair_masks(mat, pairs)` returns boolean positive and
negative masks over `mat`. `pair_loss` returns a
[`types.LossDict`](../types.md#lossdict), as in this example.

**Example**:

```python
import torch
from pytorch_metric_learning import losses


class HingePairLoss(losses.GenericPairLoss):
    def pair_loss(self, mat, pairs):
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        a1, p, a2, n = pairs
        pos_loss = torch.relu(self.distance.margin(pos_pairs, 0.2))
        neg_loss = torch.relu(self.distance.margin(0.8, neg_pairs))
        return {
            "pos_loss": {"losses": pos_loss, "indices": (a1, p)},
            "neg_loss": {"losses": neg_loss, "indices": (a2, n)},
        }


loss_func = HingePairLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()

ref_emb = torch.randn(10, 128)
ref_labels = torch.randint(0, 8, (10,))
loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
```

**Parameters**:

- **options**: The [loss options](../losses.md#loss-options).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
