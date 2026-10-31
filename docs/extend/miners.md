# How to write custom mining functions

1. Extend [`BaseMiner`](../miners.md#baseminer).
2. Implement `mine(embeddings, labels, indices_tuple, ref_emb, ref_labels)`.
3. Return a `PairIndices` or `TripletIndices` from `pytorch_metric_learning.types`.

`BaseMiner` applies `tuples_per_anchor` to what `mine` returns. Forward the keyword
to `super().__init__` to offer it from a custom `__init__`.

To accept a [precomputed distance matrix](../losses.md#precomputed-distance-matrices),
set the class attribute `supports_mat = True` and add a keyword-only
`mat: Tensor | None = None` to `mine`. Use `mat` when it is given and call
`self.distance` otherwise. Without the attribute, passing `mat` raises `ValueError`.

`BaseMiner.forward` validates the inputs, calls `mine` under `torch.no_grad()`,
and [validates its output](../miners.md#output-validation). It passes the
arguments to `mine` as follows:

- **embeddings**, **labels**: The query embeddings and their labels. Anchor
  indices refer to these rows.
- **indices_tuple**: `None`, or the caller's candidate pairs or triplets as a
  `PairIndices` or `TripletIndices`. A miner honors candidates by selecting only
  from them; `None` means every pair is a candidate, and an empty tuple means
  none is.
- **ref_emb**, **ref_labels**: `None` for self-comparison, or a separate
  reference set. Positive and negative indices refer to the reference rows.

The `utils.indexing` helpers handle all of these cases:

- `convert_to_pairs(indices_tuple, labels, ref_labels)` returns the candidate
  pairs, or all positive and negative pairs when `indices_tuple` is `None`.
- `convert_to_triplets(indices_tuple, labels, ref_labels)` does the same for
  triplets.

Both follow the [reference-set rule](../concepts.md#reference-sets), as does
`self.distance(embeddings, ref_emb)`, so `mine` handles self-comparison and a
separate reference set with the same code. See the
[index utilities](../indexing.md) for their defaults and errors.

## An example pair miner

```python
import torch
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.types import PairIndices
from pytorch_metric_learning.utils import indexing


class ExamplePairMiner(BaseMiner):
    def __init__(self, margin=0.1, *, distance=None):
        super().__init__(distance=distance)
        self.margin = margin

    def mine(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
        pos_mask = self.distance.margin(mat[a1, p], self.margin) > 0
        neg_mask = self.distance.margin(self.margin, mat[a2, n]) > 0
        return PairIndices(a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask])
```

The `ExamplePairMiner` does the following:

- Computes the distance matrix between `embeddings` and the reference set.
- Takes the candidate positive and negative pairs.
- Returns the pairs that violate the margin. `distance.margin(x, y)` is `x - y`
  for distances and `y - x` for similarities, so the same code works for both
  orientations.

Example usage:

```python
embeddings = torch.randn(128, 512)
labels = torch.randint(0, 10, size=(128,))
miner = ExamplePairMiner()
pairs = miner(embeddings, labels)
assert isinstance(pairs, PairIndices)
assert torch.all(pairs.pos_anchors != pairs.positives)
```

## An example triplet miner

```python
from pytorch_metric_learning.types import TripletIndices


class ExampleTripletMiner(BaseMiner):
    def __init__(self, margin=0.1, *, distance=None):
        super().__init__(distance=distance)
        self.margin = margin

    def mine(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a, p, n = indexing.convert_to_triplets(indices_tuple, labels, ref_labels)
        triplet_margin = self.distance.margin(mat[a, n], mat[a, p])
        keep = triplet_margin <= self.margin
        return TripletIndices(a[keep], p[keep], n[keep])
```

This miner works like `ExamplePairMiner`, but selects triplets. Because both
miners start from their candidates, they compose with other miners:

```python
candidates = ExamplePairMiner(margin=0.5)(embeddings, labels)
triplets = ExampleTripletMiner()(embeddings, labels, indices_tuple=candidates)
assert isinstance(triplets, TripletIndices)
```

## Reference embeddings

With `ref_emb` and `ref_labels`, the miner returns anchors from `embeddings` and
positives and negatives from `ref_emb`:

```python
ref_emb = torch.randn(32, 512)
ref_labels = torch.randint(0, 10, size=(32,))
a1, p, a2, n = miner(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
assert a1.max() < len(embeddings) and p.max() < len(ref_emb)
```

No pair is excluded as a self-comparison with a supplied reference set; see
[reference sets](../concepts.md#reference-sets). To mine within one batch, leave
`ref_emb` as `None`.

## Validation

`BaseMiner.forward` checks that `labels`, `ref_emb`, `ref_labels`, and candidate
tensors are on the embeddings' device and that labels have one value per
embedding, raising `ValueError` otherwise. After `mine` returns, it checks for
three equal-length triplet tensors or four pair tensors whose anchor and partner
lengths match, all one-dimensional and on the embeddings' device. Candidate and
output index tensors must be `int64` or `int32`. Invalid types or index dtypes
raise `TypeError`; invalid shapes, lengths, or devices raise `ValueError`.
Index values are not bounds-checked, so a miner must return indices within the
query and reference rows.
