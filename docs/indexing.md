# Index utilities

`pytorch_metric_learning.utils.indexing` enumerates, converts, and weights the
[index tuples](types.md#index-tuples) that miners return and losses accept. Use
these functions in [custom losses](extend/losses.md#using-indices_tuple) and
[custom miners](extend/miners.md) to handle labels, plain tuples, pair tuples,
and triplet tuples in one call.

```python
import torch
from pytorch_metric_learning.types import PairIndices, TripletIndices
from pytorch_metric_learning.utils import indexing

labels = torch.tensor([0, 0, 1, 1, 2])
```

Every function follows these conventions:

- `labels` is a one-dimensional tensor of shape `(N,)`. `ref_labels` is `None`
  or a one-dimensional tensor of shape `(M,)` on the same device. Labels are
  compared with `==` and are not otherwise validated.
- `ref_labels=None` means self-comparison: the reference set is `labels`, and
  each row is excluded from its own positives. A supplied `ref_labels` is a
  separate set, and no pair is excluded; see
  [reference sets](concepts.md#reference-sets).
- Anchor indices refer to `labels`, and positive and negative indices refer to
  `ref_labels`, or to `labels` when `ref_labels` is `None`.
- Returned index tensors are `torch.long` on the labels' device, or on the
  device of the supplied indices.
- An `indices_tuple` argument accepts `None`, a
  [`PairIndices`](types.md#pairindices), a
  [`TripletIndices`](types.md#tripletindices), or a plain tuple or list of four
  or three tensors. Plain tuples are validated as described in
  [index tuples](losses.md#index-tuples): a non-tensor entry or a dtype other
  than `int64` or `int32` raises `TypeError`, and another length, a
  multi-dimensional tensor, or mismatched anchor and partner lengths raise
  `ValueError`. Index values are not bounds-checked.

## get_all_pairs_indices

```python {.signature}
indexing.get_all_pairs_indices(labels, ref_labels=None)
```

Returns a `PairIndices` with every positive pair (equal labels) and every
negative pair (different labels), in row-major order of the
`(N, M)` comparison matrix.

```python
pairs = indexing.get_all_pairs_indices(labels)
assert pairs.pos_anchors.tolist() == [0, 1, 2, 3]
assert pairs.positives.tolist() == [1, 0, 3, 2]
assert len(pairs.neg_anchors) == 16
```

## get_all_triplets_indices

```python {.signature}
indexing.get_all_triplets_indices(labels, ref_labels=None)
```

Returns a `TripletIndices` with every `(anchor, positive, negative)` whose
positive has the anchor's label and whose negative does not. It equals
`pairs_to_triplets(get_all_pairs_indices(labels, ref_labels))`, so memory scales
with the number of triplets.

```python
triplets = indexing.get_all_triplets_indices(labels)
assert len(triplets.anchors) == 12
assert torch.all(labels[triplets.anchors] == labels[triplets.positives])
assert torch.all(labels[triplets.anchors] != labels[triplets.negatives])
```

## get_random_triplet_indices

```python {.signature}
indexing.get_random_triplet_indices(
    labels, ref_labels=None, t_per_anchor=None, weights=None
)
```

Samples triplets at random with PyTorch's global random number generator.

- **t_per_anchor**: The number of triplets per anchor. `None` draws one
  triplet per positive of each anchor. A positive integer draws that many,
  sampling positives and negatives with replacement. Zero draws none.
- **weights**: Optional nonnegative `(N, M)` tensor of negative-sampling
  weights for each anchor and reference row. `None` samples negatives
  uniformly.

Anchors without a positive, or without a negative of positive weight, get no
triplets. The draw synchronizes the device once to size its output.

```python
torch.manual_seed(0)
triplets = indexing.get_random_triplet_indices(labels, t_per_anchor=2)
assert len(triplets.anchors) == 8
```

## pairs_to_triplets

```python {.signature}
indexing.pairs_to_triplets(pairs)
```

Combines each positive pair with every negative pair that shares its anchor.
`pairs` is a `PairIndices` or a plain tuple or list of four index tensors,
validated like `indices_tuple`; three tensors raise `ValueError`. Triplets are
ordered by positive pair, then by negative pair. A positive pair whose anchor
has no negative pair produces no triplet, and empty inputs give empty outputs.

```python
pairs = PairIndices(
    torch.tensor([0, 0]),
    torch.tensor([1, 1]),
    torch.tensor([0, 2]),
    torch.tensor([3, 4]),
)
triplets = indexing.pairs_to_triplets(pairs)
assert triplets.negatives.tolist() == [3, 3]
```

## convert_to_pairs

```python {.signature}
indexing.convert_to_pairs(indices_tuple, labels, ref_labels=None)
```

Returns a `PairIndices`:

- `None`: every pair, from `get_all_pairs_indices(labels, ref_labels)`. `labels`
  is then required; `labels=None` raises `ValueError`.
- Pairs: returned as a `PairIndices`.
- A triplet `(a, p, n)`: the positive pair `(a, p)` and the negative pair
  `(a, n)`, one of each per triplet.

`labels` and `ref_labels` are not used when `indices_tuple` is supplied.

```python
plain = (torch.tensor([0, 2]), torch.tensor([1, 3]), torch.tensor([4, 4]))
assert indexing.convert_to_pairs(plain, labels).neg_anchors.tolist() == [0, 2]
```

## convert_to_triplets

```python {.signature}
indexing.convert_to_triplets(indices_tuple, labels, ref_labels=None, t_per_anchor="all")
```

Returns a `TripletIndices`:

- `None`: with `t_per_anchor="all"`, every triplet from
  `get_all_triplets_indices`; with an integer,
  `get_random_triplet_indices(labels, ref_labels, t_per_anchor=t_per_anchor)`.
  `labels=None` raises `ValueError`.
- Triplets: returned as a `TripletIndices`.
- Pairs: `pairs_to_triplets(indices_tuple)`.

`t_per_anchor` is a [`TripletsPerAnchor`](types.md#tripletsperanchor) and only
applies when `indices_tuple` is `None`.

```python
assert isinstance(indexing.convert_to_triplets(None, labels), TripletIndices)
assert len(indexing.convert_to_triplets(pairs, labels).anchors) == 2
```

## convert_to_weights

```python {.signature}
indexing.convert_to_weights(indices_tuple, labels, dtype, using_ref=False)
```

Returns the [miner weights](losses.md#miner-weights): a `(N,)` tensor of dtype
`dtype` on the labels' device, where `N = len(labels)`. Each element's weight is
its number of occurrences in the tuple divided by the largest count. `None`
gives every element weight 1, and a tuple without indices gives every element
weight 0. With `using_ref=True`, only anchors are counted, because positives
and negatives index a separate reference set. `labels` provides only the length
and device.

```python
triplets = (torch.tensor([0, 2]), torch.tensor([1, 3]), torch.tensor([4, 4]))
weights = indexing.convert_to_weights(triplets, labels, dtype=torch.float32)
assert weights.tolist() == [0.5, 0.5, 0.5, 0.5, 1.0]
anchor_weights = indexing.convert_to_weights(
    triplets, labels, dtype=torch.float32, using_ref=True
)
assert anchor_weights.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
```
