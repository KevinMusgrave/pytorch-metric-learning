# How to write custom mining functions

1. Extend `BaseMiner`
2. Implement the `mine` method
3. Inside `mine`, return a tuple of index tensors

## An example pair miner

```python
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import indexing


class ExamplePairMiner(BaseMiner):
    def __init__(self, margin=0.1, *, distance=None):
        super().__init__(distance=distance)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = indexing.get_all_pairs_indices(labels, ref_labels)
        pos_mask = self.distance.margin(mat[a1, p], self.margin) > 0
        neg_mask = self.distance.margin(self.margin, mat[a2, n]) > 0
        return a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask]
```

The `ExamplePairMiner` does the following:

- Computes the distance matrix between `embeddings` and `ref_emb`.
- Finds the indices of all positive and negative pairs.
- Returns the indices of pairs that violate the margin. `distance.margin(x, y)`
  is `x - y` for distances and `y - x` for similarities, so the same code works
  for both orientations.

Example usage:

```python
miner = ExamplePairMiner()
embeddings = torch.randn(128, 512)
labels = torch.randint(0, 10, size=(128,))
pairs = miner(embeddings, labels)
```

## An example triplet miner

```python
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import indexing


class ExampleTripletMiner(BaseMiner):
    def __init__(self, margin=0.1, *, distance=None):
        super().__init__(distance=distance)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a, p, n = indexing.get_all_triplets_indices(labels, ref_labels)
        triplet_margin = self.distance.margin(mat[a, n], mat[a, p])
        triplet_mask = triplet_margin <= self.margin
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]
```

This miner works like `ExamplePairMiner`, but finds triplets instead of pairs.

## Validation

`BaseMiner.forward` runs `mine` under `torch.no_grad()` and then validates the
output: three equal-length triplet tensors or four pair tensors whose anchor
and partner lengths match, all one-dimensional. Invalid types raise
`TypeError` and invalid shapes raise `ValueError`.

## What is `ref_emb`?

The forward function of `BaseMiner` has optional `ref_emb` and `ref_labels`
arguments. The miner returns anchors from `embeddings` and positives and
negatives from `ref_emb`:

```python
miner = ExamplePairMiner()
embeddings = torch.randn(128, 512)
labels = torch.randint(0, 10, size=(128,))
ref_emb = torch.randn(32, 512)
ref_labels = torch.randint(0, 10, size=(32,))
a1, p, a2, n = miner(embeddings, labels, ref_emb, ref_labels)
# a1 and a2 index "embeddings"; p and n index "ref_emb"
```

When `ref_emb` and `ref_labels` are `None`, they are set to `embeddings` and
`labels` before `mine` is called, and self-comparisons are excluded from the
positive pairs.
