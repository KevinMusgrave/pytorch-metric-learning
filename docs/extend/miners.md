# How to write custom mining functions

1. Extend ```BaseTupleMiner```
2. Implement the ```mine``` method
3. Inside ```mine```, return a tuple of tensors

## An example pair miner
```python
from pytorch_metric_learning.miners import BaseTupleMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

class ExamplePairMiner(BaseTupleMiner):
    def __init__(self, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_pairs = mat[a1, p]
        neg_pairs = mat[a2, n]
        pos_mask = (
            pos_pairs < self.margin
            if self.distance.is_inverted
            else pos_pairs > self.margin
        )
        neg_mask = (
            neg_pairs > self.margin
            if self.distance.is_inverted
            else neg_pairs < self.margin
        )
        return a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask]

```

The ```ExamplePairMiner``` does the following:

- Computes the distance matrix between ```embeddings``` and ```ref_emb```.
- Finds the indices of all positive and negative pairs
- Returns the indices of pairs that violate the margin

Example usage:

```python
miner = ExamplePairMiner()
embeddings = torch.randn(128, 512)
labels = torch.randint(0, 10, size=(128,))
pairs = miner(embeddings, labels)
```

## An example triplet miner
```python
from pytorch_metric_learning.miners import BaseTupleMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

class ExampleTripletMiner(BaseTupleMiner):
    def __init__(self, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)
        pos_pairs = mat[a, p]
        neg_pairs = mat[a, n]
        triplet_margin = pos_pairs - neg_pairs if self.distance.is_inverted else neg_pairs - pos_pairs
        triplet_mask = triplet_margin <= self.margin
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]
```

This miner works similarly to ```ExamplePairMiner```, but finds triplets instead of pairs.


## What is ```ref_emb```?
The forward function of ```BaseTupleMiner``` has optional ```ref_emb``` and ```ref_labels``` arguments. The miner should return anchors from ```embeddings``` and positives and negatives from ```ref_emb```. For example:

```python
miner = ExamplePairMiner()
embeddings = torch.randn(128, 512)
labels = torch.randint(0, 10, size=(128,))
ref_emb = torch.randn(32, 512)
ref_labels = torch.randint(0, 10, size=(32,))
a1, p, a2, n = miner(embeddings, labels, ref_emb, ref_labels)
# a1 and a2 contain indices of "embeddings"
# p and n contain indices of "ref_emb"
```

Typically though, ```ref_emb``` and ```ref_labels``` are left to their default value of ```None```, in which case they are set to ```embeddings``` and ```labels``` before being passed to the ```mine``` function.