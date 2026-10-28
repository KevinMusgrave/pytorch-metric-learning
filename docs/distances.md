# Distances

Distance classes compute pairwise distances/similarities between input embeddings.

Consider the TripletMarginLoss in its default form:

```python
from pytorch_metric_learning.losses import TripletMarginLoss
loss_func = TripletMarginLoss(margin=0.2)
```

This loss function attempts to minimize [d<sub>ap</sub> - d<sub>an</sub> + margin]<sub>+</sub>.

Typically, d<sub>ap</sub> and d<sub>an</sub> represent Euclidean or L2 distances. But what if we want to use a squared L2 distance, or an unnormalized L1 distance, or a completely different distance measure like signal-to-noise ratio? With the distances module, you can try out these ideas easily:

```python
### TripletMarginLoss with squared L2 distance ###
from pytorch_metric_learning.distances import LpDistance
loss_func = TripletMarginLoss(margin=0.2, distance=LpDistance(power=2))

### TripletMarginLoss with unnormalized L1 distance ###
loss_func = TripletMarginLoss(margin=0.2, distance=LpDistance(normalize_embeddings=False, p=1))

### TripletMarginLoss with signal-to-noise ratio###
from pytorch_metric_learning.distances import SNRDistance
loss_func = TripletMarginLoss(margin=0.2, distance=SNRDistance())
```

You can also use similarity measures rather than distances, and the loss function will make the necessary adjustments:

```python
### TripletMarginLoss with cosine similarity##
from pytorch_metric_learning.distances import CosineSimilarity
loss_func = TripletMarginLoss(margin=0.2, distance=CosineSimilarity())
```

With a similarity measure, the TripletMarginLoss internally swaps the anchor-positive and anchor-negative terms: [s<sub>an</sub> - s<sub>ap</sub> + margin]<sub>+</sub>. In other words, it will try to make the anchor-negative similarities smaller than the anchor-positive similarities.

All **losses, miners, and regularizers** accept a `distance` argument. So you can try out the `MultiSimilarityMiner` using `SNRDistance`, or the `NTXentLoss` using `LpDistance(p=1)` and so on. Note that some losses/miners/regularizers have restrictions on the type of distances they can accept. For example, some classification losses only allow `CosineSimilarity` or `DotProductSimilarity` as their distance measure between embeddings and weights. To view restrictions for specific loss functions, see the [losses page](losses.md)

## BaseDistance

All distances extend this class and therefore inherit its `__init__` parameters.

```python
distances.BaseDistance(normalize_embeddings=True,
						p=2,
						power=1)
```

**Parameters**:

- **normalize_embeddings**: If True, embeddings will be normalized to have an Lp norm of 1, before the distance/similarity matrix is computed.
- **p**: The norm used for embedding normalization and, in `LpDistance`, distance computation.
- **power**: If not 1, each element of the distance/similarity matrix will be raised to this power.
  **Class attribute**: `is_inverted = False` means smaller values are closer.
  Similarity subclasses set `is_inverted = True` because larger values are more
  similar. This is part of the distance definition, not a constructor argument.

**Required Implementations**:

```python
# Must return a matrix where mat[j,k] represents
# the distance/similarity between query_emb[j] and ref_emb[k]
def compute_mat(self, query_emb, ref_emb):
    raise NotImplementedError

# Must return a tensor where output[j] represents
# the distance/similarity between query_emb[j] and ref_emb[j]
def pairwise_distance(self, query_emb, ref_emb):
    raise NotImplementedError
```

## BatchedDistance

Computes distance matrices iteratively. Each call supplies an `iter_fn`
callback that consumes the blocks; the module retains no callback between calls.

```python
distances.BatchedDistance(distance, batch_size=32)
```

**Parameters**:

- **distance**: The wrapped distance function.
- **batch_size**: Each distance matrix will be size `(batch_size, len(ref_emb))`.

Call `distance(query_emb, ref_emb=None, *, iter_fn=callback)`. The required
callback receives `(mat, s, e)`, where `mat` is the current distance matrix and
`s:e` selects its query rows. The final block may be shorter than `batch_size`.
The call returns `None`; the callback owns any collected outputs. The wrapper
exposes `is_inverted` for search ordering. Other wrapped configuration is
available explicitly through `distance.distance`.

**Example usage**:

```python
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity

def fn(mat, s, e):
	print(f"At query indices {s}:{e}")

distance = BatchedDistance(CosineSimilarity())
distance(embeddings, iter_fn=fn)
distance(embeddings, ref_emb, iter_fn=fn)
```

## CosineSimilarity

```python
distances.CosineSimilarity(**kwargs)
```

The returned `mat[i,j]` is the cosine similarity between `query_emb[i]` and `ref_emb[j]`. This class is equivalent to [`DotProductSimilarity(normalize_embeddings=True)`](#dotproductsimilarity).

## DotProductSimilarity

```python
distances.DotProductSimilarity(normalize_embeddings=True, p=2, power=1, **kwargs)
```

The returned `mat[i,j]` is equal to `torch.sum(query_emb[i] * ref_emb[j])`

## LpDistance

```python
distances.LpDistance(normalize_embeddings=True, p=2, power=1, **kwargs)
```

The returned `mat[i,j]` is the Lp distance between `query_emb[i]` and `ref_emb[j]`. With default parameters, this is the Euclidean distance.

## SNRDistance

[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}

```python
distances.SNRDistance(normalize_embeddings=True, p=2, power=1, **kwargs)
```

The returned `mat[i,j]` is equal to:

```python
torch.var(query_emb[i] - ref_emb[j]) / torch.var(query_emb[i])
```
