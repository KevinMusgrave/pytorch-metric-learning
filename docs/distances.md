# Distances

Distance modules compute distances or similarities between embeddings.
Consider TripletMarginLoss with a margin of 0.2:

```python
from pytorch_metric_learning.losses import TripletMarginLoss

loss_func = TripletMarginLoss(margin=0.2)
```

This loss minimizes $\left[d_{ap} - d_{an} + m\right]_+$, where $d_{ap}$ and
$d_{an}$ are Euclidean distances between L2-normalized embeddings by default. To
use a squared L2 distance, an unnormalized L1 distance, or a different measure
such as signal-to-noise ratio, pass a distance module:

```python
from pytorch_metric_learning.distances import LpDistance, SNRDistance

# squared L2 distance
loss_func = TripletMarginLoss(margin=0.2, distance=LpDistance(power=2))
# unnormalized L1 distance
loss_func = TripletMarginLoss(
    margin=0.2, distance=LpDistance(normalize_embeddings=False, p=1)
)
# signal-to-noise ratio
loss_func = TripletMarginLoss(margin=0.2, distance=SNRDistance())
```

Similarities work too, and the loss orients its comparisons accordingly:

```python
from pytorch_metric_learning.distances import CosineSimilarity

loss_func = TripletMarginLoss(margin=0.2, distance=CosineSimilarity())
```

With a similarity, TripletMarginLoss minimizes
$\left[s_{an} - s_{ap} + m\right]_+$, pushing anchor-negative similarities below
anchor-positive similarities.

All losses, miners, and regularizers accept a `distance` option. Some restrict
the compatible distances; each component page lists its default distance and
any restriction.

## Distance orientation

`is_inverted` is a class attribute: `False` means smaller values are closer
(distances), and `True` means larger values are closer (similarities). Losses and
miners read it to orient margins and comparisons. `distance.margin(x, y)` returns
`x - y` for distances and `y - x` for similarities, so a positive value means `x`
is farther than `y` under either orientation.

## BaseDistance

All distances extend this class. Its options are keyword-only.

```python
distances.BaseDistance(*, normalize_embeddings=True, p=2, power=1)
```

**Parameters**:

- **normalize_embeddings**: If `True`, embeddings are normalized to unit Lp norm
  before values are computed.
- **p**: The norm used for embedding normalization and, in `LpDistance`, for
  the distance itself. Similarities use it only for normalization.
- **power**: If not 1, each computed value is raised to this power.

### Calling a distance

- `distance(query_emb, ref_emb=None)` returns a matrix of shape
  `(len(query_emb), len(ref_emb))` whose entry `[i, j]` compares `query_emb[i]`
  with `ref_emb[j]`. `ref_emb=None` compares `query_emb` with itself.
- `distance.pairwise(query_emb, ref_emb)` returns a vector of shape
  `(len(query_emb),)` whose entry `i` compares `query_emb[i]` with `ref_emb[i]`.
  The inputs must have the same shape; otherwise it raises `ValueError`. It
  computes only the corresponding rows, without a full matrix.

Both apply `normalize_embeddings` and `power` exactly as configured, so
`distance.pairwise(x, y)` equals the diagonal of `distance(x, y)`. Inputs must be
two-dimensional `(batch_size, embedding_size)` tensors; otherwise both raise
`ValueError`.

```python
import torch
from pytorch_metric_learning.distances import LpDistance

distance = LpDistance(power=2)
x, y = torch.randn(5, 8), torch.randn(5, 8)
assert distance(x, y).shape == (5, 5)
assert torch.allclose(distance.pairwise(x, y), distance(x, y).diagonal(), atol=1e-5)
```

### Custom distances

A subclass implements two hooks on already-normalized embeddings. `forward` and
`pairwise` call them and then apply `power`:

```python
# Return a matrix where mat[j, k] is the distance or similarity
# between query_emb[j] and ref_emb[k].
def compute_mat(self, query_emb, ref_emb):
    raise NotImplementedError


# Return a vector where output[j] is the distance or similarity
# between query_emb[j] and ref_emb[j].
def pairwise_distance(self, query_emb, ref_emb):
    raise NotImplementedError
```

`compute_mat` must return shape `(len(query_emb), len(ref_emb))`; another shape
raises `ValueError`. Call `pairwise`, not `pairwise_distance`, from losses and
application code, so normalization and `power` apply. Custom similarities set
`is_inverted = True` on the class.

## BatchedDistance

Computes a distance matrix in blocks of query rows, so peak memory is one
`(batch_size, len(ref_emb))` block instead of the full matrix.

```python
distances.BatchedDistance(distance, batch_size=32)
```

**Parameters**:

- **distance**: The wrapped distance module.
- **batch_size**: The number of query rows per block. It must be positive;
  otherwise construction raises `ValueError`.

Calling `batched(query_emb, ref_emb=None)` returns an iterator of
`(mat, start, end)` tuples, where `mat` is the block for query rows
`start:end`. The final block may be shorter than `batch_size`. The wrapper
exposes `is_inverted` for search ordering; other configuration is available
through `batched.distance`. It is not a `BaseDistance`, so pass the wrapped
distance, not the wrapper, to losses and miners.

```python
import torch
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity

embeddings, ref_emb = torch.randn(600, 16), torch.randn(100, 16)
batched = BatchedDistance(CosineSimilarity(), batch_size=256)
for mat, start, end in batched(embeddings):
    print(f"rows {start}:{end} -> {tuple(mat.shape)}")
blocks = [mat for mat, _, _ in batched(embeddings, ref_emb)]
assert torch.cat(blocks).shape == (600, 100)
```

## CosineSimilarity

```python
distances.CosineSimilarity(*, p=2, power=1)
```

`mat[i, j]` is the cosine similarity between `query_emb[i]` and `ref_emb[j]`.
Embeddings are always normalized, so this equals
[`DotProductSimilarity(normalize_embeddings=True)`](#dotproductsimilarity).

## DotProductSimilarity

```python
distances.DotProductSimilarity(*, normalize_embeddings=True, p=2, power=1)
```

`mat[i, j]` equals `torch.sum(query_emb[i] * ref_emb[j])` after optional
normalization.

## LpDistance

```python
distances.LpDistance(*, normalize_embeddings=True, p=2, power=1)
```

`mat[i, j]` is the Lp distance between `query_emb[i]` and `ref_emb[j]`. With
default parameters, this is the Euclidean distance between normalized
embeddings. For float16 and bfloat16 inputs, the matrix is computed in float32
with `torch.cdist` using its direct difference kernel and cast back, which keeps
peak memory at the size of the output matrix on every device and preserves
distances between nearby rows that the matrix-product formula would cancel to
zero.

## SNRDistance

[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python
distances.SNRDistance(*, normalize_embeddings=True, p=2, power=1)
```

`mat[i, j]` equals:

```python
torch.var(query_emb[i] - ref_emb[j]) / torch.var(query_emb[i])
```
