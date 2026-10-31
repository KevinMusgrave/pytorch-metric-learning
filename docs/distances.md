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

```python {.signature}
distances.BaseDistance(*, normalize_embeddings=True, p=2, power=1)
```

**Parameters**:

- **normalize_embeddings**: If `True`, embeddings are normalized to unit Lp norm
  before values are computed. See [normalization](#normalization).
- **p**: The norm used for embedding normalization and, in `LpDistance`, for
  the distance itself. Similarities use it only for normalization.
- **power**: If not 1, each computed value is raised to this power.

### Normalization

`distance.normalize(x, dim=1)` returns `x / max(norm_p(x), eps)` row by row,
with the same dtype as `x`. `eps` is `1e-12` for float32, float64, and
bfloat16, and `2**-14` (the smallest normal float16 value) for float16. Float16
and bfloat16 inputs are normalized in float32 and cast back, so their norms do
not underflow. A zero row normalizes to zero with finite gradients in every
dtype. A row whose norm is below `eps` is divided by `eps` instead of being
scaled to unit norm, so its gradient is the incoming gradient times `1 / eps`:
16384 in float16, where `1e-12` would overflow.

### Calling a distance

- `distance(query_emb, ref_emb=None)` returns a matrix of shape
  `(len(query_emb), len(ref_emb))` whose entry `[i, j]` compares `query_emb[i]`
  with `ref_emb[j]`. `ref_emb=None` compares `query_emb` with itself.
- `distance.pairwise(query_emb, ref_emb)` returns a vector of shape
  `(len(query_emb),)` whose entry `i` compares `query_emb[i]` with `ref_emb[i]`.
  The inputs must have the same shape; otherwise it raises `ValueError`. It
  computes only the corresponding rows, without a full matrix.

Both apply `normalize_embeddings` and `power` exactly as configured, so
`distance.pairwise(x, y)` equals the diagonal of `distance(x, y)` up to
floating-point rounding, and identical rows give exactly zero in both for
`LpDistance`. Inputs must be
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

- `compute_mat(query_emb, ref_emb)` returns the matrix whose entry `[j, k]` is
  the distance or similarity between `query_emb[j]` and `ref_emb[k]`.
- `pairwise_distance(query_emb, ref_emb)` returns the vector whose entry `j` is
  the distance or similarity between `query_emb[j]` and `ref_emb[j]`.

```python
import torch
from pytorch_metric_learning.distances import BaseDistance


class ChebyshevDistance(BaseDistance):
    def compute_mat(self, query_emb, ref_emb):
        return torch.cdist(query_emb, ref_emb, p=float("inf"))

    def pairwise_distance(self, query_emb, ref_emb):
        return (query_emb - ref_emb).abs().amax(dim=1)


distance = ChebyshevDistance(normalize_embeddings=False)
x, y = torch.randn(4, 8), torch.randn(4, 8)
assert distance(x, y).shape == (4, 4)
assert torch.allclose(distance.pairwise(x, y), distance(x, y).diagonal())
```

`compute_mat` must return shape `(len(query_emb), len(ref_emb))`; another shape
raises `ValueError`. Call `pairwise`, not `pairwise_distance`, from losses and
application code, so normalization and `power` apply. Custom similarities set
`is_inverted = True` on the class.

## BatchedDistance

Computes a distance matrix in blocks of query rows, so peak memory is one
`(batch_size, len(ref_emb))` block instead of the full matrix.

```python {.signature}
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

```python {.signature}
distances.CosineSimilarity()
```

`mat[i, j]` is the cosine similarity between `query_emb[i]` and `ref_emb[j]`, a
value in `[-1, 1]`. Embeddings are always L2-normalized and no power is
applied, so this equals
[`DotProductSimilarity(normalize_embeddings=True)`](#dotproductsimilarity).
The class takes no options: losses that require `CosineSimilarity`, such as
CircleLoss and LargeMarginSoftmaxLoss, rely on signed cosines (some take their
`acos`). For a powered or Lp-normalized dot product, use `DotProductSimilarity`
with `p` or `power`.

## DotProductSimilarity

```python {.signature}
distances.DotProductSimilarity(*, normalize_embeddings=False, p=2, power=1)
```

`mat[i, j]` equals `torch.sum(query_emb[i] * ref_emb[j])`. Unlike every other
distance, embeddings are used as given by default; pass
`normalize_embeddings=True` to normalize them first.

## LpDistance

```python {.signature}
distances.LpDistance(*, normalize_embeddings=True, p=2, power=1)
```

`mat[i, j]` is the Lp distance between `query_emb[i]` and `ref_emb[j]`. With
default parameters, this is the Euclidean distance between normalized
embeddings. For float16 and bfloat16 inputs, the matrix is computed in float32
with `torch.cdist` using its direct difference kernel and cast back, which keeps
peak memory at the size of the output matrix on every device and preserves
distances between nearby rows that the matrix-product formula would cancel to
zero.

`pairwise` computes `torch.linalg.vector_norm(x - y, ord=p, dim=-1)`, in float32
for float16 and bfloat16 inputs. It adds no epsilon, so identical rows give
exactly zero, as in the matrix, and the gradient at identical rows is zero, as
`torch.cdist` defines it.

## SNRDistance

[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python {.signature}
distances.SNRDistance(*, normalize_embeddings=True, p=2, power=1)
```

`mat[i, j]` is the variance of `query_emb[i] - ref_emb[j]` divided by the
variance of `query_emb[i]`, computed after normalization when
`normalize_embeddings=True`. The ratio is undefined for a query row whose
elements are all equal, such as a zero row, and its entries are NaN:

```python
import torch

query_emb, ref_emb = torch.randn(3, 8), torch.randn(5, 8)
mat = SNRDistance(normalize_embeddings=False)(query_emb, ref_emb)
i, j = 1, 4
expected = torch.var(query_emb[i] - ref_emb[j]) / torch.var(query_emb[i])
assert torch.allclose(mat[i, j], expected)
```
