# Retrieval backends

`utils.inference` provides nearest-neighbor search, threshold matching, and
k-means clustering for [`AccuracyCalculator`](accuracy_calculation.md) and
application code. These operate on embedding tensors the caller has already
computed; see the [evaluation recipe](recipes/evaluation.md#extract-embeddings)
for an ordinary PyTorch extraction loop.

| Backend       | Dependency    | Default for                                    |
| ------------- | ------------- | ---------------------------------------------- |
| `CustomKNN`   | PyTorch       | `AccuracyCalculator` k-NN metrics              |
| `MatchFinder` | PyTorch       | None; construct it explicitly                  |
| `FaissKNN`    | `faiss` extra | None; pass it as `knn_func`                    |
| `FaissKMeans` | `faiss` extra | None; pass it as `kmeans_func` for NMI and AMI |

Importing `utils.inference` or `utils.accuracy_calculator` does not import Faiss
or scikit-learn. Constructing a Faiss backend without the extra raises
`ModuleNotFoundError` naming `pytorch-metric-learning[faiss]`.

## k-NN function contract

A `knn_func` is called as `knn_func(query, k, reference, ref_includes_query)`
and returns `(distances, indices)`, each shaped `(len(query), k)`, on the
query's device. Rows are ordered from nearest to farthest. When
`ref_includes_query` is true, the backend searches for `k + 1` neighbors and
removes each query's own index `i`; if row `i` does not appear in a row's
results, the last neighbor is removed instead. This assumes the query is the
first `len(query)` rows of `reference`.

## CustomKNN

```python
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import CustomKNN

CustomKNN(distance, batch_size=None)
```

- **distance**: A [distance](distances.md). Similarities such as
  `CosineSimilarity` select the largest values; distances select the smallest.
  Normalization follows the distance's `normalize_embeddings` setting.
- **batch_size**: If set, compute `(batch_size, len(reference))` distance blocks
  and keep each block's top `k`. The result equals the unbatched result except
  for the order of exactly tied values; peak memory falls from the full
  `(len(query), len(reference))` matrix to one block plus the outputs.

Distances have the query's dtype; indices are `torch.long`. Tied values are
ordered by `torch.topk`, which does not guarantee index order.

`AccuracyCalculator` defaults to
`CustomKNN(LpDistance(normalize_embeddings=False, power=2), batch_size=256)`:
exact squared Euclidean search on the input device without normalization.
Normalize embeddings before evaluation when cosine geometry is intended.

```python
import torch

query = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
reference = torch.tensor([[0.0, 1.0], [3.0, 0.5], [9.0, 9.0]])
knn = CustomKNN(LpDistance(normalize_embeddings=False, power=2))
distances, indices = knn(query, 1, reference)
assert indices.tolist() == [[0], [1]]
assert distances.tolist() == [[1.0], [0.25]]
```

## MatchFinder

```python
from pytorch_metric_learning.utils.inference import MatchFinder

MatchFinder(distance, threshold=None)
```

A pair matches when its distance is at most `threshold`, or when its similarity
is at least `threshold` for a distance with `is_inverted=True`. A threshold
passed to a method overrides the constructor value; omitting both raises
`ValueError`. Computations run under `torch.no_grad()` on the inputs' device.

- `get_matching_pairs(query_emb, ref_emb=None, threshold=None, return_tuples=False)`
  returns a boolean `(len(query_emb), len(ref_emb))` tensor. With
  `return_tuples=True`, it returns the `(num_matches, 2)` index tensor from
  `nonzero()`. `ref_emb=None` compares `query_emb` with itself, so the diagonal
  matches whenever a row matches itself.
- `is_match(x, y, threshold=None)` compares row `i` of `x` with row `i` of `y`
  and returns a boolean tensor shaped `(len(x),)`.

```python
from pytorch_metric_learning.distances import CosineSimilarity

finder = MatchFinder(CosineSimilarity(), threshold=0.9)
matches = finder.get_matching_pairs(query + 1e-3, reference)
```

## FaissKNN

```python
from pytorch_metric_learning.utils.inference import FaissKNN

FaissKNN(reset_before=True, reset_after=True, index_init_fn=None, gpus=None)
```

- **reset_before**: Build a new index from `index_init_fn(dimension)` on every
  call, adding `reference` before searching.
- **reset_after**: Discard the index after every call.
- **index_init_fn**: Callable returning a Faiss index for a dimension. Defaults to
  `faiss.IndexFlatL2`, whose distances are squared Euclidean.
- **gpus**: Faiss GPU indices for the search. Without it, CUDA queries use all
  GPUs visible to a GPU-enabled Faiss build.

Faiss searches CPU float32 copies of the inputs; results return to the query's
device. A GPU-enabled Faiss build searches on GPU only when `k` is at most 2,048,
the limit in Faiss's `DeviceDefs.cuh`, and otherwise logs a warning and uses
the CPU index.

To keep a persistent index, disable both resets:

```python
knn = FaissKNN(reset_before=False, reset_after=False)
knn.train(reference)        # new index containing reference
knn.add(more_reference)     # append rows
distances, indices = knn(query, 10)
knn.save("reference.index")
restored = FaissKNN(reset_before=False, reset_after=False)
restored.load("reference.index")
```

## FaissKMeans

```python
from pytorch_metric_learning.utils.inference import FaissKMeans

FaissKMeans(**kwargs)
```

Keyword arguments go to `faiss.Kmeans`. Calling `kmeans(x, num_clusters)` trains
on a float32 CPU copy of `x` and returns one `torch.long` cluster index per row
on `x`'s device.

## macOS OpenMP runtimes

The macOS `faiss-cpu` and PyTorch wheels each bundle an OpenMP runtime. On the
development Mac, loading both in one process aborts with `OMP: Error #15`;
setting `KMP_DUPLICATE_LIB_OK=TRUE` alone avoids that abort but still crashes
during multithreaded Faiss calls. Both `KMP_DUPLICATE_LIB_OK=TRUE` and
`OMP_NUM_THREADS=1`, set before Python starts, let searches and k-means
complete, as in the repository's CPU test profile. Intel documents the
duplicate-runtime setting as unsupported. Prefer `CustomKNN` on macOS, or
install Faiss and PyTorch from one distribution that shares a single OpenMP
runtime. Earlier releases imported scikit-learn eagerly, which sets
`KMP_DUPLICATE_LIB_OK` as a side effect; this release does not.

## Removed model wrapper

`InferenceModel` and the PCA helper were removed. Extract embeddings in
application code, then call a backend directly:

```python
model.eval()
with torch.inference_mode():
    reference = torch.cat([model(batch) for batch, _ in reference_loader])
    query = model(query_batch)
distances, indices = knn(query, 10, reference)
```
