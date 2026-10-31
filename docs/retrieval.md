# Retrieval backends

`pytorch_metric_learning.evaluation` provides nearest-neighbor search,
threshold matching and selection, and k-means clustering for
[`AccuracyCalculator`](accuracy_calculation.md) and application code. These
operate on embedding tensors the caller has already computed; the
[evaluation recipe](recipes/evaluation.md#extract-embeddings) shows an ordinary
PyTorch extraction loop.

```python
from pytorch_metric_learning.evaluation import (
    CustomKNN,
    FaissKMeans,
    FaissKNN,
    MatchFinder,
)
```

| Backend       | Dependency    | Default for                                    |
| ------------- | ------------- | ---------------------------------------------- |
| `CustomKNN`   | PyTorch       | `AccuracyCalculator` k-NN metrics              |
| `MatchFinder` | PyTorch       | None; construct it explicitly                  |
| `FaissKNN`    | `faiss` extra | None; pass it as `knn_func`                    |
| `FaissKMeans` | `faiss` extra | None; pass it as `kmeans_func` for NMI and AMI |

Importing `pytorch_metric_learning.evaluation` does not import Faiss or
scikit-learn. Constructing a Faiss backend without the extra raises
`ModuleNotFoundError` naming `pytorch-metric-learning[faiss]`.

## k-NN function contract

A `knn_func` is called as `knn_func(query, k, reference, ref_includes_query)`
and returns `(distances, indices)`, each shaped `(len(query), k)`, on the
query's device. Rows are ordered from nearest to farthest. When
`ref_includes_query` is true, the backend searches for `k + 1` neighbors and
removes each query's own index `i`; if row `i` does not appear in a row's
results, the last neighbor is removed instead. This assumes the query is the
first `len(query)` rows of `reference`, which
[`AccuracyCalculator`](accuracy_calculation.md#getting-accuracy) validates.
Every index must address a row of the `reference` passed to that call.
`AccuracyCalculator` raises `ValueError` for indices of another shape or
outside the reference set. It calls `knn_func` under `torch.no_grad()` with the
[number of neighbors its selected metrics need](accuracy_calculation.md#neighbors-retrieved).

## CustomKNN

```python {.signature}
evaluation.CustomKNN(distance, batch_size=None)
```

- **distance**: A [distance](distances.md). Similarities such as
  `CosineSimilarity` select the largest values; distances select the smallest.
  Normalization follows the distance's `normalize_embeddings` setting.
- **batch_size**: If set, compute `(batch_size, len(reference))` distance blocks
  and keep each block's top `k`. The result equals the unbatched result except
  for the order of exactly tied values; peak memory falls from the full
  `(len(query), len(reference))` matrix to one block plus the outputs.

Distances have the dtype the distance returns, which is float32 for float16 queries
with an [unnormalized distance](distances.md#calling-a-distance); indices are `torch.long`. Tied values are
ordered by `torch.topk`, which does not guarantee index order. The search runs
under `torch.no_grad()`, so its outputs carry no autograd graph even when the
inputs require gradients. The backend retains no per-call result tensors after
results are discarded, including after a failed search. `AccuracyCalculator` releases
the k-NN outputs before returning its metrics. With `k=None`, all neighbors are
materialized during evaluation, so peak output storage is still quadratic when
query and reference counts grow together.

`AccuracyCalculator` defaults to
`CustomKNN(LpDistance(normalize_embeddings=False, power=2), batch_size=256)`:
exact squared Euclidean search on the input device without normalization.
Normalize embeddings before evaluation when cosine geometry is intended.

```python
import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.evaluation import CustomKNN

query = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
reference = torch.tensor([[0.0, 1.0], [3.0, 0.5], [9.0, 9.0]])
knn = CustomKNN(LpDistance(normalize_embeddings=False, power=2))
distances, indices = knn(query, 1, reference)
assert indices.tolist() == [[0], [1]]
assert distances.tolist() == [[1.0], [0.25]]
```

## MatchFinder

```python {.signature}
evaluation.MatchFinder(distance, threshold=None)
```

A pair matches when its distance is at most `threshold`, or when its similarity
is at least `threshold` for a distance with `is_inverted=True`. A threshold
passed to a method overrides the constructor value; omitting both raises
`ValueError`. Computations run under `torch.no_grad()` on the inputs' device.

- `get_match_matrix(query_emb, ref_emb=None, threshold=None)` returns a boolean
  `(len(query_emb), len(ref_emb))` tensor.
- `get_match_indices(query_emb, ref_emb=None, threshold=None)` returns a
  `torch.long` tensor shaped `(num_matches, 2)`; each row is a
  `(query index, reference index)` pair of the match matrix, in row-major order.
- `is_match(x, y, threshold=None)` compares row `i` of `x` with row `i` of `y`
  and returns a boolean tensor shaped `(len(x),)`.

For `get_match_matrix` and `get_match_indices`, `ref_emb=None` compares
`query_emb` with itself, so the diagonal matches whenever a row matches itself.

All three methods honor the configured distance normalization and `power`, including
cosine similarity and squared Euclidean distance. `is_match` uses
[`distance.pairwise`](distances.md#calling-a-distance), so it computes only
corresponding pairs, without materializing a full matrix.

```python
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.evaluation import MatchFinder

query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
reference = torch.tensor([[1.0, 0.1], [1.0, 1.0], [-1.0, 0.0]])
finder = MatchFinder(CosineSimilarity(), threshold=0.9)
matches = finder.get_match_matrix(query, reference)
assert matches.tolist() == [[True, False, False], [False, False, False]]
assert finder.get_match_indices(query, reference).tolist() == [[0, 0]]
assert finder.is_match(query, reference[:2]).tolist() == [True, False]
```

## Threshold selection

`select_threshold` chooses the threshold for a match/no-match decision from
validation scores and labels. Model inference, dataset splits, and pair
construction stay in application code.

```python {.signature}
evaluation.select_threshold(
    scores: ArrayLike,
    labels: ArrayLike,
    metric: Callable[[MatchCounts], Tensor],
    *,
    is_inverted: bool,
) -> ThresholdSelection
```

- **scores**: A one-dimensional floating-point tensor or NumPy array of finite
  distances or similarities, one per validation pair.
- **labels**: A boolean tensor or array of the same length, `True` where the
  pair should match. Both values must be present.
- **metric**: The quantity to maximize. It receives a `MatchCounts` and returns
  one value per candidate threshold, NaN where undefined.
- **is_inverted**: `True` when larger scores mean more alike, as for similarities;
  pass `distance.is_inverted`.

Invalid inputs, a metric that returns another shape, and a metric that is NaN
for every candidate raise `ValueError`. The function runs on CPU copies, keeps
no autograd graph, and returns Python floats.

```python {.signature}
class ThresholdSelection(NamedTuple):
    threshold: float
    value: float
```

`threshold` follows the [`MatchFinder`](#matchfinder) rule, so the validation
pairs at or beyond it get exactly the decisions counted in `value`, the metric at
that threshold. Candidates are every distinct way to separate the sorted scores:

- Between two neighboring scores, the threshold is their midpoint, which leaves
  the same room on both sides. If the scores are adjacent floating-point values,
  it is the weaker score of the matching side.
- If every pair should match, it is the weakest score. New pairs beyond it do
  not match.
- If no pair should match, it is `inf` for similarities and `-inf` for distances.

When several candidates tie, the one with the fewest matches wins.

```python {.signature}
class MatchCounts(NamedTuple):
    true_positives: Tensor
    false_positives: Tensor
    true_negatives: Tensor
    false_negatives: Tensor
```

Each field is a one-dimensional float64 CPU tensor with one count per candidate
threshold, strictest first. The built-in metrics take a `MatchCounts`:

```python {.signature}
evaluation.match_accuracy(counts: MatchCounts) -> Tensor
```

```python {.signature}
evaluation.match_balanced_accuracy(counts: MatchCounts) -> Tensor
```

```python {.signature}
evaluation.match_f1(counts: MatchCounts) -> Tensor
```

`match_accuracy` counts correct decisions, so it favors the majority on
imbalanced validation pairs. `match_balanced_accuracy` is the mean of recall and
specificity. `match_f1` is `2 * tp / (2 * tp + fp + fn)`.

For verification, score labeled pairs with `distance.pairwise` and pass the
threshold to `MatchFinder`:

```python
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.evaluation import MatchFinder, match_f1, select_threshold

distance = CosineSimilarity()
first = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
second = torch.tensor([[1.0, 0.1], [1.0, 0.3], [0.2, 1.0], [-1.0, 0.0]])
is_same = torch.tensor([True, True, False, False])

selection = select_threshold(
    distance.pairwise(first, second),
    is_same,
    match_f1,
    is_inverted=distance.is_inverted,
)
finder = MatchFinder(distance, threshold=selection.threshold)
assert selection.value == 1.0
assert finder.is_match(first, second).tolist() == is_same.tolist()
```

For one-shot classification, score each query by its nearest reference and label
it by whether that neighbor has the right class. A query from an unknown class
is never correct, so the threshold learns to reject it:

```python
from pytorch_metric_learning.evaluation import CustomKNN

gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
gallery_labels = torch.tensor([0, 1])
query = torch.tensor([[1.0, 0.1], [0.1, 1.0], [1.0, 1.0], [-1.0, -0.5]])
query_labels = torch.tensor([0, 1, 2, 3])

similarities, neighbors = CustomKNN(distance)(query, 1, gallery)
correct = gallery_labels[neighbors[:, 0]] == query_labels
selection = select_threshold(
    similarities[:, 0], correct, match_f1, is_inverted=distance.is_inverted
)
accepted = similarities[:, 0] >= selection.threshold
assert accepted.tolist() == [True, True, False, False]
```

A custom metric can express a constraint. This one maximizes recall among
thresholds with a false match rate of at most 1%:

```python
def recall_at_low_false_match_rate(counts):
    recall = counts.true_positives / (counts.true_positives + counts.false_negatives)
    false_match_rate = counts.false_positives / (
        counts.false_positives + counts.true_negatives
    )
    return torch.where(false_match_rate <= 0.01, recall, torch.nan)


selection = select_threshold(
    similarities[:, 0],
    correct,
    recall_at_low_false_match_rate,
    is_inverted=True,
)
assert selection.value == 1.0
```

## Logging

Retrieval and accuracy calculation use ordinary Python module loggers under
`pytorch_metric_learning`. Applications configure handlers and levels:

```python
import logging

logging.basicConfig()
logging.getLogger("pytorch_metric_learning").setLevel(logging.INFO)
```

## FaissKNN

```python {.signature}
evaluation.FaissKNN(reset_before=True, reset_after=True, index_init_fn=None, gpus=None)
```

- **reset_before**: Build a new index from `index_init_fn(dimension)` on every
  call, adding `reference` before searching. If `False`, the call searches the
  existing index, and a `reference` passed to the call is appended to it first.
- **reset_after**: Discard the index after every call.
- **index_init_fn**: Callable returning a Faiss index for a dimension. Defaults to
  `faiss.IndexFlatL2`, whose distances are squared Euclidean.
- **gpus**: A non-empty list or tuple of Faiss GPU indices for the search.
  Without it, CUDA queries use all GPUs visible to Faiss. A value that is not a
  list or tuple raises `TypeError`; an empty one raises `ValueError`.

Faiss receives float32 inputs, and results return to the query's device. An
input stays on its GPU when the index is one GPU index on that same device, and
the call runs in that device's context on its current PyTorch stream; otherwise, including for sharded multi-GPU indices, Faiss receives a CPU copy.
A GPU search requires a GPU build of Faiss, one that provides
`faiss.GpuIndex`, with at least one visible GPU, and a CUDA query or `gpus`.
The `faiss-cpu` package never searches on a GPU, even when Faiss reports one.
A GPU search runs only when `k` is at most 2,048, the limit in Faiss's
`DeviceDefs.cuh`; otherwise it logs a warning and uses the CPU index.

With `reset_before=False`, the index accumulates: each call that passes
`reference` appends those rows, and their indices continue after the rows already
present, so repeated calls search duplicated rows. Search a persistent index by
calling it without `reference`, as below. `AccuracyCalculator` always passes its
reference set, so give it a `FaissKNN` with `reset_before=True`, the default;
indices past its reference set raise `ValueError`.

To keep a persistent index, disable both resets:

```python
import os
import tempfile

import torch
from pytorch_metric_learning.evaluation import FaissKNN

reference, more_reference, query = (
    torch.randn(50, 8),
    torch.randn(20, 8),
    torch.randn(5, 8),
)
knn = FaissKNN(reset_before=False, reset_after=False)
knn.train(reference)  # new index containing reference
knn.add(more_reference)  # append rows; indices continue at 50
distances, indices = knn(query, 10)
assert indices.shape == (5, 10)

with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "reference.index")
    knn.save(path)
    restored = FaissKNN(reset_before=False, reset_after=False)
    restored.load(path)
    assert torch.equal(restored(query, 10)[1], indices)
```

## FaissKMeans

```python {.signature}
evaluation.FaissKMeans(**kwargs)
```

Keyword arguments go to `faiss.Kmeans`. Calling `kmeans(x, num_clusters)` trains
on a float32 CPU copy of `x` and returns one `torch.long` cluster index per row
on `x`'s device.

## macOS OpenMP runtimes

The macOS `faiss-cpu` and PyTorch wheels each bundle an OpenMP runtime, and
loading both in one process can abort with `OMP: Error #15` or crash in
multithreaded Faiss calls. Setting both `KMP_DUPLICATE_LIB_OK=TRUE` and
`OMP_NUM_THREADS=1` before importing either library lets searches and k-means
complete, including alongside UMAP in the same process; Intel documents the
duplicate-runtime setting as unsupported. Prefer
`CustomKNN` on macOS, or install Faiss and PyTorch from one distribution that
shares a single OpenMP runtime. The library does not change the environment.
