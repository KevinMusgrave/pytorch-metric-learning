# Accuracy Calculation

`AccuracyCalculator` computes retrieval and clustering metrics for query
embeddings against reference embeddings. Additional metrics are registered with
`custom_metrics`; see [adding custom accuracy metrics](#adding-custom-accuracy-metrics).

```python {.signature}
evaluation.AccuracyCalculator(
    include=(),
    exclude=(),
    avg_of_avgs=False,
    k=None,
    label_comparison_fn=None,
    device=None,
    knn_func=None,
    kmeans_func=None,
    custom_metrics=None,
)
```

Import it with `from pytorch_metric_learning.evaluation import AccuracyCalculator`.

### Parameters

- **include**: A sequence of metric names to compute. If empty, every
  retrieval metric, built-in or custom, is selected. Clustering metrics
  (`"NMI"`, `"AMI"`, and custom `ClusteringMetric` entries) are computed only
  when named here.
- **exclude**: A sequence of metric names to leave out.
- **avg_of_avgs**: If `True`, each k-NN metric is averaged per query class, and
  the result is the mean of those class averages, which gives every class equal
  weight. If `False`, the result is the mean over queries.
- **k**: The largest number of nearest neighbors k-NN metrics may use, typed as
  [`types.NeighborLimit`](types.md#neighborlimit):
  - `None`: retrieve every reference, excluding the query's own row when
    `ref_includes_query=True`.
  - An integer greater than 0: retrieve at most this many neighbors, capped at
    `len(reference) - int(ref_includes_query)` in each call.
  - `"max_bin_count"`: retrieve the largest number of relevant references of any
    query class, capped the same way and at least 1. A reference is relevant to a
    query when the label comparison matches their labels. With
    `ref_includes_query=True`, the query's own row is subtracted from the count
    only when the comparison matches a label with itself.

  Other values raise `ValueError`. The calculator retrieves no more neighbors
  than the selected metrics need; see
  [neighbors retrieved](#neighbors-retrieved).

- **label_comparison_fn**: A function that compares two label tensors and
  returns a boolean tensor; see
  [custom label comparison](#using-a-custom-label-comparison-function). The
  default is `torch.eq`. A custom function cannot be combined with clustering
  metrics; that combination raises `ValueError` at construction.
- **device**: A `torch.device` or device string such as `"cpu"` or `"cuda:0"`.
  Inputs are moved there before metrics are computed. If `None`, the query's
  device is used, and NumPy inputs use the CPU. The calculator never selects a
  GPU on its own.
- **knn_func**: A callable `knn_func(query, k, reference, ref_includes_query)`
  returning `(distances, indices)`; see the
  [k-NN function contract](retrieval.md#k-nn-function-contract). The default is
  `CustomKNN(LpDistance(normalize_embeddings=False, power=2), batch_size=256)`,
  an exact PyTorch squared-Euclidean search on the input device. Pass
  [`FaissKNN()`](retrieval.md#faissknn) to use the optional Faiss backend.
- **kmeans_func**: A callable `kmeans_func(x, num_clusters)` returning a
  one-dimensional tensor of cluster indices, one per row of `x`. There is no
  default. Selecting a clustering metric without it raises `ValueError` at
  construction. [`FaissKMeans()`](retrieval.md#faisskmeans) requires the `faiss`
  extra. NMI and AMI also require scikit-learn, installed by the `clustering`
  extra; a missing package raises `ModuleNotFoundError` when those metrics are
  computed.
- **custom_metrics**: A mapping from new metric names to `RetrievalMetric` or
  `ClusteringMetric` registrations; see
  [adding custom accuracy metrics](#adding-custom-accuracy-metrics). A name that
  matches a built-in metric raises `ValueError`, and any other value type raises
  `TypeError`.

Metric names outside `calculator.available_metrics`, the read-only mapping of
built-in and custom registrations, raise `ValueError`, and a string instead of a
sequence of names raises `TypeError`. The selected names are available as the
tuple `calculator.metrics`.

```python
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.evaluation import AccuracyCalculator, CustomKNN

calculator = AccuracyCalculator(
    include=("precision_at_1", "r_precision"),
    k="max_bin_count",
    knn_func=CustomKNN(CosineSimilarity()),
)
assert calculator.metrics == ("precision_at_1", "r_precision")
```

### Getting accuracy

```python {.signature}
calculator.get_accuracy(
    query,
    query_labels,
    reference=None,
    reference_labels=None,
    ref_includes_query=False,
    include=(),
    exclude=(),
)
```

It returns a `dict[str, float]` from metric names to values. A k-NN metric is
NaN when no query has a relevant reference. Metrics are computed under
`torch.no_grad()`, so inputs that require gradients build no autograd graph.

- **query**: A 2D torch tensor or NumPy array
  ([`types.ArrayLike`](types.md#arraylike)) of shape `(Nq, D)`. Nearest
  neighbors are retrieved for each query row.
- **query_labels**: Labels of shape `(Nq,)`, or `(Nq, L)` with a
  [custom label comparison](#using-a-custom-label-comparison-function). Labels
  keep their dtype, so 64-bit identifiers compare exactly.
- **reference**: A 2D torch tensor or NumPy array of shape `(Nr, D)`, the set
  neighbors are retrieved from. If `None`, the query is evaluated against
  itself: the reference is the query and `ref_includes_query` is `True`.
- **reference_labels**: Labels of shape `(Nr,)` or `(Nr, L)`. Required when
  `reference` is given; otherwise `ValueError` is raised.
- **ref_includes_query**: Set to `True` when the query is exactly the first
  `len(query)` rows of `reference` and `reference_labels`, for example
  `reference = torch.cat([query, gallery])`. Each query's own row is then
  excluded from its neighbors and from its relevant count. The calculator
  checks this layout and raises `ValueError` if those rows do not equal the
  query and its labels; a query that is an arbitrary subset of the reference is
  not supported. Set it to `False` when the reference is a separate set.
- **include**: Metric names to compute in this call. It must be a subset of the
  metrics selected at construction; otherwise `ValueError` is raised. If empty,
  all selected metrics are computed.
- **exclude**: Metric names to leave out of this call.

Embeddings and labels must have matching lengths, and embeddings must be finite;
otherwise `ValueError` is raised before any search. Floating embeddings keep
their dtype, promoted to the wider of the two when query and reference differ;
integer embeddings are converted to float32.

### Per-class accuracy

```python {.signature}
calculator.get_accuracy_per_class(
    query,
    query_labels,
    reference=None,
    reference_labels=None,
    ref_includes_query=False,
    include=(),
    exclude=(),
)
```

It takes the same arguments as `get_accuracy`, also runs under
`torch.no_grad()`, and returns a `dict[str, dict[label, float]]`. Only selected
retrieval metrics are computed; clustering metrics have no per-class form and
are left out, and naming one in `include` raises `ValueError`. Each inner
dictionary:

- Is keyed by class label: a Python scalar for one-dimensional labels, or a
  tuple for multi-column labels.
- Contains every class present in `query_labels`, in sorted label order.
- Maps a class to NaN when none of its queries has a relevant reference, or
  when the metric is NaN for every such query.

```python
import torch
from pytorch_metric_learning.evaluation import AccuracyCalculator

calculator = AccuracyCalculator(include=("precision_at_1",))
query = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [9.0, 9.0]])
labels = torch.tensor([0, 0, 1, 2])
per_class = calculator.get_accuracy_per_class(query, labels)
assert per_class["precision_at_1"][0] == 1.0
assert per_class["precision_at_1"].keys() == {0, 1, 2}
assert torch.isnan(torch.tensor(per_class["precision_at_1"][1]))
```

### Lone queries

A query with no relevant reference cannot have nonzero k-NN accuracy, and its
zero says nothing about the embedding space. Such lone queries are excluded from
k-NN metrics. Relevance uses the label comparison, and with
`ref_includes_query=True` the query's own row does not count.

For example, if `query_labels` is `[0, 0, 1, 1]` and `reference_labels` is
`[1, 1, 1, 2, 2]`, the queries labeled 0 are lone queries. When every query is
lone, the calculator logs a warning and the k-NN metrics are NaN.

### Neighbors retrieved

Each call asks `knn_func` for the fewest neighbors that give the selected
metrics their values under `k`. A metric needs:

- `precision_at_1`: 1 neighbor.
- `r_precision` and `mean_average_precision_at_r`: the largest number of
  relevant references of any query, at least 1.
- `mean_average_precision` and `mean_reciprocal_rank`: every available
  neighbor.
- A custom `RetrievalMetric`: its declared `neighbors`.

The effective `k` is the smallest of the largest need among the selected
metrics, the configured `k` bound (every available neighbor for `None`; the
largest relevant count, at least 1, for `"max_bin_count"`), and
`len(reference) - int(ref_includes_query)`. Requesting only `precision_at_1`
therefore searches for one neighbor even with `k=None`. Metric values are the
same as with a search over every allowed neighbor; tied neighbors may be
ordered differently, as with any `torch.topk` search.

### Device and memory usage

The default backend runs on the input device and processes 256 queries at a
time, so its temporary distance block is `(256, len(reference))`. Its outputs
are `(len(query), effective_k)`. When `mean_average_precision` or
`mean_reciprocal_rank` is selected with `k=None`, all references except
excluded self-matches are retrieved, so output storage is quadratic for
self-search. For large evaluations, set `k` to a number, such as `k=1000`, or
use `k="max_bin_count"`.

With the default label comparison, relevant counts come from the reference
label counts, so they need no `(classes, len(reference))` comparison matrix. A
custom `label_comparison_fn` compares each query class with every reference
label.

Metrics use float64, which MPS lacks, so with MPS inputs the search stays on MPS
and metric arithmetic runs on CPU. A custom metric then receives CPU tensors.

With [`FaissKNN`](retrieval.md#faissknn), searches run on a CPU copy unless a
GPU build of Faiss is installed, and on GPU only when `k <= 2048`.

### Default accuracy metrics

- **AMI**: adjusted mutual information between query labels and k-means
  clusters of the query embeddings.
  - [scikit-learn article](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Adjusted_mutual_information)
- **NMI**: normalized mutual information between query labels and k-means
  clusters of the query embeddings.
  - [scikit-learn article](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
  - [Slides from Northeastern University](https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf)
- **mean_average_precision**: average precision over the retrieved neighbors,
  normalized by the number of relevant references.
  - [Slides from Stanford](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf)
- **mean_average_precision_at_r**: average precision over the first `R`
  neighbors, where `R` is the query's number of relevant references.
  - [See section 3.2 of A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)
- **mean_reciprocal_rank**: the reciprocal rank of the first relevant neighbor,
  or 0 if none is retrieved.
  - [Slides from Stanford](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf)
- **precision_at_1**: 1 if the nearest neighbor is relevant, 0 otherwise.
- **r_precision**: the fraction of the first `R` neighbors that are relevant.
  - [See chapter 8 (page 161) of Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)

`mean_average_precision_at_r` and `r_precision` are correct only if `k=None`,
`k="max_bin_count"`, or `k` is at least the largest number of relevant
references of any query. With a smaller integer `k`, the calculator logs a
warning.

### Adding custom accuracy metrics

Register each metric under a new name in `custom_metrics`. The calculator owns
selection, retrieval, clustering, lone-query handling, and reduction; a metric
only computes values.

```python {.signature}
evaluation.RetrievalMetric(function, neighbors="all")
```

```python {.signature}
evaluation.ClusteringMetric(function)
```

Import them, and the `RetrievalInputs` type they receive, from
`pytorch_metric_learning.evaluation`.

A **retrieval metric** is `function(inputs: RetrievalInputs) -> Tensor` that
returns one value per query: a one-dimensional tensor of length `Nq`, NaN where
the metric is undefined for that query. Any other shape raises `ValueError`.
`neighbors`, typed as [`types.MetricNeighbors`](types.md#metricneighbors),
declares how many ranked neighbors the function reads: a positive
integer, `"relevant"` for the largest number of relevant references of any
query, or `"all"` (the default) for every neighbor allowed by `k`. The calculator
may pass fewer columns when the gallery or `k` is smaller. Other values raise
`ValueError`. A selected metric declaring `"relevant"` logs the same warning as
`r_precision` when an integer `k` is smaller than the largest relevant count.

`RetrievalInputs` is a frozen dataclass with these fields:

```python {.signature}
evaluation.RetrievalInputs(
    query_labels,
    knn_labels,
    matches,
    relevant_counts,
    reference_labels,
    ref_includes_query,
)
```

| Field                | Value                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `matches`            | `(Nq, k)` boolean tensor; `True` where the neighbor is relevant to the query, nearest first |
| `relevant_counts`    | `(Nq,)` `torch.long` number of relevant references per query, without the query's own row   |
| `query_labels`       | `(Nq,)` or `(Nq, L)` query labels on the calculator's device                                |
| `knn_labels`         | `(Nq, k)` or `(Nq, k, L)` labels of each query's neighbors, nearest first                   |
| `reference_labels`   | `(Nr,)` or `(Nr, L)` labels of every reference, including rows the search did not return    |
| `ref_includes_query` | Whether the query is the first `Nq` rows of the reference, as passed to `get_accuracy`      |

Relevance uses the calculator's label comparison. `k` here is the
[effective number of neighbors](#neighbors-retrieved) for the call.

The calculator converts the values to float64, then ignores NaN values and
[lone queries](#lone-queries), whose `relevant_counts` is 0. `get_accuracy`
returns the mean of the remaining values, or the mean of per-class means with
`avg_of_avgs=True`. `get_accuracy_per_class` returns the per-class means. A
result with no remaining values is NaN.

A **clustering metric** is `function(query_labels: Tensor, cluster_labels:
Tensor) -> float`. `cluster_labels` holds the `kmeans_func` cluster index of
each query row, and the calculator asks for as many clusters as there are unique
query labels. Like NMI and AMI, a clustering metric is computed only when named
in `include`, requires a `kmeans_func`, cannot be combined with a custom
`label_comparison_fn`, and has no per-class value.

This example adds precision at 2, built from the exported
[`precision_at_k`](#built-in-retrieval-metric-functions), and cluster purity.
`nearest_seed_clusters` is a small deterministic stand-in for a k-means function
such as `FaissKMeans()`:

```python
from functools import partial

import torch
from pytorch_metric_learning.evaluation import (
    AccuracyCalculator,
    ClusteringMetric,
    RetrievalMetric,
    precision_at_k,
)


def cluster_purity(query_labels, cluster_labels):
    _, classes = torch.unique(query_labels, return_inverse=True)
    _, clusters = torch.unique(cluster_labels, return_inverse=True)
    counts = torch.zeros(
        int(clusters.max()) + 1,
        int(classes.max()) + 1,
        dtype=torch.long,
        device=query_labels.device,
    )
    counts.index_put_((clusters, classes), torch.ones_like(classes), accumulate=True)
    return counts.max(dim=1).values.sum().item() / len(query_labels)


def nearest_seed_clusters(x, num_clusters):
    return torch.cdist(x, x[:num_clusters]).argmin(dim=1)


calculator = AccuracyCalculator(
    include=("precision_at_1", "precision_at_2", "cluster_purity"),
    kmeans_func=nearest_seed_clusters,
    custom_metrics={
        "precision_at_2": RetrievalMetric(partial(precision_at_k, k=2), neighbors=2),
        "cluster_purity": ClusteringMetric(cluster_purity),
    },
)
labels = torch.arange(3).repeat(4)
embeddings = torch.nn.functional.one_hot(labels).float() + 0.1 * torch.randn(12, 3)

accuracies = calculator.get_accuracy(embeddings, labels)
per_class = calculator.get_accuracy_per_class(embeddings, labels)
assert set(accuracies) == {"precision_at_1", "precision_at_2", "cluster_purity"}
assert set(per_class) == {"precision_at_1", "precision_at_2"}
assert set(per_class["precision_at_2"]) == {0, 1, 2}
```

### Built-in retrieval metric functions

The built-in retrieval metrics are exported from
`pytorch_metric_learning.evaluation` as functions with the retrieval-metric
contract, so custom metrics can reuse them:

```python {.signature}
evaluation.precision_at_k(inputs, k)
```

```python {.signature}
evaluation.precision_at_1(inputs)
```

```python {.signature}
evaluation.r_precision(inputs)
```

```python {.signature}
evaluation.mean_average_precision(inputs)
```

```python {.signature}
evaluation.mean_average_precision_at_r(inputs)
```

```python {.signature}
evaluation.mean_reciprocal_rank(inputs)
```

Each takes a `RetrievalInputs` and returns a `(Nq,)` float64 tensor computed as
defined in [default accuracy metrics](#default-accuracy-metrics).
`precision_at_k` counts relevant neighbors among the first `k` columns of
`matches` and divides by `k`, so register it with `neighbors=k`. The functions
do not validate their inputs. For a lone query, whose `relevant_counts` is 0,
`r_precision`, `mean_average_precision`, and `mean_average_precision_at_r`
return NaN; the calculator excludes lone queries before reducing.

```python
from pytorch_metric_learning.evaluation import RetrievalInputs, r_precision

inputs = RetrievalInputs(
    query_labels=torch.tensor([0, 1]),
    knn_labels=torch.tensor([[0, 1, 0], [0, 0, 0]]),
    matches=torch.tensor([[True, False, True], [False, False, False]]),
    relevant_counts=torch.tensor([2, 0]),
    reference_labels=torch.tensor([0, 1, 0, 2]),
    ref_includes_query=False,
)
assert precision_at_k(inputs, 2).tolist() == [0.5, 0.0]
assert r_precision(inputs)[0] == 0.5 and r_precision(inputs)[1].isnan()
```

The [evaluation recipe](recipes/evaluation.md) passes any calculator to its
split, two-stream, centroid, and hierarchical helpers.

### Graded relevance and NDCG

Binary metrics treat a reference as relevant or not. With graded relevance, a
reference earns a nonnegative gain instead, so a same-parent neighbor can count
for less than a same-class neighbor and more than an unrelated one. NDCG, from
[Järvelin and Kekäläinen](https://doi.org/10.1145/582415.582418), scores how
close the ranking is to the best possible ordering of those gains.

```python {.signature}
evaluation.ndcg(inputs, relevance_fn, k=None)
```

- **inputs**: A `RetrievalInputs`.
- **relevance_fn**: A [`types.GradedRelevance`](types.md#gradedrelevance)
  function `relevance_fn(x, y)`, called like a
  [`label_comparison_fn`](#using-a-custom-label-comparison-function) but returning
  gains rather than booleans. A negative or non-finite gain raises `ValueError`.
- **k**: The ranking depth. With `None`, it is every neighbor in `inputs`. A
  larger `k` is capped at the available neighbors.

For a query with neighbors ranked `1..k` and gains `g_i`:

```text
DCG@k  = sum(g_i / log2(i + 1) for i in 1..k)
IDCG@k = DCG@k of the query's k largest gains over the whole reference set
NDCG@k = DCG@k / IDCG@k
```

The result is a `(Nq,)` float64 tensor with values from 0 to 1. Gains are used as
returned, so linear gain is `relevance` and exponential gain is
`2**relevance - 1`. The rules below keep the normalization correct:

- The ideal ranking draws from every reference, not only the retrieved neighbors,
  so a query that retrieved no relevant reference in its `k` neighbors scores 0
  and a truncated search cannot inflate a score.
- `IDCG@k` uses the same depth as `DCG@k`. The depth is the smaller of `k` and the
  columns in `inputs`, so a calculator bound or a small gallery shortens both.
- With `ref_includes_query=True`, the query's own row is left out of the ideal
  ranking, as it is left out of the neighbors.
- A query whose gains are all 0 has `IDCG@k = 0`, so its value is NaN and the
  calculator ignores it.
- Tied neighbors keep the order the search returned, as for every other metric.

Register `ndcg` as a custom metric with `functools.partial` and declare the
neighbors it reads. Multi-column labels then serve both kinds of metric: here
`label_comparison_fn` makes the finest column drive the binary metrics, and
`relevance_fn` grades both columns.

```python
from functools import partial

import torch
from pytorch_metric_learning.evaluation import (
    AccuracyCalculator,
    RetrievalMetric,
    ndcg,
)


def same_class(x, y):
    return x[..., 0] == y[..., 0]


def hierarchy_relevance(x, y):
    same_parent = x[..., 1] == y[..., 1]
    return same_parent.to(torch.float64) + same_class(x, y).to(torch.float64)


calculator = AccuracyCalculator(
    include=("mean_average_precision_at_r", "ndcg_at_3"),
    k=3,
    label_comparison_fn=same_class,
    custom_metrics={
        "ndcg_at_3": RetrievalMetric(
            partial(ndcg, relevance_fn=hierarchy_relevance, k=3), neighbors=3
        ),
    },
)
embeddings = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [9.0]])
labels = torch.tensor([[0, 0], [1, 0], [0, 0], [1, 0], [2, 1], [2, 1]])
accuracies = calculator.get_accuracy(embeddings, labels)
assert 0 < accuracies["ndcg_at_3"] < 1
```

The calculator's `k` bounds every metric, so `k="max_bin_count"` would shorten
`ndcg_at_3` to the largest relevant count from `label_comparison_fn`. Use `None`
or an integer of at least the metric's depth. [Lone queries](#lone-queries) are also decided by
`label_comparison_fn`, so a query with no same-class reference is excluded even if
it has same-parent references. To include such queries, set `label_comparison_fn`
to `relevance_fn(x, y) > 0`.

### Using a custom label comparison function

With a custom `label_comparison_fn`, `query_labels` and `reference_labels` can be
one- or two-dimensional and can hold integers or floating-point numbers, as long
as the function can process them. With the default `torch.eq`, labels must be
one-dimensional.

The function receives label tensors whose leading dimensions broadcast against
each other, such as `(Nq, 1)` query labels and `(Nq, k)` neighbor labels, with a
trailing label dimension for two-dimensional labels. It returns a boolean tensor
of the broadcast shape without that trailing dimension. Index label columns with
`x[..., i]` so the function works for every call.

This example matches two-dimensional labels when the first column matches and
the second column does **not** match:

```python
import torch
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.evaluation import AccuracyCalculator, CustomKNN


def label_comparison_fn(x, y):
    return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])


calculator = AccuracyCalculator(
    label_comparison_fn=label_comparison_fn,
    knn_func=CustomKNN(SNRDistance()),
)
labels = torch.tensor([(1, 3), (7, 4), (1, 4), (1, 5), (7, 6), (1, 3)])
accuracies = calculator.get_accuracy(torch.randn(6, 8), labels)
per_class = calculator.get_accuracy_per_class(torch.randn(6, 8), labels)
assert set(per_class["precision_at_1"]) == {(1, 3), (1, 4), (1, 5), (7, 4), (7, 6)}
```

Example of floating-point labels:

```python
def label_comparison_fn(x, y):
    return torch.abs(x - y) < 1


labels = torch.tensor([10.0, 0.03, 0.04, 0.05])
```
