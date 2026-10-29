# Accuracy Calculation

`AccuracyCalculator` computes retrieval and clustering metrics for query
embeddings against reference embeddings. Subclasses add metrics through the
[custom metric contract](#adding-custom-accuracy-metrics).

```python
evaluation.AccuracyCalculator(
    include=(),
    exclude=(),
    avg_of_avgs=False,
    k=None,
    label_comparison_fn=None,
    device=None,
    knn_func=None,
    kmeans_func=None,
)
```

Import it with `from pytorch_metric_learning.evaluation import AccuracyCalculator`.

### Parameters

- **include**: A sequence of metric names to compute. If empty, every metric
  except the clustering metrics (`"NMI"`, `"AMI"`, and names listed by
  `requires_clustering`) is selected. Clustering metrics are computed only when
  named here.
- **exclude**: A sequence of metric names to leave out.
- **avg_of_avgs**: If `True`, each k-NN metric is averaged per query class, and
  the result is the mean of those class averages, which gives every class equal
  weight. If `False`, the result is the mean over queries.
- **k**: The number of nearest neighbors retrieved for k-NN metrics:
  - `None`: retrieve every reference, excluding the query's own row when
    `ref_includes_query=True`.
  - An integer greater than 0: retrieve at most this many neighbors, capped at
    `len(reference) - int(ref_includes_query)` in each call.
  - `"max_bin_count"`: retrieve the largest number of relevant references of any
    query class, capped the same way and at least 1. A reference is relevant to a
    query when the label comparison matches their labels. With
    `ref_includes_query=True`, the query's own row is subtracted from the count
    only when the comparison matches a label with itself.

  Other values raise `ValueError`.

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

Metric names outside `metric_functions()` raise `ValueError`, and a string
instead of a sequence of names raises `TypeError`. The selected names are
available as the tuple `calculator.metrics`.

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

```python
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
NaN when no query has a relevant reference.

- **query**: A 2D torch tensor or NumPy array of shape `(Nq, D)`. Nearest
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

```python
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

It takes the same arguments as `get_accuracy` and returns a
`dict[str, dict[label, float]]`. Only selected k-NN metrics, those listed by
`requires_knn()`, are computed. Each inner dictionary:

- Is keyed by class label: a Python scalar for one-dimensional labels, or a
  tuple for multi-column labels.
- Contains every class present in `query_labels`, in sorted label order.
- Maps a class to NaN when none of its queries has a relevant reference.

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

### Device and memory usage

The default backend runs on the input device and processes 256 queries at a
time, so its temporary distance block is `(256, len(reference))`. Its outputs
are `(len(query), effective_k)`. When `k=None`, all references except excluded
self-matches are retrieved, so output storage is quadratic for self-search.
For large evaluations, set `k` to a number, such as `k=1000`, or use
`k="max_bin_count"`.

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

Extend the class, write one method per metric, and register it:

- `metric_functions()` maps metric names to methods. Each method is called with
  the keyword arguments listed below and accepts `**kwargs` for the rest.
- `requires_knn()` lists metrics that need the k-NN keyword arguments. Only
  these metrics are computed by `get_accuracy_per_class`.
- `requires_clustering()` lists metrics that need `cluster_labels`. Like NMI and
  AMI, they are computed only when named in `include`, and they require a
  `kmeans_func`.

A k-NN metric computes one value per query and returns
`self.aggregate(per_query, **kwargs)`. `aggregate` drops lone queries, applies
`avg_of_avgs`, and returns a per-class dictionary inside
`get_accuracy_per_class`. The module functions `precision_at_k`, `r_precision`,
`mean_average_precision`, and `mean_reciprocal_rank` in
`pytorch_metric_learning.evaluation.accuracy_calculator` return per-query
float64 tensors.

This example adds precision at 2 and cluster purity. `nearest_seed_clusters` is
a small deterministic stand-in for a k-means function such as `FaissKMeans()`:

```python
import torch
from pytorch_metric_learning.evaluation import AccuracyCalculator
from pytorch_metric_learning.evaluation.accuracy_calculator import precision_at_k


class Calculator(AccuracyCalculator):
    def calculate_precision_at_2(self, knn_labels, query_labels, **kwargs):
        per_query = precision_at_k(
            knn_labels, query_labels, 2, self.label_comparison_fn
        )
        return self.aggregate(per_query, **kwargs)

    def calculate_cluster_purity(self, query_labels, cluster_labels, **kwargs):
        _, classes = torch.unique(query_labels, return_inverse=True)
        _, clusters = torch.unique(cluster_labels, return_inverse=True)
        counts = torch.zeros(
            int(clusters.max()) + 1,
            int(classes.max()) + 1,
            dtype=torch.long,
            device=query_labels.device,
        )
        counts.index_put_(
            (clusters, classes), torch.ones_like(classes), accumulate=True
        )
        return counts.max(dim=1).values.sum().item() / len(query_labels)

    def metric_functions(self):
        return {
            **super().metric_functions(),
            "precision_at_2": self.calculate_precision_at_2,
            "cluster_purity": self.calculate_cluster_purity,
        }

    def requires_knn(self):
        return [*super().requires_knn(), "precision_at_2"]

    def requires_clustering(self):
        return [*super().requires_clustering(), "cluster_purity"]


def nearest_seed_clusters(x, num_clusters):
    return torch.cdist(x, x[:num_clusters]).argmin(dim=1)


calculator = Calculator(
    include=("precision_at_1", "precision_at_2", "cluster_purity"),
    kmeans_func=nearest_seed_clusters,
)
labels = torch.arange(3).repeat(4)
embeddings = torch.nn.functional.one_hot(labels).float() + 0.1 * torch.randn(12, 3)

accuracies = calculator.get_accuracy(embeddings, labels)
per_class = calculator.get_accuracy_per_class(embeddings, labels)
assert set(accuracies) == {"precision_at_1", "precision_at_2", "cluster_purity"}
assert set(per_class) == {"precision_at_1", "precision_at_2"}
assert set(per_class["precision_at_2"]) == {0, 1, 2}
```

Every metric method receives these keyword arguments:

| Keyword               | Value                                                                       |
| --------------------- | --------------------------------------------------------------------------- |
| `query`               | Query embeddings on the calculator's device                                 |
| `reference`           | Reference embeddings on the calculator's device                             |
| `query_labels`        | Query labels                                                                |
| `reference_labels`    | Reference labels                                                            |
| `ref_includes_query`  | Whether the query is the first `len(query)` rows of the reference           |
| `label_comparison_fn` | The label comparison in use, `torch.eq` by default                          |
| `per_class`           | `True` inside `get_accuracy_per_class`; pass it on through `self.aggregate` |

When any selected metric is listed in `requires_knn()`, every metric method also
receives:

| Keyword               | Value                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `knn_labels`          | `(Nq, k)` labels of each query's neighbors, nearest first, or `(Nq, k, L)` for 2D labels |
| `knn_distances`       | `(Nq, k)` `knn_func` values for those neighbors; squared Euclidean by default            |
| `relevant_counts`     | `(Nq,)` number of relevant references per query, without the query's own row             |
| `not_lone_query_mask` | `(Nq,)` boolean mask of queries with at least one relevant reference                     |
| `unique_query_labels` | The sorted unique query labels, one row per class                                        |
| `query_label_index`   | `(Nq,)` index of each query's label in `unique_query_labels`                             |

When any selected metric is listed in `requires_clustering()`, every metric
method also receives `cluster_labels`, a one-dimensional tensor with the
`kmeans_func` cluster index of each query row. The calculator asks for as many
clusters as there are unique query labels.

The [evaluation recipe](recipes/evaluation.md) passes any calculator to its
split, two-stream, and hierarchical helpers.

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
