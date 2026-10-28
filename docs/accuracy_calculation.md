# Accuracy Calculation

The AccuracyCalculator class computes several accuracy metrics given a query and reference embeddings. It can be easily extended to create custom accuracy metrics.

```python
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
AccuracyCalculator(include=(),
                    exclude=(),
                    avg_of_avgs=False,
                    return_per_class=False,
                    k=None,
                    label_comparison_fn=None,
                    device=None,
                    knn_func=None,
                    kmeans_func=None)
```

### Parameters

- **include**: Optional. A list or tuple of strings, which are the names of metrics you want to calculate. If left empty, every k-NN metric is calculated. Clustering metrics (`"NMI"`, `"AMI"`, and custom metrics listed by `requires_clustering`) are calculated only when named here.
- **exclude**: Optional. A list or tuple of strings, which are the names of metrics you **do not** want to calculate.
- **avg_of_avgs**: If True, the average accuracy per class is computed, and then the average of those averages is returned. This can be useful if your dataset has unbalanced classes. If False, the global average will be returned.
- **return_per_class**: If True, the average accuracy per class is computed and returned.
- **k**: The number of nearest neighbors that will be retrieved for metrics that require k-nearest neighbors. The allowed values are:
  - `None`. Retrieve all available reference embeddings, excluding the query's own row when `ref_includes_query=True`.
  - An integer greater than 0. Retrieve at most this many neighbors, capped at `len(reference) - int(ref_includes_query)` for each call. The configured value is preserved when evaluating smaller galleries or subsets.
  - `"max_bin_count"`. This means k will be set to `max(bincount(reference_labels)) - self_count` where `self_count == 1` if the query and reference embeddings come from the same source.
- **label_comparison_fn**: A function that compares two torch arrays of labels and returns a boolean array. The default is `torch.eq`. If a custom function is used, then you must exclude clustering based metrics ("NMI" and "AMI"). The example below shows a custom function for two-dimensional labels. It returns `True` if the 0th column matches, and the 1st column does **not** match.
- **device**: A `torch.device` or device string such as `"cpu"` or `"cuda:0"`. Inputs are moved there before metrics are computed. If `None`, the query's device is used, and NumPy inputs use the CPU. The calculator never selects a GPU on its own.
- **knn_func**: A callable that takes in 4 arguments (`query, k, reference, ref_includes_query`) and returns `distances, indices`; see the [k-NN function contract](retrieval.md#k-nn-function-contract). The default is `CustomKNN(LpDistance(normalize_embeddings=False, power=2), batch_size=256)`, an exact PyTorch squared-Euclidean search on the input device. Pass [`FaissKNN()`](retrieval.md#faissknn) to use the optional Faiss backend.
- **kmeans_func**: A callable that takes in 2 arguments (`x, nmb_clusters`) and returns a 1-d tensor of cluster assignments. There is no default. It is required when a clustering metric is included; otherwise construction raises `ValueError`. [`FaissKMeans()`](retrieval.md#faisskmeans) requires the `faiss` extra. NMI and AMI also require scikit-learn, installed by the `clustering` extra; a missing package raises `ModuleNotFoundError` when those metrics are computed.

```python
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.utils.retrieval import CustomKNN

def example_label_comparison_fn(x, y):
    return (x[:, 0] == y[:, 0]) & (x[:, 1] != y[:, 1])

knn_func = CustomKNN(SNRDistance())
AccuracyCalculator(exclude=("NMI", "AMI"),
                    label_comparison_fn=example_label_comparison_fn,
                    knn_func=knn_func)
```

### Getting accuracy

Call the `get_accuracy` method to obtain a dictionary of accuracies.

```python
def get_accuracy(self,
	query,
    query_labels,
	reference=None,
	reference_labels=None,
	ref_includes_query=False,
	include=(),
	exclude=()
):
# returns a dictionary mapping from metric names to accuracy values
# The default metrics are:
# "precision_at_1"
# "r_precision"
# "mean_average_precision"
# "mean_average_precision_at_r"
# "mean_reciprocal_rank"
# "NMI" and "AMI" are computed only when included with a kmeans_func
```

- **query**: A 2D torch or numpy array of size `(Nq, D)`, where Nq is the number of query samples. For each query sample, nearest neighbors are retrieved and accuracy is computed.
- **query_labels**: A 1D torch or numpy array of size `(Nq)`. Each element should be an integer representing the sample's label.
- **reference**: A 2D torch or numpy array of size `(Nr, D)`, where Nr is the number of reference samples. This is where nearest neighbors are retrieved from.
- **reference_labels**: A 1D torch or numpy array of size `(Nr)`. Each element should be an integer representing the sample's label.
- **ref_includes_query**: Set to True if `query` is a subset of `reference` or if `query is reference`. Set to False otherwise.
- **include**: Optional. A list or tuple of strings, which are the names of metrics you want to calculate. If left empty, all metrics specified during initialization will be calculated.
- **exclude**: Optional. A list or tuple of strings, which are the names of metrics you do not want to calculate.

Note that labels can be 2D if a [custom label comparison function](#using-a-custom-label-comparison-function) is used.

Embeddings must be finite; NaN or infinite values raise `ValueError` before any search.

### Lone query labels

If some query labels don't appear in the reference set, then it's impossible for those labels to have non-zero k-nn accuracy. Zero accuracy for these labels doesn't indicate anything about the quality of the embedding space. So these lone query labels are excluded from k-nn based accuracy calculations.

For example, if the input `query_labels` is `[0,0,1,1]` and `reference_labels` is `[1,1,1,2,2]`, then 0 is considered a lone query label.

### Device and memory usage

The default backend runs on the input device and processes 256 queries at a
time, so its temporary distance block is `(256, len(reference))`. Its outputs
are `(len(query), effective_k)`. When `k=None`, all references except excluded
self-matches are retrieved, so output storage is quadratic for self-search. For large evaluations,
set `k` to a number, like `k = 1000`, or try `k = "max_bin_count"`.

With [`FaissKNN`](retrieval.md#faissknn), searches run on a CPU copy unless a
GPU-enabled Faiss build is installed, and on GPU only when `k <= 2048`.

### Explanations of the default accuracy metrics

- **AMI**:

  - [scikit-learn article](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/Adjusted_mutual_information)

- **NMI**:

  - [scikit-learn article](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
  - [Slides from Northeastern University](https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf)

- **mean_average_precision**:

  - [Slides from Stanford](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf)

- **mean_average_precision_at_r**:

  - [See section 3.2 of A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)

- **mean_reciprocal_rank**:

  - [Slides from Stanford](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf)

- **precision_at_1**:

  - Fancy way of saying "is the 1st nearest neighbor correct?"

- **r_precision**:

  - [See chapter 8 (page 161) of Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)

**Important note**

AccuracyCalculator's `mean_average_precision_at_r` and `r_precision` are correct only if `k = None`, **or** `k = "max_bin_count"`, **or** `k >= max(bincount(reference_labels))`

### Adding custom accuracy metrics

Let's say you want to use the existing metrics but also compute precision @ 2, and a fancy mutual info method. Extend the class, write a method for each metric, and add the methods to `metric_functions()`, which maps metric names to methods:

```python
from pytorch_metric_learning.utils import accuracy_calculator

class YourCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_2(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(
            knn_labels,
            query_labels[:, None],
            2,
            self.avg_of_avgs,
            self.return_per_class,
            self.label_comparison_fn,
        )

    def calculate_fancy_mutual_info(self, query_labels, cluster_labels, **kwargs):
        return fancy_computations

    def metric_functions(self):
        return {
            **super().metric_functions(),
            "precision_at_2": self.calculate_precision_at_2,
            "fancy_mutual_info": self.calculate_fancy_mutual_info,
        }

    def requires_clustering(self):
        return super().requires_clustering() + ["fancy_mutual_info"]

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_2"]
```

Every metric function is called with the following kwargs:

```python
kwargs = {"query": query,                    # query embeddings
    "reference": reference,                  # reference embeddings
    "query_labels": query_labels,
    "reference_labels": reference_labels,
    "ref_includes_query": e}  # True if query is reference, or if query is a subset of reference.
```

If your method requires a k-nearest neighbors search, then append your method's name to the `requires_knn` list, as shown in the above example. If any of your accuracy methods require k-nearest neighbors, they will also receive the following kwargs:

```python
	{"label_counts": label_counts,           # A dictionary mapping from reference labels to the number of times they occur
    "knn_labels": knn_labels,                # A 2d array where each row is the labels of the nearest neighbors of each query. The neighbors are retrieved from the reference set
    "knn_distances": knn_distances           # The knn_func distance for each neighbor in knn_labels; squared Euclidean by default
    "lone_query_labels": lone_query_labels   # The set of labels (in the form of a torch array) that have only 1 occurrence in reference_labels
    "not_lone_query_mask": not_lone_query_mask} # A boolean mask, where True means that a query element has at least 1 possible neighbor in reference.
```

If your method requires cluster labels, then append your method's name to the `requires_clustering` list, as shown in the above example. Like NMI and AMI, it is then calculated only when named in `include` and a `kmeans_func` is supplied. Then, if any of your methods need cluster labels, `self.get_cluster_labels()` will be called, and the kwargs will include:

```python
    {"cluster_labels": cluster_labels} # A 1D array with a cluster label for each element in the query embeddings.
```

Now when `get_accuracy` is called, the returned dictionary will contain `precision_at_2` and `fancy_mutual_info`:

```python
calculator = YourCalculator(
    include=("precision_at_1", "precision_at_2", "fancy_mutual_info"),
    kmeans_func=your_kmeans_func,
)
acc_dict = calculator.get_accuracy(query_embeddings,
    query_labels,
    reference_embeddings,
    reference_labels,
    ref_includes_query=True
)
# Now acc_dict contains "precision_at_1", "precision_at_2", and "fancy_mutual_info"
```

The [evaluation recipe](recipes/evaluation.md) passes any calculator to its
split, two-stream, and hierarchical helpers.

### Using a custom label comparison function

If you define your own `label_comparison_fn`, then `query_labels` and `reference_labels` can be 1D or 2D, and consist of integers or floating point numbers, as long as your `label_comparison_fn` can process them.

Example of 2D labels:

```python
def label_comparison_fn(x, y):
    return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])

# these are valid labels
labels = torch.tensor([
    (1, 3),
    (7, 4),
    (1, 4),
    (1, 5),
    (1, 6),
])
```

Example of floating point labels:

```python
def label_comparison_fn(x, y):
    return torch.abs(x - y) < 1

# these are valid labels
labels = torch.tensor([
    10.0,
    0.03,
    0.04,
    0.05,
])
```
