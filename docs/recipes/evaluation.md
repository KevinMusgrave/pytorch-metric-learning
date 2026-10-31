# Retrieval evaluation

Use this executable recipe from the repository root with the frozen development
environment. It extracts embeddings with an ordinary PyTorch loop, then passes
explicit query and reference tensors to
[`AccuracyCalculator`](../accuracy_calculation.md). It uses synthetic tensors,
completes in under a second on CPU, and performs no downloads. The Python file
in `docs/recipes` is the canonical executable source; it is not an installed
library API.

| Task                                                                                   | Executable source                                                                                                        | Command                                                       |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| Extract embeddings and evaluate split, two-stream, centroid, or hierarchical retrieval | [View evaluation.py](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v3.0.0rc4/docs/recipes/evaluation.py) | `uv run --frozen --no-sync python -m docs.recipes.evaluation` |

The command imports the installed package, so install the checkout as
described in the [development guide](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/DEVELOPMENT.md#set-up-a-checkout),
or prefix it with `PYTHONPATH=src`. Verify the contracts:

```sh
uv run --frozen --no-sync --env-file tests/cpu.env python -m pytest tests/recipes/test_evaluation.py
```

The command prints `precision_at_1` and `r_precision` for one query split against
itself and against a separate gallery. The tests assert exact metric values for
each composition below.

## Extract embeddings

`extract_embeddings(model, loader)` iterates a `DataLoader` yielding
`(data, labels)`, concatenates `model(data)` in loader order, and L2-normalizes
each row. It runs under `torch.inference_mode()`, so outputs carry no autograd
graph. The caller chooses `model.eval()` or training mode, the device, the
dtype, any trunk/embedder composition, and any dimensionality reduction. The
function changes none of them. Drop the normalization when the selected
distance should see raw embeddings.

Labels are used as returned by the loader. Map strings or sparse identifiers to
integers before evaluation, for example with
`torch.unique(labels, return_inverse=True)` over the full label set.

## Compose query and reference splits

`split_accuracy(calculator, query, references, query_in_reference)` takes
`(embeddings, labels)` pairs. When `query_in_reference` is true, it places the
query first in the reference tensor and passes `ref_includes_query=True`, so
each query's own row is excluded from its neighbors and from its relevant
count. Additional reference splits follow in the given order. When it is false,
the reference is only the given splits and no row is treated as a self-match.

| Query | References | Query in reference | Meaning                                           |
| ----- | ---------- | ------------------ | ------------------------------------------------- |
| `val` | none       | yes                | Leave-one-out retrieval within `val`              |
| `val` | `train`    | no                 | Retrieve `val` queries from a `train` gallery     |
| `val` | `train`    | yes                | Retrieve from `val ∪ train`, excluding each query |

Equal shapes or equal labels never imply self-matches. Pass the flag that
describes how the reference was assembled.

## Nearest-centroid accuracy

`nearest_centroid_accuracy(calculator, query, reference)` takes
`(embeddings, labels)` pairs. It replaces the reference with one mean embedding
per class and passes the centroids and their classes to `get_accuracy`. With one
reference per class, `precision_at_1` is nearest-centroid accuracy: 1 when the
nearest centroid has the query's class, 0 otherwise. Use a calculator that
selects `precision_at_1`:

```python
import torch
from docs.recipes.evaluation import nearest_centroid_accuracy
from pytorch_metric_learning.evaluation import AccuracyCalculator

reference = torch.tensor([[0.0, 0.0], [2.0, 0.0], [6.0, 0.0], [8.0, 0.0]])
reference_labels = torch.tensor([0, 0, 1, 1])
query = torch.tensor([[0.5, 0.0], [3.0, 0.0], [5.0, 0.0]])
query_labels = torch.tensor([0, 1, 1])

calculator = AccuracyCalculator(include=("precision_at_1",))
accuracy = nearest_centroid_accuracy(
    calculator, (query, query_labels), (reference, reference_labels)
)
assert accuracy["precision_at_1"] == 2 / 3
```

The calculator's `knn_func` sets the geometry, `avg_of_avgs` sets the weighting
of classes, and a query whose class has no reference is a
[lone query](../accuracy_calculation.md#lone-queries), excluded from the result.
The centroids are arithmetic means with the reference's dtype and device. Class
sums accumulate in float64, or float32 on MPS, so a large low-precision gallery
does not overflow before the division. For cosine geometry, normalize embeddings
before calling the function, and use a `knn_func` with `CosineSimilarity`, which
normalizes each centroid before comparing. Other metrics in the calculator would
rank against one reference per class, so they are not meaningful here.

## Two-stream evaluation

`two_stream_accuracy(calculator, anchors, references, labels)` evaluates
anchors against a separate reference stream whose row `i` shares label `i` with
anchor `i`. The reference never includes the anchors, so no row is removed.

## Parent-conditioned hierarchical retrieval

`parent_conditioned_accuracy` takes labels shaped `(N, L)`, ordered from the
finest level to the coarsest. For each level `l < L - 1`, it evaluates level-`l`
labels within each parent group defined by column `l + 1`, considering only
references with that parent. The result for a level is the unweighted mean over
parent groups present in the query, under the key `<metric>_level<l>`. With more
than one evaluated level, `AVERAGE_<metric>` is the unweighted mean over levels.
For example, a cat query is ranked only against other animals, and a truck query
only against other vehicles.

For global, unconditioned evaluation of one label level, pass that column to
`split_accuracy` directly.

## Published benchmark splits

The library does not download or parse datasets. The splits below are the
conventional metric-learning protocols for common benchmarks. Applications load
images with their own data pipeline and must respect each dataset's license and
terms.

| Dataset                                                                                                                                          | Training split                  | Test split                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- | ------------------------------ |
| [CUB-200-2011](https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf), Wah et al.                                                         | Classes 1–100, 5,864 images     | Classes 101–200, 5,924 images  |
| Cars196, Krause et al., _3D Object Representations for Fine-Grained Categorization_                                                              | Classes 1–98, 8,054 images      | Classes 99–196, 8,131 images   |
| [Stanford Online Products](https://arxiv.org/abs/1511.06452), Oh Song et al.                                                                     | `Ebay_train.txt`, 59,551 images | `Ebay_test.txt`, 60,502 images |
| [iNaturalist 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.pdf), Van Horn et al. | 325,846 images                  | 136,093 images                 |

The iNaturalist retrieval split is the `Inaturalist_{train,test}_set1.txt`
protocol published with [Smooth-AP](https://github.com/Andrew-Brown1/Smooth_AP#data)
by Brown et al. Train and test classes are disjoint in every row, so evaluate the
test split against itself with `query_in_reference=True`.
