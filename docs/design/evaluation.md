# Retrieval and evaluation

Evaluation owns retrieval metrics and the nearest-neighbor, threshold-matching,
and clustering backends they use. It operates on embedding tensors supplied by
the caller. It does not own models, model modes, data loaders, dataset
acquisition, split orchestration, dimensionality reduction, logging,
checkpoints, or optimizer scheduling. Those application concerns are shown in
the [evaluation recipe](../recipes/evaluation.md).

Labels keep the dtype the caller supplies, and floating embeddings keep their
dtype, promoted to a common one when query and reference differ; integer or
NumPy inputs become float32 embeddings. Casting labels to float32 made large
integer identifiers collide (PR #763). Results are Python floats because they
are consumed by logging and comparison code rather than by autograd.
`get_accuracy` returns `dict[str, float]`, and `get_accuracy_per_class` returns
per-class values keyed by class label, every query class included and NaN for
classes without relevant references. A single method returning either floats or
unlabeled lists made results ambiguous to type checkers and to readers when
classes were missing.

A metric is a registration record, not a method on a subclass.
`RetrievalMetric(function, neighbors)` maps a frozen `RetrievalInputs` (the
relevance matrix, relevant counts, and query and neighbor labels) to one value
per query, NaN where undefined; `ClusteringMetric(function)` maps query and
cluster labels to one float. The record's type declares which backend the
metric needs, and `neighbors` declares how much ranking it reads. The calculator
owns lone-query exclusion and every reduction (pooled mean, mean of class means,
per-class means), so a metric cannot return a type that depends on how it is
reduced. The earlier contract required three parallel method overrides
(`metric_functions`, `requires_knn`, `requires_clustering`), passed every
internal keyword to every metric, and returned a float or a dictionary
depending on a `per_class` flag the metric had to forward; each internal name
became public API. Clustering metrics have no per-query values and therefore
no per-class form; naming one in `get_accuracy_per_class` raises instead of
returning an incomplete result. Evaluation runs under `torch.no_grad()`:
results are Python floats, so a graph built from embeddings that still
require gradients would only hold memory until the call returned.

Query and reference sets remain explicit. Self-matches are excluded only when
the caller states that the query occupies the first rows of the reference set.
Equal shapes or labels never imply that relationship. The k-NN backends remove each
query's own index rather than its first neighbor, because colocated
embeddings with different labels can tie with the query and sort before it. Label comparisons may be
customized and need not be scalar equality. Relevant counts, lone queries, and
`k="max_bin_count"` all use the comparator: with self-search, the query's own
row is subtracted only when the comparator says its label matches itself.
Subtracting it unconditionally gave zero relevant references, infinite MAP, and
NaN R-precision for comparators that reject self-matches.

The effective `k` of a call is the largest neighbor count the selected metrics
declare, bounded by the configured `k` and the available references.
Precision at 1 needs one neighbor; R-precision and MAP@R need the largest
relevant count; MAP and MRR need every neighbor the configured `k` allows.
Metric values are unchanged because no metric reads past its declared count,
and requesting only precision at 1 no longer materializes a `(queries,
references)` result. With the default equality comparator, relevant counts
come from `torch.unique(..., return_counts=True)` over the reference labels and
a sorted lookup per query class, which is linear in the reference count instead
of a `(classes, references)` comparison. A custom comparator has no ordering to
exploit, so it is compared with every reference label.

The default k-NN backend is an exact PyTorch search on the input device:
squared Euclidean `LpDistance` without normalization, matching the ordering and
distance values of the former `faiss.IndexFlatL2` default. Queries are processed
in blocks of 256 so temporary memory is one `(256, references)` block plus the
`(queries, k)` outputs. Tie order follows `torch.topk` and is not part of the
contract. The calculator moves inputs only to a device the caller names; it
never chooses a GPU itself.

A positive integer `k` is an upper bound on the available neighbors for each
evaluation call. The calculator caps it at the reference count minus one when
self-matches are excluded, before the backend adds its self-search candidate.
The configured `k` remains unchanged so one calculator can evaluate galleries
and parent-conditioned subsets of different sizes. This bound applies to every
backend and prevents requesting nonexistent neighbors or padding metric inputs.

Search results belong to the caller, not to the backend. Batched search
iterates over `BatchedDistance` blocks within one call and stores nothing on
either module, so result matrices are released when the caller drops them;
metric-only calls retain no k-NN outputs. `CustomKNN` and `MatchFinder` search
under `torch.no_grad()`, like the non-differentiable Faiss backend, so every
backend returns results without autograd history. The calculator checks the
returned index range once per call and raises `ValueError` for indices outside
the reference set. A `FaissKNN` with `reset_before=False` appends each passed
reference to its persistent index, and the resulting offsets used to surface as
an `IndexError` from label lookup.
With `k=None`, those outputs are quadratic in the reference count, so retaining
them between evaluation and training would consume substantial GPU memory.

Threshold matching applies the same distance normalization and power for
matrix comparisons and corresponding-row comparisons through
`BaseDistance.pairwise`, so it needs no quadratic intermediate. Evaluation diagnostics use standard module loggers;
applications configure the `pytorch_metric_learning` logger hierarchy.

NDCG is a function a caller registers, not a built-in metric, because a gain
scale is a modeling choice with no default. Its normalization must rank every
reference: a metric that saw only retrieved neighbors would compute the ideal
ranking from the same truncated list, so a poor search would look perfect.
`RetrievalInputs` therefore carries the reference labels and the self-match flag.
The ideal is computed once per unique query label, like relevant counts with a
custom comparator, so the cost is `(classes, references)` label comparisons
rather than `(queries, references)`. Binary metrics keep `label_comparison_fn`;
`relevance_fn` is a separate argument of the metric so one calculator can report
both from the same multi-column labels, which is what #339 asked for without
adding per-metric label columns to the calculator.

Threshold selection takes scores and labels rather than a model or embeddings,
so inference, splits, and pair construction stay with the application and any
source of scores works. The metric is a required callable over confusion counts
for every candidate, computed in one sorted pass, so constraints such as a
false-match-rate ceiling need no extra options. `is_inverted` has no default
because guessing the direction of a score silently inverts every decision. A
returned threshold is the midpoint of the gap it separates: both neighbors give
the same validation result, and an observed score would sit on the validation
data's edge. Comparisons convert a Python threshold to the score dtype, so a
midpoint that rounds onto the excluded score falls back to the kept score, which
preserves the reported value under `MatchFinder`. Selection runs on CPU copies
because float64 counts are unavailable on MPS and a one-dimensional sort is cheap
relative to the inference that produced the scores.

Faiss and scikit-learn are optional and imported only when a Faiss backend is
constructed or a mutual-information metric is computed. Missing packages raise
`ModuleNotFoundError` naming the extra. Clustering metrics run only when named
in `include` with an explicit `kmeans_func`, so a default calculator has no
optional dependency or hidden clustering backend. On macOS, the separately
bundled OpenMP runtimes of the Faiss and PyTorch wheels require
`KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` to coexist; the CPU test
profile sets both.

Faiss runs on CPU copies with `faiss-cpu`, including when embeddings originate
on CUDA. GPU search requires a GPU build, detected by its `GpuIndex` type:
`faiss-cpu` on macOS reports a GPU through `get_num_gpus()` without providing
GPU indexes. Backend-specific GPU acceleration is distinct from
PyTorch CUDA support. A CUDA tensor is passed directly only to one GPU index on
that tensor's device. `faiss.contrib.torch_utils` requires a single-device index
and takes a raw pointer without comparing devices, so sharded indices and tensors
on another GPU receive CPU copies. It binds the stream of
`torch.cuda.current_device()` rather than of the index's device, so direct calls
and their input conversion run inside `torch.cuda.device` of that tensor;
otherwise Faiss work on one GPU is not ordered against the PyTorch stream
producing its input.
The Faiss GPU path is limited to `k <= 2048`, the maximum
in Faiss's [`DeviceDefs.cuh`](https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/utils/DeviceDefs.cuh).

Metric definitions and examples live in `docs/accuracy_calculation.md`;
backend contracts live in `docs/retrieval.md`.
