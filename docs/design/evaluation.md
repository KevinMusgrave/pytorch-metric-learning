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
either module, so result matrices and their autograd graphs are released when
the caller drops them; metric-only calls retain no k-NN outputs.
With `k=None`, those outputs are quadratic in the reference count, so retaining
them between evaluation and training would consume substantial GPU memory.

Threshold matching applies the same distance normalization and power for
matrix comparisons and corresponding-row comparisons through
`BaseDistance.pairwise`, so it needs no quadratic intermediate. Evaluation diagnostics use standard module loggers;
applications configure the `pytorch_metric_learning` logger hierarchy.

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
PyTorch CUDA support. The Faiss GPU path is limited to `k <= 2048`, the maximum
in Faiss's [`DeviceDefs.cuh`](https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/utils/DeviceDefs.cuh).

Metric definitions and examples live in `docs/accuracy_calculation.md`;
backend contracts live in `docs/retrieval.md`.
