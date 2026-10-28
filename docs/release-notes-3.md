# Version 3.0.0

Version 3 focuses the library on composable metric-learning components and
explicit PyTorch training and evaluation. It removes training orchestration,
simplifies extension contracts, and fixes numerical and distributed behavior.

## Supported environment

Python 3.12 or newer is required deliberately; Python 3.10 and 3.11 are no
longer supported. The tested minimums are PyTorch 2.14.0 and NumPy 2.5.3.
Published requirements permit newer versions. Development and release builds
use exact versions in the frozen lockfile. Optional Faiss and scikit-learn
backends require at least 1.15.1 and 1.9.1 respectively. Documentation tooling
is a development dependency; the former `docs` installation extra is removed.

## Explicit validation and typing

Public argument checks raise `TypeError` for incompatible types and `ValueError`
for invalid values or shapes, including with `python -O`. Custom miners use
`validate_output` instead of `output_assertion`; custom reducers use the
`check_sizes_*` hooks instead of `assert_sizes_*`. The distance validation helper
is now `utils.common_functions.check_distance_type`.

`BaseDistance.is_inverted` is a class attribute, not a constructor parameter.
Custom similarities set `is_inverted = True` on their class. `p` still controls
embedding normalization, including for similarities.

`pytorch_metric_learning.types` exposes `LossTerm`, `LossDict`, `PairIndices`,
`TripletIndices`, and `IndicesTuple`. Loss computation, mining, and reducer
boundaries have type annotations, and distributions include `py.typed`. Runtime
loss terms remain dictionaries and mined indices remain tuples. A default
reducer returns a tensor; `DoNothingReducer` returns the typed loss dictionary.

## Removed training orchestration

The `trainers` package and `utils.logging_presets` are removed. Applications own
ordinary PyTorch loops, optimizers, schedules, freezing, logging, and checkpoints.
The `with-hooks` extra, record-keeper integration, and TensorBoard dependency are
removed. Install the `faiss` extra when choosing that optional retrieval backend.

`TuplesToWeightsSampler` is removed. Compute embeddings in caller code, use
`utils.loss_and_miner_utils.convert_to_weights`, and pass full-dataset weights
to PyTorch's `WeightedRandomSampler`. Other metric-learning samplers remain.

Trainer key schemas, loss tracking, generic checkpoint helpers, and their public
utility functions are removed with their callers. There are no compatibility
aliases. The [agent-facing training recipes](recipes/training.md) preserve basic,
classifier, proxy, two-stream, cascaded, adversarial, weighted-sampling, and memory
composition with explicit gradient ownership.

## Removed evaluation orchestration and datasets

The `testers` package, `utils.inference.InferenceModel`, `run_pca`, and the
`datasets` package are removed, with the `datasets` extra. Extract embeddings
in application code and pass explicit query and reference tensors to
`AccuracyCalculator`. The [evaluation recipe](recipes/evaluation.md) preserves
split composition, two-stream evaluation, and parent-conditioned hierarchical
metrics, and records the published dataset splits and citations. String and
sparse labels are no longer remapped; convert them to integers before
evaluation. Common-function helpers used only by these components, including
label mapping, download, evaluation data-loader, and standardization helpers,
are removed.

The retrieval backends moved from `utils.inference` to `utils.retrieval`.

`AccuracyCalculator` changes:

- The default `knn_func` is an exact PyTorch squared-Euclidean search,
  `CustomKNN(LpDistance(normalize_embeddings=False, power=2), batch_size=256)`.
  It replaces `FaissKNN()` and returns the same distances; tie order may differ.
  Pass `FaissKNN()` explicitly to keep Faiss.
- `NMI` and `AMI` are no longer computed by default. Name them in `include`
  and pass a `kmeans_func`, such as `FaissKMeans()`. They need the new
  `clustering` extra for scikit-learn.
- `device=None` uses the query's device instead of selecting CUDA when
  available. Device strings are accepted.
- Non-finite embeddings raise a clear `ValueError`.
- Fixed `k` is capped at each gallery's available neighbor count, excluding
  self-matches when requested. Small galleries and parent-conditioned subsets
  can reuse the same calculator without changing its configured `k`.

`MatchFinder` returns boolean tensors on the input device instead of NumPy
arrays or Python booleans. With `return_tuples=True`, it returns an
`(num_matches, 2)` index tensor. A missing threshold raises `ValueError`.
Faiss and scikit-learn are imported only when used, and missing packages name
the extra to install. Earlier releases imported scikit-learn eagerly, which
set `KMP_DUPLICATE_LIB_OK` as a side effect on macOS; see
[retrieval backends](retrieval.md#macos-openmp-runtimes).

The core dependencies are now NumPy and PyTorch. tqdm, SciPy, Pillow, and
torchvision are no longer installed; scikit-learn moved to the `clustering`
extra.

All Jupyter notebooks and the separate example gallery are removed. The README
and canonical task index lead to small executable recipes beside API contracts.

## Explicit loss composition and parameter placement

`EmbeddingRegularizerMixin`, `WeightRegularizerMixin`, and `WeightMixin` are
removed. Every loss still accepts `embedding_regularizer` and
`embedding_reg_weight`. ArcFace, CosFace, large-margin softmax, SphereFace,
sub-center ArcFace, normalized softmax, ProxyAnchor, ProxyNCA, and SoftTriple
losses declare keyword-only `weight_regularizer`, `weight_reg_weight`, and
`weight_init_func`; pass these by name. `weight_init_func` is any callable that
initializes a tensor in place, including `torch.nn.init` functions.

Regularization penalties are added after the reducer instead of passing through
it. Loss values are unchanged for built-in reducers. A custom reducer that
changed `already_reduced` regularization entries no longer sees them.
`MultipleReducers` entries for `"embedding_reg_loss"` or `"weight_reg_loss"` are
no longer used. Custom losses override `sub_loss_names` instead of
`_sub_loss_names` and add weight penalties in `regularization_terms`.

Losses with learnable parameters no longer move or cast those parameters to
match the embeddings during `forward`. Previously one float64 call permanently
converted a float32 proxy matrix and broke the optimizer's next step. Call
`loss_func.to(device, dtype)` and include its parameters in the optimizer. A
mismatch outside autocast raises `ValueError`. `SubCenterArcFaceLoss` accepts
`num_classes` and `embedding_size` positionally, and
`get_outliers(..., return_dominant_centers=False)` now returns only the outliers.

Under autocast, ProxyNCA converts embeddings to the proxies' dtype for both
loss and logits computation. CPU float16 encoder outputs are supported while
the parameters and optimizer state retain their configured precision.

## Statistics removed

The `collect_stats` argument, the global `common_functions.COLLECT_STATS` flag,
and every recorded statistic attribute are removed, for example
`miner.pos_pair_dist`, `reducer.num_past_filter`, `loss.avg_angle`, and
`distance.final_avg_query_norm`. The consumer of these values, record-keeper,
was removed earlier. Recording forced host synchronization and side effects in
`forward`. Compute the same quantities from what the library returns:

```python
a1, p, a2, n = miner(embeddings, labels)
distances = miner.distance(embeddings)
pos_pair_dist = distances[a1, p].mean()
neg_pair_dist = distances[a2, n].mean()
```

Use `torch.nn.Module.register_forward_hook` to observe a module's inputs and
outputs without changing library code. The internal `ModuleWithRecords*` base
classes are replaced by `ModuleWithReducer`, `ModuleWithDistance`, and
`ModuleWithReducerAndDistance`.

## Extension hooks

`GenericPairLoss` subclasses implement `_compute_matrix_loss(mat, pos_mask,
neg_mask)` when constructed with `mat_based_loss=True`, or
`_compute_pair_loss(pos_pairs, neg_pairs, indices_tuple)` otherwise, instead of
one `_compute_loss` whose signature depended on the flag. Regularizer
`compute_loss` takes one argument, `x`, which is embeddings or weights.
`HDCMiner.set_idx_externally` and `reset_idx` are replaced by a `candidates`
argument: `miner(embeddings, labels, candidates=pairs)`. The miner keeps no
candidate state between calls. `PNPLoss` raises `ValueError` for unsupported
variants; `SelfSupervisedLoss` and `CrossBatchMemory` raise `TypeError` for
unsupported loss types.

## Smaller utility surface

`BaseLossWrapper` is removed. `CrossBatchMemory` and `SelfSupervisedLoss`
inherit `torch.nn.Module` directly and accept subclasses of their supported
loss types. Their `supported_losses()` methods return tuples of types instead
of class-name strings. ProxyNCA and its subclasses remain unsupported because
their learned proxies cannot serve the wrappers' separate-reference contract.

`EmbeddingsAlreadyPackagedAsTriplets` is removed. For consecutively packed
triplets, create `torch.arange(len(embeddings), device=embeddings.device)` and
use `.reshape(-1, 3).unbind(1)` as the loss's `indices_tuple`; see
[fixed triplet sampling](samplers.md#fixedsetoftriplets).

`common_functions.LOGGER`, `LOGGER_NAME`, and `set_logger_name` are removed.
Configure ordinary Python logging through
`logging.getLogger("pytorch_metric_learning")`; individual modules use
`logging.getLogger(__name__)`. The [retrieval logging example](retrieval.md#logging)
shows application-owned setup.

The redundant `common_functions.to_device`, `to_dtype`, `numpy_to_torch`,
`to_numpy`, and `is_list_or_tuple` helpers are removed. Use `Tensor.to`,
`torch.as_tensor`, `Tensor.cpu().numpy()`, and `isinstance(x, (list, tuple))`
directly. Explicit dtype conversions are honored during autocast as well.

`BatchedDistance(distance, batch_size=32)` receives its required callback when
called: `distance(query, reference, iter_fn=callback)`. It stores no callback
between calls. Wrapped distance options are available through `.distance`;
the wrapper directly exposes only `is_inverted` for search ordering.

The module containing `ProxyNCALoss` is now `losses.proxy_nca_loss`; the usual
`losses.ProxyNCALoss` import is unchanged. Distance/reducer composition bases
live in `utils.module_with_distance_and_reducer`.

`BaseDistance.largest_dist` is removed; use `smallest_dist` with the opposite
reduction or check `is_inverted`. `CrossBatchMemory` checkpoints store the queue
cursor and filled flag under a `queue_state` key instead of `_extra_state`.
`CrossBatchMemory` no longer moves or casts its queue to match each batch; place
it with `.to(device, dtype)`. The label queue dtype is the new keyword-only
`label_dtype`, `torch.long` by default.
`DynamicSoftMarginLoss` likewise checks rather than moves its histogram buffer.

`MarginLoss` registers a fixed `beta` as a checkpointed buffer; both fixed and
learnable beta follow `.to(device, dtype)`. `CrossBatchMemory.reset_queue()`
clears its buffers in place, preserving device, dtype, and storage.
`SparseCentersRegularizer` registers its same-class mask as a non-persistent
buffer so module placement moves it once instead of copying it on every call.
Autocast inputs to `CrossBatchMemory` are converted to its queue dtype while
preserving anchor gradients. Dynamic soft margin accumulates histogram
contributions in the buffer's dtype, including with CPU float16 autocast.
Both operations preserve the configured buffer precision.

Batched `CustomKNN` searches release per-call distance/index matrices and
autograd graphs after use, including on errors. `AccuracyCalculator` therefore
does not retain these potentially quadratic allocations between evaluation and
training.

`ManifoldLoss(l, K, lambdaC)` is now `ManifoldLoss(embedding_size, num_proxies,
lambda_c)`, matching the other losses' parameter names.

P2SGrad normalizes its weights under `torch.no_grad()` and uses a differentiable
snapshot for each computation. Parameter identity and storage remain stable,
and multiple forwards can precede one backward without changing saved weights.

`MatchFinder.is_match()` now applies the configured normalization and distance
power, matching the semantics of `get_matching_pairs()` for cosine similarity,
normalized distances, and squared distances.

`common_functions.TorchInitWrapper` is removed; pass any in-place initializer,
for example `functools.partial(torch.nn.init.kaiming_uniform_, a=1)`, as
`weight_init_func`. The unused `loss_and_miner_utils.matched_size_indices` and
`repeat_to_match_size` are removed. Reducers dispatch reduction types through an
explicit table: tuple reductions default to `element_reduction` with anchor
indices, so custom reducers implement only `element_reduction` unless tuples
need different treatment. `AccuracyCalculator` metrics are registered by
overriding `metric_functions()` instead of discovered from `calculate_` method
names.

## Sampler randomness

`MPerClassSampler`, `HierarchicalSampler`, and `FixedSetOfTriplets` use
PyTorch random number generators and accept a keyword-only `generator`.
`common_functions.NUMPY_RANDOM`, `safe_random_choice`, `get_labels_to_indices`,
`try_next_on_generator`, and `EmbeddingDataset` are removed; seed with
`torch.manual_seed` or pass a `torch.Generator`. Sampling distributions are
unchanged, but seeded sequences differ from earlier releases.
`MPerClassSampler` no longer yields more indices than `len()` reports when one
pass over the classes exceeds `length_before_new_iter`. `HierarchicalSampler`
reports its length without generating batches during construction.

## Correctness and performance

Verifying every documented equation found and fixed these defects, each with a
reference test that fails on the previous implementation:

- `SmoothAPLoss` built positive sets from positional blocks of the wrong size;
  results were correct only when the number of classes equaled the samples per
  class. It now derives positives from the labels, so labels need not be grouped
  or balanced, and it returns the embeddings' dtype.
- `ThresholdConsistentMarginLoss` ignored `ref_emb`, indexing the query
  embeddings with reference indices.
- `LiftedStructureLoss` compared query and reference indices when given a
  separate `ref_emb`; it now raises `ValueError` for that unsupported case. It is no longer listed as
  supported by `CrossBatchMemory` or `SelfSupervisedLoss`, which pass reference
  embeddings.
- `TupletMarginLoss` returned NaN when an anchor and positive were identical.
- `AngularLoss` normalized anchors and positives but not negatives. Values
  change only for unnormalized embeddings.

Distributed wrappers now define full-batch and local-anchor gradients explicitly,
handle non-distributed fallback, and synchronize queue order. Cross-batch memory
preserves state across empty enqueues, checkpoint restore, and multiple forwards
before backward. Dynamic soft margin uses detached, checkpointed histogram state
and handles histogram endpoints safely. See [distributed contracts](distributed.md)
and [loss contracts](losses.md) for the supported combinations.

All-triplet enumeration avoids an unconditional cubic candidate mask while
preserving index order, values, and gradients. See [performance measurements](performance.md)
for the measured workload and its limits.
