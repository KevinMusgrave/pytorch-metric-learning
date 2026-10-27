# Next major release

This document describes changes implemented on the modernization branch. The
release is not yet accepted for publication; CUDA/NCCL and policy acceptance
remain outstanding in [the modernization plan](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/docs/plans/modernization.md).

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
`HDCMiner` no longer mutates externally set candidates; repeated calls with the
same candidates return the same pairs. `PNPLoss` and `SelfSupervisedLoss`
raise `ValueError` for unsupported variants and losses.

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
