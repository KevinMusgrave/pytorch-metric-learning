# Next major release

This document describes changes implemented on the modernization branch. The
release is not yet accepted for publication; CUDA/NCCL and policy acceptance
remain outstanding in [the modernization plan](plans/modernization.md).

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

## Correctness and performance

Distributed wrappers now define full-batch and local-anchor gradients explicitly,
handle non-distributed fallback, and synchronize queue order. Cross-batch memory
preserves state across empty enqueues, checkpoint restore, and multiple forwards
before backward. Dynamic soft margin uses detached, checkpointed histogram state
and handles histogram endpoints safely. See [distributed contracts](distributed.md)
and [loss contracts](losses.md) for the supported combinations.

All-triplet enumeration avoids an unconditional cubic candidate mask while
preserving index order, values, and gradients. See [performance measurements](performance.md)
for the measured workload and its limits.
