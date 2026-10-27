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
