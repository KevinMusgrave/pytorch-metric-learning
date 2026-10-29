# Measuring performance

Run benchmarks from the repository's frozen development environment. These
measurements establish a reference for changes; they do not impose a universal
speed requirement across machines. The runner imports this checkout's `src`
directory directly; an editable project install is optional.

```sh
uv run --frozen --no-sync python -m benchmarks.run \
  --batch-size 128 --dimensions 128 --repeats 20 --warmup 5 \
  --output /tmp/pml-cpu.json
```

The default run measures contrastive loss, all-triplet margin loss, pair-margin
mining, and normalized-softmax proxy loss. Inputs are standard-normal tensors
with balanced contiguous labels, four examples per class. Distances normalize
embeddings inside the measured call. The contrastive negative margin is 1.4,
triplet margin is 0.2, miner margins are 0.2 and 1.4, and proxy temperature is
0.1. The proxy benchmark includes gradients for its learnable weights.

Change `--batch-size`, `--dimensions`, `--samples-per-class`, `--dtype`, and
`--threads` to describe the intended workload. Batch size must contain at least
two complete classes. Each case uses its own seeded inputs, independent of how
many random parameters the implementation initializes. Record multiple shapes
before drawing conclusions about scaling.

```sh
uv run --frozen --no-sync python -m benchmarks.run \
  --device cuda --batch-size 512 --dimensions 128 \
  --output /tmp/pml-cuda.json

uv run --frozen --no-sync python -m benchmarks.run \
  --case proxy --mode compile --output /tmp/pml-compiled-proxy.json
```

Compilation uses `torch.compile(..., fullgraph=True)`. An unsupported operation
fails the command; there is no automatic eager fallback. The first step,
including lazy graph compilation, is reported separately. Module construction,
compiler registration, and process startup precede that measurement. The default five
additional warmup steps precede twenty measured steps. Use a fresh process when
comparing startup behavior and record whether the compiler cache was warm.

For distributed measurements, run one process per device:

```sh
uv run --frozen --no-sync torchrun --standalone --nproc_per_node=2 \
  -m benchmarks.run --case distributed --device cuda \
  --batch-size 128 --dimensions 128 --output /tmp/pml-nccl.json
```

The distributed case includes a linear encoder, full-batch contrastive loss,
autograd-aware gathering, and DDP gradient synchronization. `--batch-size` is
per rank; the report also records the global size. Compare this case with its
own previous measurements, since the single-process contrastive case does not
include an encoder. Gloo/CPU works with `--device cpu` for local experiments.
Correctness tests launch their own workers and must not be wrapped in torchrun.

For a local macOS Gloo run, use the loopback interface and an explicit local
address so rendezvous does not depend on the laptop's hostname resolving:

```sh
GLOO_SOCKET_IFNAME=lo0 uv run --frozen --no-sync torchrun \
  --standalone --local-addr=127.0.0.1 --nproc_per_node=2 \
  -m benchmarks.run --case distributed --device cpu \
  --batch-size 128 --dimensions 128 --output /tmp/pml-gloo.json
```

Every measured step uses a fresh computation graph and cleared gradients.
Forward timing includes graph construction; backward timing includes gradient
computation and synchronization. CUDA synchronization brackets both intervals.
Distributed barriers align the start of steps outside the measured intervals.
Reports include raw per-rank samples, medians, spread, and the Git revision.
Use the slowest rank when comparing distributed performance.

CUDA peak memory is additional allocated tensor memory above the pre-forward
baseline, including saved intermediates and gradients. Model parameters and
inputs are already present at that baseline. This is not allocator-reserved
memory or total device usage. CPU peak tensor memory is reported as null.
The runner verifies finite losses/gradients or valid miner indices before
sampling; mathematical correctness is established by the reference tests.

## Initial CPU reference

Measured on 2026-10-27 UTC at clean revision
`f8186f88d475ca90905a45a7fca191ffcfb2be16`: Apple M1 Pro, macOS 26.7.1,
Python 3.12.13, PyTorch 2.14.0, float32, one thread per rank, 128 dimensions,
four examples per class, seed 17, ten warmup steps after the first call, and
fifty measured steps. Values below are median milliseconds. This is a shared
development laptop, not an isolated performance lab; use the raw sample spread
and repeat on the target hardware before setting a performance budget.

| Case                     | Batch 128 forward | Batch 128 backward | Batch 512 forward | Batch 512 backward |
| ------------------------ | ----------------: | -----------------: | ----------------: | -----------------: |
| Contrastive              |             0.353 |              0.293 |             3.846 |              2.667 |
| All-triplet margin       |             6.076 |              0.682 |           341.808 |              8.841 |
| Pair-margin miner        |             0.367 |                  — |             4.784 |                  — |
| Normalized-softmax proxy |             0.092 |              0.134 |             0.252 |              0.343 |

Raw results: [batch 128](benchmarks/cpu-m1-pro-b128.json) and
[batch 512](benchmarks/cpu-m1-pro-b512.json). The large triplet forward cost was
a concrete optimization target: the baseline enumeration constructed a cubic
candidate mask even though this balanced label distribution produces only
quadratically many valid triplets.

The [two-rank Gloo measurement](benchmarks/gloo-m1-pro-b128.json) uses 64 rows
per rank, 128 globally. The slower median rank took 1.491 ms forward and
1.129 ms backward. This includes the encoder and communication, unlike the
single-process contrastive row above.

The [full-graph compiled proxy measurement](benchmarks/compiled-proxy-m1-pro-b128.json)
succeeded at batch 128: 0.127 ms forward and 0.130 ms backward, versus 0.092 ms
and 0.134 ms eager. Its first forward/backward took 9.467 seconds; compiler
registration and import costs are additional and were not timed. Compiler-cache
state was not isolated. This workload does not justify enabling compilation by
default. CUDA timing and peak tensor memory measurements appear below.

## Valid-triplet expansion

The clean-revision follow-up at `24cef70da287973ed41e09ed26616e2b03b083d9`
uses the same hardware, float32 inputs, seed, shapes, one CPU thread, ten warmup
steps, and fifty measured steps. Only actual positive pairs are expanded by
their anchors' negative counts; ordered indices, values, and gradients remain
covered by the independent numerical references.

| Batch | Baseline forward | Expanded forward | Baseline backward | Expanded backward |
| ----- | ---------------: | ---------------: | ----------------: | ----------------: |
| 128   |         6.076 ms |         1.032 ms |          0.682 ms |          0.658 ms |
| 512   |       341.808 ms |        15.358 ms |          8.841 ms |          9.052 ms |

The batch-512 forward is about 22.3 times faster on this workload; backward is
roughly unchanged. This removes the unconditional cubic candidate allocation,
but materialized valid triplets still consume memory proportional to their
count. CPU peak tensor memory remains unmeasured, and these timings do not
establish a CUDA result. Raw results: [batch 128](benchmarks/triplet-expanded-m1-pro-b128.json)
and [batch 512](benchmarks/triplet-expanded-m1-pro-b512.json).

## Explicit loss composition

The explicit regularization and parameter-placement change at
`190a39a06e2ab5cb6d1623a341cda8cede4da560` was compared with its parent,
`31f7e1e`, on the same hardware and settings as above: float32, one thread, 128
dimensions, four examples per class, seed 17, ten warmup steps, and fifty
measured steps. Validation losses are identical between revisions. Values are
median milliseconds.

| Case                     | Batch | Before forward | After forward | Before backward | After backward |
| ------------------------ | ----- | -------------: | ------------: | --------------: | -------------: |
| Contrastive              | 128   |          0.361 |         0.359 |           0.304 |          0.297 |
| Contrastive              | 512   |          3.802 |         3.678 |           2.672 |          2.576 |
| Normalized-softmax proxy | 128   |          0.092 |         0.088 |           0.131 |          0.131 |
| Normalized-softmax proxy | 512   |          0.252 |         0.242 |           0.340 |          0.332 |

The differences are within run-to-run variation on this laptop; the change
neither adds nor removes measurable cost on these workloads. Raw results are
`docs/benchmarks/composition-{before,after}-{contrastive,proxy}-b{128,512}.json`.

## CUDA and NCCL reference

Measured on 2026-10-28 UTC at clean revision
`0616ea964cc6b87dc8a8c0abcb562204b16bf1a3`: Linux x86_64, Intel Xeon Gold 6342,
two NVIDIA A40 GPUs (46,068 MiB each), NVIDIA driver 595.91.07, Python 3.12.13,
PyTorch 2.14.0+cu130, CUDA 13.0, and NCCL 2.30.7. Each workload uses float32,
128 dimensions, four examples per class, seed 17, one CPU thread per rank,
ten warmup steps after the first call, and fifty measured steps. Timings are
median milliseconds; memory is peak additional allocated tensor memory in MiB.

This host requires `NCCL_P2P_DISABLE=1`: direct GPU-to-GPU transfers hang even
in a bare PyTorch/NCCL probe. Distributed results therefore include NCCL's
alternative transport and cannot establish direct peer-to-peer performance.
See the [host workaround](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/DEVELOPMENT.md#cuda-and-distributed-verification).
These measurements are specific to this shared rented host; retain the raw
sample spread when comparing later runs.

| Case                     | Batch | Forward | Backward | Peak MiB |
| ------------------------ | ----: | ------: | -------: | -------: |
| Contrastive              |   128 |   2.787 |    3.890 |    1.299 |
| Contrastive              |   512 |   3.339 |    5.119 |   19.478 |
| All-triplet margin       |   128 |   3.138 |    3.218 |    4.045 |
| All-triplet margin       |   512 |   3.686 |    4.347 |   65.729 |
| Pair-margin miner        |   128 |   0.763 |        — |    0.546 |
| Pair-margin miner        |   512 |   0.919 |        — |    8.712 |
| Normalized-softmax proxy |   128 |   1.228 |    2.322 |    0.331 |
| Normalized-softmax proxy |   512 |   1.582 |    3.205 |    1.318 |

Raw results: [batch 128](benchmarks/cuda-a40-b128.json) and
[batch 512](benchmarks/cuda-a40-b512.json).

The distributed workload includes the encoder and full-batch contrastive loss.
Each timing below is the slower per-rank median for that interval; memory is
the maximum across ranks.

| Rows per rank | Global batch | Forward | Backward | Peak MiB per rank |
| ------------: | -----------: | ------: | -------: | ----------------: |
|            64 |          128 |   7.184 |    6.626 |             1.361 |
|           256 |          512 |   7.086 |    6.306 |            19.728 |

Raw results: [64 rows per rank](benchmarks/nccl-a40-b64.json) and
[256 rows per rank](benchmarks/nccl-a40-b256.json). Reproduce the eager runs
with the commands above, adding `--warmup 10 --repeats 50`, and set
`NCCL_P2P_DISABLE=1` on this host before running `torchrun`.

The [full-graph compiled proxy](benchmarks/compiled-proxy-a40-b128.json) at
batch 128 takes 0.475 ms forward and 1.118 ms backward, versus 1.228 ms and
2.322 ms eager. Peak additional memory is 0.174 MiB versus 0.331 MiB. Its first
forward/backward takes 27.607 seconds using fresh `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` directories. Compiler registration and import precede that
measurement. Compilation helps steady-state execution for this workload; its
startup cost and workload dependence still require measurement before adoption.

## Sampling and half-precision distances

The vectorized random triplet sampling and the float32 `cdist` path for
low-precision `LpDistance` were compared with the preceding branch revision
`83a9355` on the Apple M1 Pro described above, with one CPU thread, 128
dimensions, four examples per class, three warmup calls, and twenty measured
calls. Values are median milliseconds for one call.

| Case                                                 | Batch | Before | After |
| ---------------------------------------------------- | ----: | -----: | ----: |
| `get_random_triplet_indices(t_per_anchor=100)`       |   128 |  1.267 | 0.571 |
| `get_random_triplet_indices(t_per_anchor=100)`       |   512 |  5.242 | 3.356 |
| `DistanceWeightedMiner` (normalized, 128 dimensions) |   512 | 54.919 | 3.592 |
| `LpDistance` matrix, bfloat16 inputs                 |   512 | 47.047 | 0.303 |

The previous sampler looped over labels with host random draws; the previous
low-precision distance gathered every row pair, allocating memory proportional
to queries times references times dimensions. Both paths now stay on the
tensor's device with one host synchronization for the sampled count. Sampled
triplets follow the same distributions as before, but seeded sequences differ.
Reproduce the comparison with the public functions above and
`time.perf_counter`; CUDA timings for these paths are recorded with the release
acceptance run.

## Large batches and tuple memory

The number of valid pairs and triplets can grow rapidly with batch size and
label composition. Triplet enumeration allocates only valid triplets, but the
result can still exceed available memory. A PyTorch `nonzero` error mentioning
`INT_MAX` means an intermediate exceeds its indexing limit. Reduce batch size
or use a miner that selects a bounded subset of pairs or triplets.
