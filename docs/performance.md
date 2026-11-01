# Measuring performance

The [GPU release comparison](benchmarks/gpu-3.0.0/README.md) measures released
v3.0.0 against v2.9.0 on RTX 4090, RTX 5090, L40S, A100, and H100 systems
on 2026-11-01 UTC.
It reports absolute v3 forward/backward time, peak memory, verified batch
capacity, and version ratios for seven common losses plus batch-hard triplet
mining. [Full measurements](benchmarks/gpu-3.0.0/measurements.csv) retain both
versions, all shapes, BF16, failures, and process ranges.
The [large-triplet follow-up](benchmarks/gpu-3.0.0/large-triplet/README.md)
uses shorter timing blocks where the main protocol exceeds its time limit.

For release announcements, NT-Xent provides the clearest improvement: at batch
512, 128 dimensions, four samples per class, and float32, v3 is **2.28–33.5×
faster** and reduces peak PyTorch allocation by a factor of **79.2–186** across
these five systems. On the RTX 5090, that is 0.733 ms per forward/backward pair and 82.7 MiB
peak allocation. Under the 90% memory budget, its verified NT-Xent batch limit
increases from 768 to 19,968; the H100 increases from 1,024 to 31,744.

These are loss-only results. Other losses have smaller gains, overlapping timing
ranges, or regressions; the report includes all of them. The figures do not
describe end-to-end training speed or establish one GPU as universally faster
than another rented system.

Released v3.0.0 also has major BF16 regressions at batch 2,048: batch-hard triplet
is 2.62–17.2× slower than v2.9.0, and Contrastive is 1.90–5.30× slower across these
systems. Their peak allocations increase by factors of 13.5–19.5 and 6.12–7.09,
respectively. The [low-precision distance fix](#low-precision-distance-fix)
documents an unreleased correction separately from the release measurements.

## Development measurements

Run benchmarks from the repository's frozen development environment. These
measurements establish a reference for changes; they do not impose a universal
speed requirement across machines. The runner imports the library from the
repository's `src` directory.

```sh
uv run --frozen --no-sync python -m benchmarks.run \
  --batch-size 128 --dimensions 128 --repeats 20 --warmup 5 \
  --output /tmp/pml-cpu.json
```

The default run measures contrastive loss, all-triplet margin loss, NT-Xent
loss, pair-margin mining, pair-margin mining followed by triplet margin loss
(`mined-triplet`, which converts mined pairs to triplets), and
normalized-softmax proxy loss. Inputs are standard-normal tensors with
balanced contiguous labels, four examples per class unless
`--samples-per-class` says otherwise. Distances normalize embeddings inside the
measured call. The contrastive negative margin is 1.4, triplet margin is 0.2,
NT-Xent temperature is 0.07, miner margins are 0.2 and 1.4, and proxy
temperature is 0.1. The proxy benchmark includes gradients for its learnable
weights.

The `separate-cosine`, `reused-cosine`, `separate-lp`, and `reused-lp` cases run
`MultiSimilarityMiner`, `MultiSimilarityLoss`, and `ContrastiveLoss` on one batch.
A `separate` case computes one distance matrix per module. A `reused` case
computes one matrix with gradients first and passes it to all three as
[`mat`](losses.md#precomputed-distance-matrices). Compare each pair to measure what
repeated distance computation costs; their losses are equal for the same seed.
These cases include no encoder, so they overstate the saving on a whole training
step.

The `full-ntxent` and `anchor-batched-ntxent` cases, and the matching triplet pair,
compare a full-batch `backward()` with
[`anchor_batched_backward`](losses/anchor_batched_backward.md) in chunks of
`--anchor-batch-size` anchors. An anchor-batched case does all of its work in
`forward_ms`, so compare totals, and its backward time is null. The
`unbounded-mined-triplet` and `bounded-mined-triplet` cases compare a
`TripletMarginMiner` with and without
[`tuples_per_anchor=32`](miners.md#bounding-the-output). Peak memory for all of these
needs `--device cuda`.

The default run includes every case except `distributed`. Timings of a few milliseconds
vary between processes: the same case differed by up to 122% across four 20-repeat
runs on a rented GPU, while peak memory was identical. Compare timings from several
processes with `--repeats 100` or more, and treat a difference as real only when the
ranges of the two cases do not overlap.

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

## Systematic release comparison

`benchmarks/release.json` defines a matched comparison of the exact released
v2.9.0 and v3.0.0 commits. The matrix covers NT-Xent, Contrastive, all-triplet
margin, batch-hard mining plus triplet margin, ArcFace, SupCon, MultiSimilarity,
and ProxyAnchor. Batch sizes are 128, 512, 2,048, and 8,192. The standard profile
uses 128-dimensional embeddings and four samples per class; the wide/dense
profile uses 512 dimensions and sixteen per class. Proxy losses instead use
cyclic labels and fixed class counts of 1,024 and 10,000 respectively. Those
class counts do not grow with the batch.

Both releases run in separate processes in the same frozen environment on each
GPU. The inputs are CPU-seeded standard-normal embeddings. Proxy weights are
identical, with ArcFace's old weight layout transposed and its old degree-based
margin translated to the same 0.5 radians. Other settings are explicit in
`benchmarks/measurement.py`. Before timing, each loss's scalar value and full
embedding/parameter gradients must match at float32 (`rtol=2e-4`, `atol=2e-6`).
No model, optimizer, or data-loading work is included.
The matrix uses ordinary full loss calls and backward passes. Anchor batching,
compilation, and distributed execution remain separate experiments.

Timing uses three independent process rounds with interleaved version order.
Each process warms up for at least ten pairs and one second, then measures
three blocks of 100 forward/backward pairs. Each pair builds a fresh graph and
clears gradients. CUDA synchronizes only at block boundaries. The primary time
is elapsed block wall time divided by 100; CUDA events provide averaged
forward/backward breakdowns. The event objects are initialized before timing.
Normal dispatch, event-recording, and gradient-clearing overhead remains in the
wall time. Report the mean of process means and their range; overlapping ranges
are not evidence of a reliable speed difference.

Peak total PyTorch allocation, incremental peak above the warmed pre-forward
allocation, and allocator reservation are distinct metrics. The warmed baseline
contains inputs, parameters, and any retained workspaces. The main report uses
total allocated memory for memory ratios; incremental allocation excludes the
warmed baseline and is useful for understanding additional loss costs. Neither
metric includes all device usage, such as CUDA context memory. Capacity probes use
90% of device memory as the tensor allocator budget, leaving 10% outside it.
They double batches and refine a success/failure bracket to 128 rows or 5%,
whichever is larger. Each successful endpoint is confirmed in two more fresh
processes, each with a first step, one warmup, and three measured steps. OOM,
indexing limits, nonfinite results, a five-minute subprocess timeout, and the
1,048,576-row search ceiling remain distinct outcomes. A search ceiling is a
lower bound, never a measured hardware limit. Actual encoder/optimizer training
needs additional memory beyond these loss-only limits.

Float32 is the main comparison. BF16 inputs and parameters are also measured at
batches 512 and 2,048 in the standard profile, without autocast. Finite losses
and gradients are required, but the full cross-version gradient equivalence
check is float32. BF16 results do not establish mixed-precision training quality.

Run from a clean, committed benchmark checkout:

```sh
uv sync --frozen --no-build --no-install-project --all-extras --python 3.12.13

git worktree add --detach /tmp/pml-v2 v2.9.0
git worktree add --detach /tmp/pml-v3 v3.0.0
uv run --frozen --no-sync python -m benchmarks.compare \
  --source-2 /tmp/pml-v2 --source-3 /tmp/pml-v3 \
  --output /tmp/pml-gpu-results
```

Use `--resume` with the same output directory after an interruption. Resume
requires the same complete configuration, source and benchmark revisions, and
GPU identity. Completed records are retained; unexpected worker errors stop the
campaign for investigation. The source revisions are checked against the JSON
matrix and dirty library source is rejected. The existing runner above remains
available for distributed, compilation, and other development experiments.

After collecting completed campaign directories from each GPU:

```sh
uv run --frozen --no-sync python -m benchmarks.report \
  /tmp/rtx4090 /tmp/rtx5090 /tmp/l40s /tmp/a100 /tmp/h100 \
  --output docs/benchmarks/gpu-3.0.0
```

The report generator requires complete timing, correctness, and capacity
records. It produces Markdown tables, CSV measurements, standalone SVG scaling
charts, compressed raw block measurements, and hardware/source provenance.
GPU and host CPU both affect the result, especially on small eager workloads;
measure the target training setup before setting a performance budget.

Large all-triplet batches can fit in memory but exceed the five-minute limit
when timing 300 pairs per process. The separate `benchmarks/large-triplet.json`
experiment measures batch 8,192 using three blocks of ten pairs in each of three
fresh processes, with the same warmup and input settings. It preserves the
primary experiment's timeouts and gives the shorter protocol its own records:

```sh
uv run --frozen --no-sync python -m benchmarks.compare \
  --config benchmarks/large-triplet.json --phases speed \
  --source-2 /tmp/pml-v2 --source-3 /tmp/pml-v3 \
  --output /tmp/pml-large-triplet

uv run --frozen --no-sync python -m benchmarks.report \
  /tmp/pml-large-triplet --timing-only \
  --output /tmp/pml-large-triplet-report
```

`--timing-only` requires the complete timing matrix and exports the block count,
process ranges, memory, CSV, and raw evidence. Full-gradient agreement and batch
capacity remain separate measurements in the primary campaign.

## Low-precision distance fix

The follow-up compares v2.9.0, released v3.0.0, and candidate
`32bcfa8213a9a48d95153a07a6cf6e0e753e1a18` on an A100 and another RTX 5090.
The candidate is not part of the released v3.0.0 measurements above.
The [complete timings](benchmarks/gpu-3.0.0/low-precision-fix/README.md)
include NT-Xent and ProxyAnchor controls, float32 batches 128, 512, and 2,048,
and BF16 batches 512 and 2,048. Every successful configuration again averages
900 forward/backward pairs across three fresh processes.

At BF16 batch 2,048, 128 dimensions, and four samples per class:

| GPU      | Loss                 | v2.9 ms | v3.0 ms | Candidate ms | v3.0 → candidate peak MiB |
| -------- | -------------------- | ------: | ------: | -----------: | ------------------------: |
| RTX 5090 | Contrastive          |    2.56 |    13.3 |         2.23 |               2,118 → 356 |
| RTX 5090 | Batch-hard + Triplet |    1.10 |    16.4 |         1.38 |               2,118 → 151 |
| A100     | Contrastive          |    7.63 |    23.4 |         6.32 |               2,118 → 308 |
| A100     | Batch-hard + Triplet |    4.93 |    30.0 |         5.47 |               2,118 → 103 |

The candidate removes the severe regression on these workloads. It does not
win every comparison: RTX 5090 batch-hard triplet remains 25.6% slower than
v2.9.0, while the A100 process ranges overlap (4.64–5.36 ms for v2.9.0 and
4.41–6.54 ms for the candidate). Contrastive uses about 3% more peak allocation
than v2.9.0. The A100 controls did not reproduce the original 53% ProxyAnchor
slowdown at float32 batch 2,048: the new release comparison measured 16%, and
the candidate's range overlaps v2.9.0. Small NT-Xent timings also overlap on
that system; their original mean difference is not a stable regression estimate.

The correction avoids the direct Euclidean-distance backward kernel's
batch-by-batch-by-dimension allocation when float32 matrix evaluation is
numerically safe. It screens distances for cancellation and retains the direct
calculation for nearby or duplicate rows, other norms, small matrices,
compilation, CUDA graph capture, and reduced float32 matmul precision.
These measurements use standard-normal inputs and highest float32 matmul
precision; clustered training embeddings can take the slower direct path.

Values and full gradients were checked against direct-difference references
for float16 and BF16 random, nearby, duplicate, and mixed-scale inputs. GPU
tests also cover CUDA graph capture and reduced matmul precision. The four
follow-up losses passed float32 scalar and full-gradient comparison between
released v3.0.0 and the candidate before timing.
The [A100 validation](benchmarks/gpu-3.0.0/low-precision-fix/gpu-1-validation.json)
and [RTX 5090 validation](benchmarks/gpu-3.0.0/low-precision-fix/gpu-2-validation.json)
record the errors and archive checks. An additional
[BF16 comparison at batch 2,048](benchmarks/gpu-3.0.0/low-precision-fix/bf16-gradient-validation.json)
on the A100 found unchanged Contrastive and batch-hard triplet loss values, with
maximum embedding-gradient absolute differences of `5.96e-8` and `2.38e-7`.

`benchmarks/low-precision-fix.json` pins all three source commits. To reproduce
the timing matrix, create a clean worktree for each source and run the generic
campaign API from a clean, committed benchmark checkout:

```sh
git worktree add --detach /tmp/pml-fix 32bcfa8213a9a48d95153a07a6cf6e0e753e1a18
uv run --frozen --no-sync python - <<'PY'
import json
from pathlib import Path
from benchmarks.compare import Campaign, speed_plan

config = json.loads(Path("benchmarks/low-precision-fix.json").read_text())
sources = dict(zip(config["versions"], map(Path, (
    "/tmp/pml-v2", "/tmp/pml-v3", "/tmp/pml-fix"
)), strict=True))
campaign = Campaign(config, sources, Path("/tmp/pml-fix-results"), "cuda", True)
for request in speed_plan(config):
    campaign.run(request)
PY

uv run --frozen --no-sync python -m benchmarks.report \
  /tmp/pml-fix-results --timing-only --output /tmp/pml-fix-report
```

## Pre-release development reference (2026-10-29)

These historical measurements precede numerical changes in the final v3.0.0
release. They document development progress and should not be used as final
release performance claims.

Measured on 2026-10-29 UTC with Python 3.12.13, PyTorch 2.14.0, float32, 128
dimensions, seed 17, and one CPU thread per rank. CPU rows ran on an Apple M1
Pro with ten warmup and fifty measured steps. CUDA rows ran on NVIDIA A40s with
CUDA 13.0. Values are median milliseconds; peak is the additional allocated
CUDA memory in MiB. Neither machine was isolated for benchmarking: use the raw
sample spread and repeat on the target hardware before setting a budget.

| Case          | M1 Pro b128 fwd / bwd | M1 Pro b512 fwd / bwd | A40 b128 fwd / bwd | A40 b512 fwd / bwd | A40 b512 peak |
| ------------- | --------------------: | --------------------: | -----------------: | -----------------: | ------------: |
| Contrastive   |         0.291 / 0.196 |         2.888 / 1.473 |      2.026 / 2.901 |      1.994 / 2.439 |          19.5 |
| All-triplet   |         1.197 / 0.446 |        19.489 / 5.879 |      3.810 / 3.241 |      3.841 / 3.312 |          63.2 |
| NT-Xent       |         0.463 / 0.240 |         6.131 / 1.984 |      2.974 / 3.867 |      2.973 / 3.888 |          18.5 |
| Mined triplet |         0.909 / 0.265 |        12.285 / 2.808 |      5.606 / 3.249 |      4.562 / 3.261 |          27.6 |
| Pair miner    |             0.361 / — |             4.635 / — |          0.857 / — |          0.832 / — |           7.7 |
| Proxy         |         0.073 / 0.136 |         0.184 / 0.342 |      1.538 / 3.152 |      1.517 / 3.142 |           1.6 |

Raw results: M1 Pro [batch 128](benchmarks/cpu-m1-pro-v3-b128.json) and
[batch 512](benchmarks/cpu-m1-pro-v3-b512.json); A40
[batch 128](benchmarks/cuda-a40-v3-b128.json) and
[batch 512](benchmarks/cuda-a40-v3-b512.json). The A40 timings at these sizes
are dominated by kernel launches, so they barely change with batch size.

The full-graph compiled proxy at batch 128 took 0.064 ms forward and 0.067 ms
backward on the M1 Pro and 0.331 ms and 0.576 ms on the A40, after first steps
of 3.3 s and 9.9 s
([M1 Pro](benchmarks/compiled-proxy-m1-pro-v3-b128.json),
[A40](benchmarks/compiled-proxy-a40-v3-b128.json)). This workload does not
justify compiling by default.

The distributed case, a linear encoder with the global-objective contrastive
loss and DDP, took 1.315 ms forward and 0.980 ms backward on the slower of two
Gloo ranks on the M1 Pro with 64 rows per rank
([raw](benchmarks/gloo-m1-pro-v3-b64.json)). With two A40s and NCCL
with peer-to-peer transport disabled (`NCCL_P2P_DISABLE=1`), the slower rank took 5.360 ms and
4.849 ms with 64 rows per rank and 5.104 ms and 4.623 ms with 256
([64 rows](benchmarks/nccl-a40-v3-b64.json),
[256 rows](benchmarks/nccl-a40-v3-b256.json)). Disabling peer-to-peer
transport means these numbers do not measure direct GPU-to-GPU transfers.

### Comparison with version 2.9.0

The same runner measured the v2.9.0 release on the same machines. Losses agree
to float32 rounding for every case except the proxy loss, whose weights are
initialized with a different layout. Composition paths changed the most:
version 2 compared every positive pair with every negative pair when it turned
mined pairs into triplets and when it computed NT-Xent, while version 3 groups
negatives by anchor. The table uses 512 samples with sixteen per class, where
those Cartesian intermediates are largest.

| Case, 512 × 16 per class  | M1 Pro v2.9.0 fwd / bwd | M1 Pro v3 fwd / bwd | A40 v2.9.0 fwd / bwd | A40 v3 fwd / bwd | A40 peak v2.9.0 → v3 |
| ------------------------- | ----------------------: | ------------------: | -------------------: | ---------------: | -------------------: |
| NT-Xent                   |      15,202 / 15,781 ms |    6.149 / 2.046 ms |     185.5 / 143.7 ms | 1.692 / 1.782 ms |    31,627 → 18.4 MiB |
| Pair miner → triplet loss |         2,262 / 16.6 ms |    30.91 / 10.99 ms |      11.69 / 4.65 ms | 2.521 / 3.106 ms |        801 → 123 MiB |

With four samples per class, all-triplet enumeration at batch 512 took 331.6 ms
forward on the M1 Pro in v2.9.0 and 19.5 ms in v3, and its A40 peak fell from
148 MiB to 63 MiB; NT-Xent's A40 peak fell from 6,485 MiB to 18.5 MiB. Raw
results: v2.9.0 on the M1 Pro [batch 128](benchmarks/cpu-m1-pro-v2-b128.json),
[batch 512](benchmarks/cpu-m1-pro-v2-b512.json),
[NT-Xent](benchmarks/cpu-m1-pro-v2-ntxent-b512-spc16.json), and
[mined triplet](benchmarks/cpu-m1-pro-v2-mined-triplet-b512-spc16.json); v3 on
the M1 Pro [NT-Xent](benchmarks/cpu-m1-pro-v3-ntxent-b512-spc16.json) and
[mined triplet](benchmarks/cpu-m1-pro-v3-mined-triplet-b512-spc16.json);
v2.9.0 on the A40 [batch 128](benchmarks/cuda-a40-v2-b128.json),
[batch 512](benchmarks/cuda-a40-v2-b512.json),
[NT-Xent](benchmarks/cuda-a40-v2-ntxent-b512-spc16.json), and
[mined triplet](benchmarks/cuda-a40-v2-mined-triplet-b512-spc16.json); v3 on the
A40 [NT-Xent](benchmarks/cuda-a40-v3-ntxent-b512-spc16.json) and
[mined triplet](benchmarks/cuda-a40-v3-mined-triplet-b512-spc16.json).

## Large batches and tuple memory

The number of valid pairs and triplets can grow rapidly with batch size and
label composition. Triplet enumeration allocates only valid triplets, but the
result can still exceed available memory. A PyTorch `nonzero` error mentioning
`INT_MAX` means an intermediate exceeds its indexing limit. Reduce batch size
or use a miner that selects a bounded subset of pairs or triplets.
