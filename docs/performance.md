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

## Version 3.0.0 reference

Measured on 2026-10-29 UTC with Python 3.12.13, PyTorch 2.14.0, float32, 128
dimensions, seed 17, and one CPU thread per rank. CPU rows ran on an Apple M1
Pro at revision `e25aba5` with ten warmup and fifty measured steps. CUDA rows
ran on a rented host with two NVIDIA A40s (driver 595.91.07, CUDA 13.0, NCCL
2.30.7) at revision `01bf1da`, whose library source is identical to `e25aba5`.
Values are median milliseconds; peak is the additional allocated CUDA memory in
MiB. These are a shared laptop and a rented host, not isolated performance labs:
use the raw sample spread and repeat on the target hardware before setting a
budget.

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
(`NCCL_P2P_DISABLE=1`, required on that host), the slower rank took 5.360 ms and
4.849 ms with 64 rows per rank and 5.104 ms and 4.623 ms with 256
([64 rows](benchmarks/nccl-a40-v3-b64.json),
[256 rows](benchmarks/nccl-a40-v3-b256.json)). Disabling peer-to-peer
transport means these numbers do not measure direct GPU-to-GPU transfers.

## Comparison with version 2.9.0

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
[mined triplet](benchmarks/cuda-a40-v3-mined-triplet-b512-spc16.json). The
v2.9.0 reports record the release commit `bee27da` as dirty because the
version 3 runner was copied into that checkout; v2.9.0's import of tqdm was
satisfied by a no-op stub, which the measured code does not call.

## Large batches and tuple memory

The number of valid pairs and triplets can grow rapidly with batch size and
label composition. Triplet enumeration allocates only valid triplets, but the
result can still exceed available memory. A PyTorch `nonzero` error mentioning
`INT_MAX` means an intermediate exceeds its indexing limit. Reduce batch size
or use a miner that selects a bounded subset of pairs or triplets.
