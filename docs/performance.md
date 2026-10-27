# Measuring performance

Run benchmarks from the repository's frozen development environment. These
measurements establish a reference for changes; they do not impose a universal
speed requirement across machines.

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
[batch 512](benchmarks/cpu-m1-pro-b512.json). The large triplet forward cost is
a concrete optimization target: the current enumeration constructs a cubic
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
default. CUDA performance and peak tensor memory remain unmeasured.
