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
including compilation and startup, is reported separately. The default five
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
