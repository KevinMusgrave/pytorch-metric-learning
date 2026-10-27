# Performance measurements

The benchmark runner exercises public loss, miner, and distributed APIs. It is
development tooling, separate from the installable library. It depends on the
numerical and distributed modules; neither module imports the runner.

Measure autograd-enabled forward and backward separately on fresh graphs. Clear
gradients before each step, warm up before sampling, and synchronize CUDA at
timing boundaries. Report the first step separately so compilation and startup
costs do not disappear into steady-state timings. Compiled runs require a full
graph and fail on unsupported operations rather than silently timing an eager
fallback. No universal performance threshold is imposed.

Record raw samples, medians, hardware, software, revision, dirty-worktree state,
dtype, shapes, label distribution, seeds, threads, and execution mode. CUDA peak
memory is the incremental allocated tensor memory above the pre-forward
baseline. CPU peak tensor memory is unavailable through that API and is reported
as null. Allocator reservation, other processes, and model/input storage are not
that incremental metric.

The distributed case includes a linear encoder, the full-batch contrastive
wrapper, and DDP gradient synchronization. Each rank reports its own measurements;
aggregate comparisons must use the slowest rank. Barriers align samples outside
the timing window. Run this case with torchrun; correctness tests launch their
own workers and remain a separate command.

The quick boundary command executes small real workloads and checks finite
losses, gradients, and valid miner indices before reporting measurements. It
does not assert machine-dependent timing thresholds. Longer measurements and
GPU runs are deliberate experiments, not an implicit part of ordinary tests.
