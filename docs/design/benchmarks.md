# Performance measurements

The benchmark runner exercises public loss, miner, and distributed APIs. It is
development tooling, separate from the installable library. It depends on the
numerical and distributed modules; neither module imports the runner.
The runner imports the sibling `src` directory directly, so measurements use
that source even when another version is installed, and the runner needs no
project installation.

Measure autograd-enabled forward and backward separately on fresh graphs. Clear
gradients before each step, warm up before sampling, and synchronize CUDA at
timing boundaries. Report the first step separately so lazy graph compilation
does not disappear into steady-state timings. Module construction, compiler
registration, and process startup precede that interval. Compiled runs require a full
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

The reused-distance cases measure one proposal for #211 without adding public
API: a benchmark-only subclass of a distance returns a matrix computed once, with
gradients, before mining. Interleave the cases of a comparison; running them in
sequence on a laptop produced a false 26% regression that five interleaved
rounds did not reproduce.

Short runs execute small real workloads and checks finite
losses, gradients, and valid miner indices before reporting measurements. It
does not assert machine-dependent timing thresholds. Longer measurements and
GPU runs are deliberate experiments, not an implicit part of ordinary tests.
