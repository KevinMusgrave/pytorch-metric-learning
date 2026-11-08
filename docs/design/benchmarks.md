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

The reused-distance cases compare recomputing a matrix in each module with passing
one `mat` to all of them. Interleave the cases of a comparison; running them in
sequence on a laptop produced a false 26% regression that five interleaved
rounds did not reproduce.

Short runs execute small real workloads and checks finite
losses, gradients, and valid miner indices before reporting measurements. It
does not assert machine-dependent timing thresholds. Longer measurements and
GPU runs are deliberate experiments, not an implicit part of ordinary tests.

## Release comparison

`benchmarks.compare` runs the exact released 2.9.0 and 3.0.0 source checkouts in
fresh subprocesses sharing one frozen Python environment. `release.json` fixes
the source commits and experimental matrix. The subprocess boundary prevents
imports, allocator history, and out-of-memory failures from leaking between
versions or batch sizes. The existing runner remains useful for development,
compilation, and distributed experiments.

The comparison covers seven common losses and the composition of batch-hard
mining with triplet loss. It explicitly translates ArcFace's degrees to radians
and transposes its old weight layout. Both versions receive identical CPU-seeded
embeddings and proxy weights; proxy class counts stay fixed as batch size grows.
Loss and full embedding/parameter gradients are compared before the performance
matrix. BF16 runs use BF16 inputs and parameters directly, without autocast;
nonfinite values are reported as failures rather than speedups.

Three fresh-process rounds interleave the versions for each shape. After warmup,
each process times three blocks of 100 forward/backward pairs with a fresh graph
and cleared gradients for every pair. Synchronization occurs at block boundaries,
not between the measured forwards and backwards. The primary time is the block's
wall time divided by 100. Preinitialized CUDA events record the forward/backward
breakdown, averaged across the same pairs. Event instrumentation and ordinary
dispatch/gradient-clearing costs remain in the wall time. CPU diagnostics use host
timers for that breakdown. All block averages are retained, and summaries report
the process means and their range. An out-of-memory result is never a timing.

Memory reporting includes baseline allocation, peak total allocated tensors,
peak incremental allocation, and peak allocator reservation. The batch search
uses the same full forward/backward workload with an explicit 90% allocator
budget, leaving 10% of device memory outside that budget. It doubles batches,
then refines the successful/failing bracket to 128 rows or 5%, whichever is
larger. The successful endpoint is confirmed in two additional fresh processes.
An OOM, indexing limit, nonfinite result, timeout, or configured search ceiling
remains a distinct outcome. This is loss-only capacity, not a claim about a
complete encoder, optimizer, or training pipeline.

A verified batch must be positive and pass both confirmation processes. If
every attempted batch fails, or either confirmation fails, report null rather
than an unverified size. A confirmation failure supplies the limiting status,
including when the initial search reached its ceiling. Refinement after an
unaligned successful batch advances to the next aligned candidate while the
bracket exceeds the requested resolution. Invalid bounds and resolution
settings fail before launching a worker.

The release report's memory comparison uses total PyTorch allocation. The
incremental metric subtracts the warmed pre-forward allocation, which includes
retained workspaces as well as inputs and parameters. For example, the RTX 5090
measurements retain about 64 MiB beyond inputs and parameters after warmup.
Calling that baseline only input/parameter storage would overstate the meaning
of incremental ratios.

A slow configuration can exceed the timing deadline while still fitting in
memory. Preserve that timeout in the primary matrix. A separate campaign can
use shorter blocks on both versions to obtain timings, with its own complete
configuration and raw records so different sampling protocols remain visible.
Timing-only exports still require the complete speed matrix. The primary report
continues to require its full gradient-comparison and capacity evidence.

Regression follow-ups preserve the released sources and pin a separate candidate
commit. `low-precision-fix.json` measures the affected Euclidean-distance losses
alongside NT-Xent and ProxyAnchor controls on two GPU types, using the same
900-pair protocol. Candidate results never replace the original release results.
The numerical tests cover difficult input distributions separately from the
standard-normal performance inputs; a faster result on those inputs is not a
claim about clustered embeddings or every execution mode.

Campaign files bind resume to the complete configuration, source revisions,
benchmark revision, and GPU identity. Completed raw records are appended after
each subprocess. A failed worker with an unexpected exception stops the campaign
instead of being labeled as insufficient memory. The benchmark boundary suite
executes real small workloads and checks matched inputs, resume isolation,
capacity bracketing, and failure classification without timing thresholds.
