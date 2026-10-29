# Numerical components

Losses, miners, distances, reducers, and regularizers share tensor conventions
and pair/triplet conversion. These mutually dependent
building blocks form one policy boundary. They must not depend on training,
dataset downloading, retrieval backends, or distributed process management.

Embeddings have shape `(batch, dimensions)`. Label equality defines positive
pairs; unequal labels define negative pairs. A pair tuple contains four index
tensors `(anchor_positive, positive, anchor_negative, negative)`, while a triplet
tuple contains `(anchor, positive, negative)`. Reference embeddings may have a
different batch size. Excluding self-comparisons depends on whether the query
and reference sets are the same set, not merely on equal shapes.

All-triplet enumeration expands each actual positive pair by its anchor's
negative count and indexes the flattened negative rows through prefix offsets.
Temporary storage scales with the query/reference pair matrices and emitted
triplets, rather than an unconditional `(queries, references, references)` mask.
The output remains ordered by anchor, positive, then negative, uses int64 indices
on the labels' device, and preserves the distinction between the same label
tensor and an independent reference set. Empty valid sets produce empty index
tensors. Large valid triplet sets still require memory proportional to their
output; enumeration does not make materializing every triplet free.

Tuple-frequency weights divide each example's occurrence count by the maximum
count. With separate references, only anchor indices address query weights.
Weighted random triplet sampling uses float32 probabilities for float16 inputs;
the original precision choice followed [PyTorch's multinomial CUDA issue](https://github.com/pytorch/pytorch/issues/19900).

Losses compose explicitly. `BaseMetricLossFunction`, `BaseMiner`, and
`BaseRegularizer` extend `torch.nn.Module` directly and declare their options as
keyword-only parameters; concrete classes forward `**options` typed by
`types.LossOptions` or `types.RegularizerOptions` so IDEs and the type checker
see the full signature without cooperative multiple inheritance. A loss's
reducer reduces every term it computes, named by the `sub_loss_names` class
attribute, and returns one tensor. The reducer object is used as given: copying
it hid shared state and made `loss.reducer` a different object from the one the
caller held. `BaseReducer.forward` sums the per-term reductions, so a single
reducer serves multi-term losses and `MultipleReducers` is the only mechanism
for per-term choices. Embedding and weight regularization are constructor
arguments, not mixins discovered by walking the class hierarchy or matching
class names. Their weighted, already reduced penalties are added after
reduction; `loss_terms()` returns the unreduced objective terms together with
those penalties for callers who want their own aggregation. Weight-bearing
losses declare `weight_regularizer`, `weight_reg_weight`, and
`weight_init_func` as keyword-only arguments and store their class vectors in a
parameter named `weight` with one row per class or center, the `nn.Linear`
layout, so regularizers, initializers, and documentation address one shape.
`ArcFaceLoss` takes its margin in radians because the paper does, and
`LargeMarginSoftmaxLoss` rejects non-integer margins because the binomial
expansion is defined for integers only.

The public types in `pytorch_metric_learning.types` describe this protocol
without wrapping its dictionaries or tensors. `LossDict` maps sub-loss names to
`LossTerm`, whose `losses` entry is always a tensor: an empty tensor reduces to
a differentiable zero, which removed the literal-zero sentinel every reducer had
to special-case. `IndicesTuple` distinguishes the existing three-tensor triplets
from four-tensor pairs. Loss computation and mining expose these annotations,
and reducer inputs share the same term type. Every reducer returns a tensor.
The distribution includes `py.typed`, and the policy check type-checks the
package, tests, and recipes, so the annotations are verified rather than
decorative. `GenericPairLoss` exposes one hook, `pair_loss(mat, pairs)`, with
`pair_values` and `pair_masks` helpers; the former constructor flag selecting
between two private hooks made the contract depend on construction state.
`CrossBatchMemory` and `SelfSupervisedLoss` accept any `BaseMetricLossFunction`
and rely on the wrapped loss's own `ref_emb` validation instead of an allowlist
of classes, so custom losses compose without registration.

Public index utilities live in `utils.indexing`; everything else under `utils`
with a leading underscore is private. Random triplet sampling draws positives
and negatives with row-wise `torch.multinomial` over the match and difference
masks, with one host synchronization for the maximum count, instead of a Python
loop over labels with host random draws.

Validation uses explicit exceptions and remains active under `python -O`.
Invalid argument types raise `TypeError`; invalid values, shapes, tuple lengths,
and reduction names raise `ValueError`. Miners validate output before handing
indices to losses, and memory validates enqueue masks before modifying its
queue. Distance orientation is a class property: similarities declare
`is_inverted = True`, while distance classes inherit `False`. The normalization
norm `p` still applies to similarities that normalize embeddings.

The version 2-to-3 migration guide indexes removed exports, moved utilities,
and changed calls beside their replacements. It links to the canonical API
contracts and recipes instead of introducing compatibility code. Its standalone
Python examples include observable value, shape, or gradient checks and run in
the numerical boundary suite; signature declarations remain covered separately.

Learnable loss parameters follow ordinary PyTorch module placement. A forward
pass never moves, casts, or reallocates parameters to match the embeddings.
Doing so changed `state_dict` dtypes and broke optimizer state after a single
mixed-precision call. Outside autocast, a placement mismatch raises
`ValueError`; under autocast, parameters may keep higher precision. Constant
coefficient tensors are non-persistent buffers so `.to()` moves them without
changing checkpoint contents. P2SGrad still renormalizes its weight columns in
place each forward, as its algorithm specifies, under `torch.no_grad()`.
Each computation uses a differentiable clone of the normalized parameter, so
another forward cannot mutate the weights saved for its backward pass.
Parameter identity and storage remain stable; `.data` never bypasses autograd.

Tensor conversion uses explicit `Tensor.to`, `torch.as_tensor`, and
`Tensor.cpu().numpy()` calls. Device-only transfers preserve label dtypes;
explicit dtype requests also apply during autocast. The removed conversion
helpers silently ignored dtype requests under CUDA autocast, making placement
and accumulation depend on hidden global state.
Histogram pair deduplication returns tensor-index tuples on the input device,
avoiding Python-list conversion, host synchronization, and deprecated list
indexing.

Memory and self-supervision wrappers check supported loss types with
`isinstance`, so subclasses that preserve those contracts remain usable.
`supported_losses()` returns the actual types. ProxyNCA is explicitly excluded,
including subclasses: although it derives from NCA, its references are learned
proxies and it rejects the separate references required by these wrappers.
The wrappers directly inherit `torch.nn.Module`; no name-dispatch base is needed.

ProxyNCA converts embeddings to its proxies' dtype before normalization and
distance computation, in both the loss and `get_logits`. This keeps distance
inputs compatible under float16 and bfloat16 autocast without changing the
parameter or optimizer state. The conversion remains part of autograd so
encoder gradients pass through it. Outside autocast, the placement check still
requires the caller to match the module's device and dtype.

Distances declare whether larger values mean greater similarity. Losses and
miners use that declaration when comparing margins. Reducers own averaging and
filtering; losses retain the indices and reduction type needed by the reducer.
An empty valid set must yield a differentiable zero without introducing NaNs.

Cross-batch memory stores detached reference embeddings. Its queue buffers follow module
placement like learnable parameters: forward checks, rather than changes, their
device and dtype, and labels are cast only to the configured `label_dtype`. Each forward uses a
snapshot of the occupied embeddings and labels so subsequent queue writes cannot
invalidate tensors saved for backward. This supports accumulating several
forward losses before backpropagating. Only the occupied portion is copied while
the queue is warming up. An empty enqueue leaves the queue and its filled state
unchanged; an empty query yields a differentiable zero. Self-comparison removal
with no current queue indices removes nothing. A call cannot enqueue more rows
than the queue capacity, including when an enqueue mask is supplied.
Under autocast, the incoming batch is converted to the queue's dtype before
mining and loss computation. The conversion preserves anchor gradients; only
enqueued values are detached. Converting the batch once keeps anchors and
references compatible with the wrapped distance without casting the larger
queue or changing its configured precision.
The queue cursor and filled flag are saved under a `queue_state` entry by an
explicitly registered state-dict hook, so a `state_dict` checkpoint restores
the next loss and gradients, not just the raw buffers. A checkpoint without
that entry reports it as a missing key. The checkpoint contains tensors and primitive metadata and supports
`torch.load(..., weights_only=True)`. Older checkpoints without that metadata
do not describe a resumable queue and are not silently interpreted as one.

Dynamic soft margin weights distance differences by a running cumulative
histogram. Histogram updates are detached statistics, not part of the gradient
graph. Normalize each batch histogram before applying momentum so a larger
triplet count does not silently override that parameter. Bin centers include
both endpoints of `[min_val, -min_val]`; out-of-range observations saturate their
weights at the nearest endpoint without clipping the loss's distance difference.
The histogram is a registered buffer and survives checkpointing and device moves.
Bin coordinates and contributions use the buffer's dtype, including under
autocast. Both histogram interpolation and cumulative-weight lookup use these
coordinates so rounding near a bin boundary cannot select different bins.
Matched-pair mining masks a separate distance tensor and never mutates the
distance output saved by autograd. Unequal streams contribute only real matched
diagonal pairs; their extra rows remain eligible negatives. Empty valid sets
leave histogram state unchanged.

This contract follows the histogram and detached-weight algorithm from Zhang
et al., [Learning Local Descriptors With a CDF-Based Dynamic Soft Margin](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Local_Descriptors_With_a_CDF-Based_Dynamic_Soft_Margin_ICCV_2019_paper.pdf),
and the [authors' implementation](https://github.com/lg-zhang/dynamic-soft-margin-pytorch).
Domenico Muscillo contributed the original PML implementation. Independent tests
use triangular-basis probabilities and explicit distance formulas to verify
values, gradients, and sequential updates. Float16 histogram and gradient comparisons allow
`5e-4` absolute and `5e-3` relative error for rounding during normalization and
accumulation; float32 and float64 retain PyTorch's default comparison tolerances.

Some implementation choices are numerical rather than mathematical.
`LpDistance` computes float16 and bfloat16 matrices with `torch.cdist` in
float32 and casts the result, because `torch.cdist` lacks CPU kernels for those
types and expanding every row pair costs memory proportional to queries times
references times dimensions. The call fixes `compute_mode` to the direct
difference kernel: the default switches to $\|x\|^2 + \|y\|^2 - 2 x \cdot y$
above 25 rows, and that formula cancels to exactly zero for nearby rows whose
half-precision difference is representable, erasing their gradients. The direct
kernel costs more time on CPU than the matrix product, but it preserves every
representable distance and keeps peak memory at the output size. The cast is
differentiable, so the input dtype and autograd are preserved outside autocast.
Tests compare values and both input gradients against explicit float64 norms
for normalized and unnormalized embeddings, L1 and L2 distances, and powers 1
and 2, and check rows differing by one machine epsilon on both sides of the
25-row threshold. Comparisons allow four machine epsilons of relative error and
one of absolute error for low-precision rounding. Default reductions avoid data-dependent shapes: `ThresholdReducer`
masks and counts instead of indexing, and `DivisorReducer` selects with
`torch.where`, so the common loss paths neither synchronize with the host nor
break full-graph compilation.
`ManifoldLoss` inverts its random-walk matrix in float32,
which half precision lacks, and masks excluded terms with $-\infty$ before
log-sum-exp rather than subtracting margins, avoiding cancellation; a sample's
correlation with its own proxy contributes no loss. `DistanceWeightedMiner`
clamps distances at `cutoff` to bound weight variance and works in log space,
following the first equation of Section 4 of Wu et al. It masks same-class
pairs, pairs at or beyond `nonzero_loss_cutoff`, and non-finite weights before
subtracting each anchor's maximum log weight, so the clamped self-pair cannot
dominate the normalizer and underflow every eligible negative, and anchors
whose negatives lie at different distance scales each keep a nonzero row. `SupConLoss`
subtracts each row's maximum before exponentiation.

The mathematical definitions, supported arguments, and paper references remain
in `docs/losses.md`, `docs/miners.md`, `docs/distances.md`, `docs/reducers.md`,
and `docs/regularizers.md`. In particular, P2SGrad computes mean squared error
between cosine scores and one-hot labels, which gives the paper's specified
gradient. Reference implementations in the tests compare values and gradients;
their upstream provenance is recorded in `THIRD_PARTY_NOTICES.md`.

The development baseline uses current PyTorch tensor operations. Compatibility
with obsolete PyTorch releases is not a requirement.

Trainer-specific key checking, loss tracking, generic model checkpointing,
parameter freezing, SQLite conversion, and batch-unpacking helpers have been
removed with the orchestration layer. Retained numerical utilities do not own
model lifecycles or depend on logging integrations.

Modules do not record diagnostic statistics. Forward passes return tensors and
leave no side-effect attributes, which avoids host synchronization and keeps
modules compatible with graph capture. Diagnostics such as mined-pair counts or
mean pair distances are computed by callers from miner outputs and distance
matrices, or observed with PyTorch forward hooks.
