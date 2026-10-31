# Numerical components

Losses, miners, distances, reducers, and regularizers share tensor conventions
and pair/triplet conversion. These mutually dependent building blocks form one
policy boundary. They must not depend on training, dataset downloading,
retrieval backends, or distributed process management.

## Tensor contracts

Embeddings have shape `(batch, dimensions)`. Label equality defines positive
pairs; unequal labels define negative pairs. A loss with class attribute
`label_ndim = 2`, currently `MultiSupConLoss`, instead takes boolean
`(batch, classes)` label sets and defines pairs by thresholded Jaccard
similarity. Mined indices are named tuples:
`PairIndices(pos_anchors, positives, neg_anchors, negatives)` and
`TripletIndices(anchors, positives, negatives)`. Named tuples let code dispatch
with `isinstance` instead of counting tensors; plain 3- and 4-tuples are
validated and converted once at the public boundary of losses, miners, and the
`utils.indexing` functions, which users call directly.

`ref_emb=None` means the query set is its own reference set and self-pairs are
excluded. Passing `ref_emb` always describes a separate set, even when it is the
same tensor. Tensor identity previously encoded this distinction, which made
`SelfSupervisedLoss` clone labels only to defeat the check and made results
depend on object identity. `compute_loss` and `mine` receive `ref_emb=None` for
self-comparison, and the indexing utilities take `ref_labels=None` with the same
meaning. With `ref_emb`, labels and reference labels are both given or both
omitted, so a missing reference label set is never mistaken for self-comparison.

Inputs must be on the embeddings' device; a mismatch raises `ValueError`.
Module state follows ordinary PyTorch placement: a forward pass never moves,
casts, or reallocates parameters or buffers. Moving state changed `state_dict`
dtypes and broke optimizer state after one mixed-precision call; moving inputs
silently hid host-to-device copies. Outside autocast, a parameter placement
mismatch raises `ValueError`; under autocast, parameters may keep higher
precision. Constant coefficient tensors are non-persistent buffers so `.to()`
moves them without changing checkpoint contents.

Validation uses explicit exceptions and remains active under `python -O`.
Invalid argument types raise `TypeError`; invalid values, shapes, tuple
lengths, and devices raise `ValueError`. Index tensors, in `indices_tuple` and
in loss terms, must be `int64` or `int32`, the dtypes PyTorch indexing accepts
as positions; a `bool` or `uint8` tensor would index as a mask, so those and
non-integer dtypes raise `TypeError`. Index values are not bounds-checked,
because a range check reads device values on the host; out-of-range indices are
a caller error.

`indices_tuple=None` means unrestricted: every pair or triplet the labels
define. A supplied tuple with no indices, such as an empty miner result, means
nothing was selected, for losses and for miner candidates alike; it is never
reinterpreted as `None`. Miner weights follow the same rule, so an empty miner
result weights every embedding by zero and the loss reduces to a differentiable
zero; treating it as unrestricted gave classification losses their unmined loss.

## Loss terms and reduction

A loss returns a dictionary of terms. Each term holds a `losses` tensor,
`indices`, and an optional `divisor`. The term kind follows from `indices`:
`None` for one already reduced value, a tensor for per-element losses, an
`(anchor, other)` pair, or a triplet. A one-element tuple is rejected so that
one per-element form exists and the `LossTerm` type matches the validator. A
separate `reduction_type` key could disagree with the index structure and
distinguished pairs that every reducer treated alike. Reducers have one hook,
`element_reduction(losses, anchors, embeddings, labels)`, and receive the
anchor indices of any tuple term.

A loss with nothing valid to compute returns `{}` or terms with empty `losses`.
`BaseReducer.forward` sums
term reductions starting from a zero connected to the embeddings, so empty
batches yield a differentiable zero without per-loss zero-term helpers.

Per-anchor terms contain only anchors for which the objective is defined. An
anchor that lacks the pairs its objective needs, such as a positive for SupCon
or a positive and a negative for generalized lifted structure, emits no loss
value rather than a zero or a masked placeholder. Mean reducers then average
over the anchors that define the objective, independent of how many others the
batch holds, and a batch with no such anchor reduces to a differentiable zero.
Each loss page states which anchors it excludes.

Losses whose paper defines one batch-level value (DynamicSoftMarginLoss,
HistogramLoss, ManifoldLoss, P2SGradLoss, PNPLoss, and
ThresholdConsistentMarginLoss) return a single term with `indices=None` and
take no `reducer` option. A reducer could only pass such a term through, so
accepting one would advertise a configuration with no effect.

A `divisor` states a paper-defined normalization, such as ProxyAnchor's number
of proxies with positives. Only `DivisorReducer` can apply it; other reducers
raise `ValueError` for a term with a divisor rather than silently computing a
different objective. `MultipleReducers` routes individual terms.

`PerAnchorReducer` aggregates the supplied losses by anchor with `index_add`
and `bincount`, then passes one value per anchor that has losses to its inner
reducer. A dense `(batch, batch)` matrix with a sentinel failed when references
outnumbered queries and let duplicate tuples overwrite each other.

Regularization penalties are reduced by their regularizer and added after the
objective's reduction; `loss_terms()` returns objective and penalty terms
together for callers who aggregate themselves. An objective term named like a
penalty raises `ValueError` from both entry points, since merging the two
dictionaries would otherwise drop one term and make `forward` and
`loss_terms()` disagree.

## Composition

The public surface is each public module's `__all__`, as stated in
`docs/versioning.md`. Implementation submodules define only the classes their
package exports and underscored helpers, and every type alias or result tuple
that appears in a public signature, such as `PNPVariant` or
`SubCenterOutliers`, lives in `types`. Literal aliases defined beside one class
were importable only through implementation paths, so annotating a call meant
depending on a module layout outside the versioning guarantee.

`BaseMetricLossFunction`, `BaseMiner`, and `BaseRegularizer` extend
`torch.nn.Module` directly and declare keyword-only options typed by
`types.LossOptions` or `types.RegularizerOptions`, so IDEs and the type checker
see full signatures without cooperative multiple inheritance. The reducer
object is used as given; copying it hid shared state.

`BaseProxyLoss` owns the learnable class matrix shared by the classification
and proxy losses: a parameter named `weight` with one row per class or center
(the `nn.Linear` layout), its initializer, and its regularization term. Five
losses had duplicated that code. `ProxyNCALoss` derives from it and reuses the
NCA objective as a function instead of inheriting `NCALoss`, whose references
are embeddings rather than learned proxies. The cosine-margin base shared by
ArcFace, CosFace, and large-margin softmax is private. Its target-cosine hook
receives the labels, and `ArcFaceLoss.target_margins` supplies the margin, so
`DynamicArcFaceLoss` derives per-class margins from a non-persistent buffer.
Mutating a wrapped loss's `margin` each forward left state on the module and
outside PyTorch placement. It derives from `SubCenterArcFaceLoss` because the
paper uses sub-centers and `sub_centers=1` reduces to ArcFace. Margins are
parameterized by the bounds the paper reports, and identical class sizes raise
instead of dividing by a zero count range.

`GenericPairLoss` exposes one hook, `pair_loss(mat, pairs)`, with
`pair_values` and `pair_masks` helpers. Miners accept optional candidate tuples
through `indices_tuple`, like losses, and every built-in miner restricts its
search to them, so cascaded mining composes without miner state.
`CrossBatchMemory`, `SelfSupervisedLoss`, and `DistributedLossWrapper` accept
`types.MetricLoss`: a `BaseMetricLossFunction` or a `MultipleLosses`, whose
forward has the same signature and passes reference arguments to each loss.
They check this union with `isinstance` and rely on the wrapped loss's own
`ref_emb` validation. `MultipleLosses` accepts only the same union, because a
module with another signature, such as `CrossBatchMemory`, would receive
reference arguments in the wrong positions.

Distances declare whether larger values mean greater similarity with the class
attribute `is_inverted`. `BaseDistance.pairwise` applies normalization and the
power to row-wise comparisons, so miners and threshold matching compute the
same quantity as matrix comparisons. `LpDistance.pairwise` uses
`torch.linalg.vector_norm` of the difference, not `pairwise_distance`, whose
`1e-6` offset made identical rows about `sqrt(d) * 1e-6` apart while the matrix
gave zero. `CosineSimilarity` has no `p` or `power` options: several losses and
regularizers require it and take `acos` of, or assume signed, unit-L2 cosines,
which any other setting silently broke. `DotProductSimilarity` keeps both
options and is the only distance that does not normalize by default: its name
promises a plain dot product. `NPairsLoss` and `NormalizedSoftmaxLoss` default
to a normalized instance because their objectives are bounded cosines.

`MultiSupConLoss` keeps the standard pair tuples and derives Jaccard weights from
the labels inside the loss. Embedding a similarity matrix in 4- and 5-tuples
would have required separate conversion, queue, and self-comparison code.
Boolean labels make the matrix product exact for sets without reading device
values, at roughly 2 ms against 450 ms for 512 queries, 4096 references, and
1000 classes on CPU; a kernel for Eq. 4's label counts needs row-wise `minimum`
to avoid an 8 GiB intermediate, with no demonstrated need. Two empty sets have
similarity 1, so a row's self-comparison is always positive and the existing
self-comparison removal applies, and `threshold=1` recovers SupCon.
`CrossBatchMemory` stores label sets through `label_shape` and takes candidate
pairs from an explicit `pair_function`, instead of inspecting the wrapped loss
or duplicating the queue.

## Index enumeration

Pair-to-triplet conversion groups negatives by anchor: it sorts negative pairs
by anchor and uses `searchsorted` and prefix offsets to emit, for each positive
pair, the negatives sharing its anchor. Memory scales with the emitted triplets.
Comparing every positive pair with every negative pair allocated a boolean
matrix of 1.8 GiB for 512 samples with 16 per class, against 87 MiB of output.
All-triplet enumeration converts all pairs the same way. Output order is by
positive pair, then negative pair, and empty inputs produce empty tensors.

Self-comparison removal keeps a pair unless its positive equals the queue slot
or global row of its anchor, `positive == curr_batch_idx[anchor]`, which needs
no host synchronization.

Random triplet sampling draws positives and negatives with row-wise
`torch.multinomial` over masks, in float32 for half-precision weights following
[PyTorch's multinomial CUDA issue](https://github.com/pytorch/pytorch/issues/19900),
with one host synchronization for the maximum count. Tuple-frequency weights
divide each example's `bincount` by the maximum count; with separate references,
only anchors address query weights.

## Numerical choices

Objectives over softmax probabilities are computed in log space. NCA and
ProxyNCA take the difference of two masked log-sum-exp values; masking with the
dtype's finite minimum keeps gradients finite for rows without positives, which
are excluded by label. NT-Xent, tuplet margin, and lifted structure reduce
negatives per anchor with a segment log-sum-exp (`scatter_reduce` maximum, then
`index_add` of shifted exponentials) instead of a positives-by-negatives matrix,
accumulating half precision in float32. Exponentiating probabilities and adding
a small constant before the logarithm lost float16 hard positives entirely.
NT-Xent subtracts each anchor's maximum negative score before temperature
scaling, then takes softplus of the negative log-sum-exp minus the shifted
positive logit. Subtracting the positive from a completed log-sum-exp lost
small losses in the presence of large common score offsets; shifting before
scaling also avoids rounding away score differences during division.
Tuplet margin uses the positive cosine directly at zero margin. At nonzero
margin, identical and antipodal positives use exact endpoint angles with a
zero angular gradient, while interior cosines retain the paper's derivative.
Moving endpoint cosines inward by a dtype-dependent epsilon changed the loss
substantially in float16 and erased valid near-endpoint gradients at zero margin.

Normalization divides by `max(norm, eps)` with `eps = max(1e-12,
finfo(dtype).tiny)`, computing float16 and bfloat16 in float32 and casting
back. `torch.nn.functional.normalize`'s `1e-12` underflows to zero in float16,
so a zero row produced NaN; even computed in float32, the gradient of a row
below `eps` is `1 / eps` times the incoming gradient and overflows float16.
The float16 `eps` of `2**-14` keeps that factor at 16384, while every other
dtype keeps the `1e-12` of float32.

`LpDistance` computes float16 and bfloat16 matrices with `torch.cdist` in
float32 and casts the result, because `torch.cdist` lacks CPU kernels for those
types and expanding every row pair costs memory proportional to queries times
references times dimensions. The call fixes `compute_mode` to the direct
difference kernel: the default switches to $\|x\|^2 + \|y\|^2 - 2 x \cdot y$
above 25 rows, which cancels to zero for nearby half-precision rows and erases
their gradients. Default reductions avoid data-dependent shapes:
`ThresholdReducer` masks and counts, and `DivisorReducer` selects with
`torch.where`. `DivisorReducer` builds the divisor directly in the accumulation
dtype; converting through the default dtype first turned a float64 divisor of
`1e-50` into zero.

`ManifoldLoss` inverts its random-walk matrix in float32 and masks excluded
terms with $-\infty$ before log-sum-exp. `DistanceWeightedMiner` clamps
distances at `cutoff` and works in log space following Section 4 of Wu et al.,
masking ineligible pairs before subtracting each anchor's maximum log weight.
`RankedListLoss` computes its weighted means as softmax-weighted averages over
each anchor's violating set, which cannot overflow. ProxyAnchor applies miner
weights by adding their logarithm inside the log-sum-exp, so a weight
multiplies its exponential term and a zero weight excludes the sample; the
paper defines no miner weighting, and version 2's `w - 1` offset scaled terms
by $e^{w-1}$.

## Stateful modules

Cross-batch memory stores detached reference embeddings. Each forward validates
shapes, devices, the enqueue mask, and exact label conversion to `label_dtype`
(a round trip, because `torch.can_cast` accepted int64 to int32 and merged
distinct labels), then builds a snapshot of the occupied queue with the new
rows written, runs the miner and wrapped loss on it, and only then writes the
rows and advances the cursor. A failing call leaves the queue unchanged, and
snapshots keep tensors saved for backward valid across later forwards. Queue
buffers follow module placement; under autocast, the batch is converted to the
queue dtype while preserving anchor gradients. The cursor and filled flag are
saved under a `queue_state` entry by registered state-dict hooks, so checkpoints
restore the next loss and load with `weights_only=True`.

Dynamic soft margin weights distance differences by a running cumulative
histogram updated in place with detached statistics, following Zhang et al.,
[Learning Local Descriptors With a CDF-Based Dynamic Soft Margin](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Local_Descriptors_With_a_CDF-Based_Dynamic_Soft_Margin_ICCV_2019_paper.pdf),
and the [authors' implementation](https://github.com/lg-zhang/dynamic-soft-margin-pytorch).
Each batch histogram is normalized before momentum, bin centers include both
endpoints of `[min_val, -min_val]`, and out-of-range observations saturate their
weights without clipping the loss.

P2SGrad renormalizes its weight rows in place under `torch.no_grad()` and uses
a differentiable clone for each computation, so parameter identity is stable
and multiple forwards can precede one backward.

## Scope

Modules do not record diagnostic statistics; forward passes return tensors and
leave no side-effect attributes, which avoids host synchronization. Callers
compute diagnostics from returned indices and distance matrices or with forward
hooks. Training orchestration, checkpoint helpers, and logging integrations are
outside this boundary. Mathematical definitions and paper references live on
the component pages; upstream provenance for reference tests is in
`THIRD_PARTY_NOTICES.md`. The development baseline uses current PyTorch;
compatibility with obsolete PyTorch releases is not a requirement.

## Shared distance matrices

Issue #211 asked to compute one distance matrix for a miner and its losses. Issue
#697 asked to pass per-call metadata, such as embedding ids, to a custom distance.
One caller-owned matrix addresses both, so `forward` on losses and miners takes an
optional keyword-only `mat`. Per-call metadata stays in application code and no
signature on `BaseDistance` or `compute_mat` changes.

Support is a class declaration, `supports_mat`, not a guess from a signature. A
class that does not read a batch-by-reference matrix raises rather than ignoring
`mat`, because an ignored matrix would silently train on different values. The
base `compute_loss` and `mine` signatures do not change, so custom subclasses keep
working; only classes that declare support receive a `mat` keyword, and the base
class calls them through a cast because the override adds it. Wrappers that build
their own reference or gather inputs have no `mat`: a caller's matrix would not
describe the rows they compare.

The gradient rules follow from how the modules run. Miners run under
`torch.no_grad()`, so a matrix computed inside one has no graph, and a loss built
on it would train nothing without raising; a loss therefore checks for a graph when
its inputs are tracked. Losses sharing a matrix share its graph, so their sum takes
one `backward()`. No module caches a matrix on itself or on the distance. A cache
keyed on tensors retains graphs between steps, goes stale after in-place updates,
and hides which matrix a loss used. Tests establish that modules do not change
`mat` in place, that no module keeps a reference, and that every declaring class
reads `mat` instead of its distance. The receiving module's `distance` still decides
orientation through `is_inverted` and `margin`. A matrix from another distance
cannot be detected, so this is a documented caller obligation.

`TripletMarginLoss(swap=True)` needs reference-to-reference distances, which `mat`
does not hold. Falling back to `distance.pairwise` would mix the caller's matrix
with values from a distance that lacks the caller's metadata, so it raises.

`docs/benchmarks/reused-distance-m1-pro-v3.json` measures the saving for a miner
and two losses on one CPU thread, with no encoder, as the median of five
interleaved rounds:

| Batch size | Cosine saving | Lp saving | One matrix, forward and backward   |
| ---------- | ------------- | --------- | ---------------------------------- |
| 128        | 20.4%         | 8.9%      | 0.11 to 0.16 ms of about 2 ms      |
| 512        | 16.9%         | 3.8%      | 0.75 ms of about 25 to 30 ms       |
| 2048       | 9.6%          | 2.6%      | 13 to 14 ms of about 400 to 540 ms |

Repeated distance computation is a minor CPU cost that shrinks as batches grow,
because mask construction and pair selection dominate. A whole training step with
an encoder would save a smaller share. #697 is the stronger reason for the API,
because it had no alternative besides mutating the distance object before each call.

GPU time and peak memory have not been measured for this path. The `separate` and
`reused` benchmark cases are ready for that run with `--device cuda`.

## Bounded mining output

Issue #193 asked miners to limit how many tuples they return. `tuples_per_anchor`
lives on `BaseMiner`, so every miner gains it, including custom miners that forward
the keyword. It applies after `mine`, because the miner's selection rule is the
contract; sampling candidates first would change which tuples are hard. Per-anchor
sampling keeps anchors represented, which a global `num_tuples` would not: a global
cap draws mostly from the anchors with the most hard tuples. It samples without
replacement, since a repeated tuple only reweights the loss. The option bounds what
a loss receives, not the miner's own candidate enumeration. `triplets_per_anchor` on
triplet losses stays, because it draws from labels without a miner and has different
semantics.

## Anchor-batched gradients

Issue #586 asked for loss batching as a utility. A loss cannot do this from inside
`forward`, which must return one differentiable scalar, so the graph it returns
holds every chunk at once. `anchor_batched_backward` owns the backward pass instead:
it backpropagates each chunk to a detached leaf, then passes the summed gradient
through the encoder once.

Exactness decides the design. The reduced objective is
`sum(elements) / count` for each term, and both come from the whole batch, so a mean
of chunk means is wrong whenever chunks differ. Each term therefore accumulates
its sum, count, and gradient, and the division happens after the last chunk. A term
needs its own gradient buffer because terms have different denominators. Self-pairs
are removed through explicit candidates, because a supplied reference set means
"no self-exclusion" everywhere else.

`decomposes_by_anchor` is declared per class because shape cannot reveal it.
`IntraPairVarianceLoss` returns per-pair elements yet subtracts a mean over all pairs,
and batching it changed values by 28% and gradients by 47% in the probe that selected
the supported set. Losses are admitted only after matching a full batch; reducers are
admitted by exact type, because a subclass could change the denominator.

Gradient state is part of the contract, because optimizers read it. Autograd reaches
more than the embeddings: the loss's learnable parameters, such as a custom distance's,
get accumulated and delivered the same way, or they would silently stop training. A
term whose elements are all filtered out still backpropagates, which yields zeros and
not `None`, because `AdamW` applies weight decay to a zero gradient and skips a `None`
one. A term with no elements at all is skipped, as `BaseReducer` does, and a parameter
that no term reaches stays `None`. The embeddings always receive at least zeros, as
they do from the reducer's `zero_loss` start value.

Chunks differentiate private copies, never the live parameters. A chunk's gradient is
an intermediate value: it has no denominator yet, so at float16 a sum over tuples
overflows before normalization even within a single chunk, and a tensor hook on the
parameter would run on every chunk and again at delivery. `torch.func.functional_call`
swaps widened, hook-free copies in for each chunk and restores the originals even when
a chunk raises, and the final `backward` delivers the normalized gradient to the live
parameter once, cast to its dtype. Rejecting hooked or low-precision parameters would
have been simpler, but both are ordinary PyTorch usage that a full `backward` handles.

Other memory is untouched: the encoder graph and the gradient remain whole, and each
term holds one `(N, D)` gradient buffer.

These two memory features have CPU timings and no memory measurements. At 256
embeddings, 128 dimensions, and one thread, chunks of 32 anchors cost 4.8 ms against
2.6 ms for NTXentLoss and 10.5 ms against 7.1 ms for TripletMarginLoss, and
`tuples_per_anchor=32` cost 22.7 ms against 10.0 ms for a mined triplet step. Peak
memory needs `--device cuda`, which the benchmark cases support, so the memory
benefit of both features remains a design expectation.

## Decoupled contrastive loss

`DecoupledContrastiveLoss` shares `NTXentLoss`'s pair layout: one element per positive
pair, with negatives grouped by anchor. It does not shift scores by the largest
negative, because that shift exists to keep `softplus` accurate, and nothing here
compares a positive with a sum containing it. An anchor without negatives has no
defined objective, so its pairs are removed instead of contributing the infinite
logarithm of an empty sum. Its default reducer must keep negative values, which the
objective can take.

`decomposes_by_anchor` is set per instance for this loss. Equation 7 normalizes DCLW's
weights by a mean over all positive pairs, which is a batch statistic, while equation 6
has none. The weights stay in the graph because the paper writes no stop-gradient.

## Selectively contrastive triplet loss

The paper's equation and the authors' code differ, so the defaults follow a stated
rule: the equation decides behavior unless it leaves a setting open. Equation 1 omits a
temperature, but cosine similarities in `[-1, 1]` at a unit temperature give weak
gradients, so the default is the code's 0.1, with `temperature=1` recovering the
equation. The code's extra `Neg > 0.8` hard rule changes which case applies, so it is an
opt-in `hard_threshold`. Mining stays a separate miner rather than a built-in selection,
so the loss composes with any tuple source. The loss rejects non-inverted distances,
because the hard case minimizes the value directly, and minimizing a distance would pull
the negative closer without any error. The authors' repository has no license file, so
the implementation comes from the paper; their code served only as a numerical
cross-check outside this repository.

## Spherical embedding regularizer

Issue #346 asked whether SEC duplicates `CenterInvariantRegularizer`. It does not. The
center invariant penalty compares squared norms, so its gradient grows with the cube of
the norm, while SEC compares norms, with a gradient linear in the deviation. The tests
show that neither is a rescaling of the other. SEC computes on the raw row, so it needs
unnormalized embeddings, which is where the losses' distances leave them. It needs no
`distance` argument, because equation 17 is defined only on the Euclidean norm.

## Class centroids

`CentroidTripletLoss` was removed in version 2 because classes of different sizes crashed
it. Its `(class, sample)` padding appended index 0 for missing samples, which put another
class's row into each padded slot, as pgrosjean found in #451, and reshaping tuples then
failed whenever the tuple count was not divisible by the classes. Version 3 has no padding.
`_common.class_centroids` computes class sums with `index_add`, and the leave-one-out mean
of a row is its class sum minus the row. Cost is linear in the batch, with one
`(rows, classes)` distance matrix and no per-sample centroid loop.

`CentroidTripletLoss` computes in float32 for low-precision inputs and stays there through
the reducer. The default squared distance can overflow float16 for valid embeddings even
when the result fits, and `inf - inf` is a NaN that `AvgNonZeroReducer` filters out, so
casting back early would silently return zero with zero gradients. It also keeps very small
`softplus` values from rounding to zero and leaving the reducer's count.

Singletons are skipped rather than raising, as version 1 did. Batches with an occasional
singleton are ordinary in long-tailed training, and raising stops a run at a random step
for a row that simply has no objective. The row still contributes a negative centroid,
because a centroid of one sample is well defined. `swap` and `triplets_per_anchor` were
not restored: the paper defines neither, and every anchor-class triplet is cheap here.

`GE2ELoss` shares that helper but normalizes first, because equation 4 defines centroids
over unit vectors. Averaging raw embeddings would let large embeddings dominate a class's
centroid, and leave-one-out on every class would stop scoring negatives against the
centroids that equation 9 specifies. The own-class score replaces one entry of the
matrix, so negatives use a single `(rows, classes)` comparison. `w > 0` comes from
training `raw_w` with `w = softplus(raw_w)`. A forward clamp was tried first and is wrong: its
derivative is zero below the boundary, so one step past zero left `w` with no gradient and
embeddings with almost none, and training stopped. Softplus cannot cross zero and always
has a gradient. It suits this loss better than `exp`, because its chain-rule factor is
nearly 1 at the paper's scale, so updates there match training `w` directly, while `exp`
rescales every update by `w`.

## Center loss

Algorithm 1 of Wen et al. updates centers by equation 4, with a `1 + n` denominator, not
by the gradient of `L_C`, which has no denominator and grows with the class's batch count.
A `Parameter` would let an optimizer apply that different rule, so `centers` is a buffer
and no gradient reaches it.

The update is an explicit method rather than a side effect of `forward`. `forward` runs for
evaluation, for several micro-batches of one step, and under checkpointing, and each of those
would otherwise move the centers a different number of times. It also avoids the failure in
#731: writing to a tensor a pending graph has saved raises at `backward()`. The loss
indexes `centers` into a new tensor, so a later in-place update does not touch the graph,
and a test confirms that either order works. The cost is that a caller who forgets the
call trains against random centers, which is why the docs state it first.

The loss stays in float32 for low-precision inputs, unlike most losses, because features in
this paper's setting reach hundreds, and half a squared distance then exceeds the float16
range for one row even when the reduced value fits.

## Resolved reports

#731 was a library defect in version 2, not caller ordering. Its `update_histogram` ran
with gradients on, so `self.hist_` kept the batch's graph alive into the next step. The
next `backward()` then walked the previous step's freed graph and raised "second time", and
adding `retain_graph=True` moved the failure to the previous step's weights, which the
optimizer had changed. Running the same training loop on v2.9.0 produces both errors in
that order. Version 3 updates the histogram under `torch.no_grad()` into a buffer, and
`weigh_loss` reads a cumulative sum that is a new tensor, so later updates cannot touch a
pending graph. The tests pin this, with several forwards before `backward()`.
