# Numerical components

Losses, miners, distances, reducers, and regularizers share tensor conventions
and pair/triplet conversion. These mutually dependent building blocks form one
policy boundary. They must not depend on training, dataset downloading,
retrieval backends, or distributed process management.

## Tensor contracts

Embeddings have shape `(batch, dimensions)`. Label equality defines positive
pairs; unequal labels define negative pairs. Mined indices are named tuples:
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
lengths, and devices raise `ValueError`.

## Loss terms and reduction

A loss returns a dictionary of terms. Each term holds a `losses` tensor,
`indices`, and an optional `divisor`. The term kind follows from `indices`:
`None` for one already reduced value, a tensor for per-element losses, an
`(anchor, other)` pair, or a triplet. A separate `reduction_type` key could
disagree with the index structure and distinguished pairs that every reducer
treated alike. Reducers have one hook, `element_reduction(losses, anchors,
embeddings, labels)`, and receive the anchor indices of any tuple term.

A loss with nothing valid to compute returns `{}`. `BaseReducer.forward` sums
term reductions starting from a zero connected to the embeddings, so empty
batches yield a differentiable zero without per-loss zero-term helpers.

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
together for callers who aggregate themselves.

## Composition

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
ArcFace, CosFace, and large-margin softmax is private.

`GenericPairLoss` exposes one hook, `pair_loss(mat, pairs)`, with
`pair_values` and `pair_masks` helpers. Miners accept optional candidate tuples
through `indices_tuple`, like losses, and every built-in miner restricts its
search to them, so cascaded mining composes without miner state.
`CrossBatchMemory` and `SelfSupervisedLoss` accept any `BaseMetricLossFunction`
and rely on the wrapped loss's own `ref_emb` validation.

Distances declare whether larger values mean greater similarity with the class
attribute `is_inverted`. `BaseDistance.pairwise` applies normalization and the
power to row-wise comparisons, so miners and threshold matching compute the
same quantity as matrix comparisons.

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

`LpDistance` computes float16 and bfloat16 matrices with `torch.cdist` in
float32 and casts the result, because `torch.cdist` lacks CPU kernels for those
types and expanding every row pair costs memory proportional to queries times
references times dimensions. The call fixes `compute_mode` to the direct
difference kernel: the default switches to $\|x\|^2 + \|y\|^2 - 2 x \cdot y$
above 25 rows, which cancels to zero for nearby half-precision rows and erases
their gradients. Default reductions avoid data-dependent shapes:
`ThresholdReducer` masks and counts, and `DivisorReducer` selects with
`torch.where`.

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
