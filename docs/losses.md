# Losses

Every loss takes embeddings and labels and returns a scalar tensor:

```python
import torch
from pytorch_metric_learning import losses

embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))

loss_func = losses.TripletMarginLoss()
loss = loss_func(embeddings, labels)
```

With a [miner](miners.md), pass the mined pairs or triplets as the third argument:

```python
from pytorch_metric_learning import miners

miner_func = miners.MultiSimilarityMiner()
miner_output = miner_func(embeddings, labels)
loss = loss_func(embeddings, labels, miner_output)
```

Tuple losses accept pair or triplet indices without labels. With `ref_emb`,
positives and negatives index the reference set:

```python
ref_emb = torch.randn(16, 128)
ref_labels = torch.randint(0, 8, (16,))
miner_func = miners.PairMarginMiner()
pairs = miner_func(embeddings, labels)
ref_pairs = miner_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)

loss = loss_func(embeddings, indices_tuple=pairs)
loss = loss_func(embeddings, indices_tuple=ref_pairs, ref_emb=ref_emb)
```

Tuple losses also accept a separate reference set. Anchors come from
`embeddings`, and positives and negatives come from `ref_emb`:

```python
loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
```

## Forward arguments

```python {.signature}
loss_func(embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None)
```

- **embeddings**: A `(N, D)` tensor of query embeddings.
- **labels**: A `(N,)` tensor of labels on the embeddings' device. Labels on
  another device raise `ValueError`; the loss never copies them.
- **indices_tuple**: Optional pair or triplet indices; see
  [index tuples](#index-tuples).
- **ref_emb**: Optional `(M, D)` reference embeddings on the embeddings'
  device; see [reference embeddings](#reference-embeddings).
- **ref_labels**: A `(M,)` tensor of reference labels. It requires `ref_emb`;
  passing it alone raises `ValueError`.

The return value is a scalar tensor connected to the embeddings and to the
loss's parameters. When there are no valid pairs, triplets, or queries, the
result is a differentiable zero.

### Reference embeddings

`ref_emb=None` compares the embeddings with each other, and a supplied `ref_emb`
is a separate reference set, even when it is the same tensor. See
[reference sets](concepts.md#reference-sets) for the indexing rule and the
`labels`/`ref_labels` combinations that raise `ValueError`.

### Index tuples

`indices_tuple` takes one of two forms, matching what [miners](miners.md)
return:

- `types.PairIndices(pos_anchors, positives, neg_anchors, negatives)`: positive
  pairs `(pos_anchors[i], positives[i])` and negative pairs
  `(neg_anchors[j], negatives[j])`.
- `types.TripletIndices(anchors, positives, negatives)`: triplets
  `(anchors[i], positives[i], negatives[i])`.

Plain tuples or lists of four or three tensors are accepted and converted to
these named tuples. Every tensor must be one-dimensional, have dtype `int64` or
`int32`, and be on the embeddings' device. Anchors index `embeddings`;
positives and negatives index `ref_emb`, or `embeddings` when `ref_emb` is
`None`. A non-tensor entry or another index dtype, including `bool` and
`uint8`, raises `TypeError`; a wrong length, dimension, or device raises
`ValueError`. Indices must be in range: losses do not check bounds, because that
would synchronize the device, so negative or out-of-range indices are a caller
error with undefined results.

`indices_tuple=None` means no restriction: the loss uses every pair or triplet
its labels define. A supplied tuple whose tensors are empty means that nothing
was selected, which gives a differentiable zero.

Losses use the tuples in one of three ways:

- Pair losses use the pairs directly and read triplets `(a, p, n)` as the pairs
  `(a, p)` and `(a, n)`. Triplet losses use the triplets directly and form
  triplets from pairs by combining each anchor's positives with its negatives.
- Classification, proxy, and some ranking losses use the tuples only to weight
  embeddings; see [miner weights](#miner-weights).
- InstanceLoss, ManifoldLoss, P2SGradLoss, PNPLoss, and RankedListLoss raise
  `ValueError` when `indices_tuple` is supplied.

## Supported arguments

Each loss accepts the arguments listed for it below. Passing an unsupported
argument raises `ValueError`. These lists are verified by the test suite.

**Losses that accept `indices_tuple` without `labels`**

- CircleLoss
- ContrastiveLoss
- DynamicSoftMarginLoss
- GeneralizedLiftedStructureLoss
- HistogramLoss
- IntraPairVarianceLoss
- LiftedStructureLoss
- MarginLoss
- MultiSimilarityLoss
- NTXentLoss
- SignalToNoiseRatioContrastiveLoss
- SupConLoss
- ThresholdConsistentMarginLoss
- TripletMarginLoss
- TupletMarginLoss

**Losses that accept `ref_emb` and `ref_labels`**

- AngularLoss
- CircleLoss
- ContrastiveLoss
- DynamicSoftMarginLoss
- GeneralizedLiftedStructureLoss
- IntraPairVarianceLoss
- MarginLoss
- MultiSimilarityLoss
- MultiSupConLoss
- NCALoss
- NTXentLoss
- SignalToNoiseRatioContrastiveLoss
- SupConLoss
- ThresholdConsistentMarginLoss
- TripletMarginLoss
- TupletMarginLoss

Use a loss from this list whenever queries are compared with a separate
reference set, such as a gallery, a second augmented view, or a memory bank.
[CrossBatchMemory](losses/cross_batch_memory.md) and
[SelfSupervisedLoss](losses/self_supervised_loss.md) with `symmetric=False`
pass a reference set to the wrapped loss, so they require a loss from this
list. Every other loss raises `ValueError` when `ref_emb` is supplied.

## Loss options

Most loss signatures end with `**options`, declared by
`pytorch_metric_learning.types.LossOptions`. Losses that return one
already-reduced value (DynamicSoftMarginLoss, HistogramLoss, ManifoldLoss,
P2SGradLoss, PNPLoss, ThresholdConsistentMarginLoss) list `distance` (except
P2SGradLoss), `embedding_regularizer`, and `embedding_reg_weight` explicitly and
take no `reducer`. Wrappers (CrossBatchMemory, MultipleLosses,
SelfSupervisedLoss) take the losses they wrap instead. The options are:

- **distance**: A [distance](distances.md) module. `None` selects the loss's
  default distance. A loss that supports only some distances raises `TypeError`
  for another distance class and `ValueError` for an incompatible setting.
- **reducer**: A [reducer](reducers.md) module. `None` selects the loss's
  default reducer. The reducer object is used as given, not copied; a loss with
  several terms applies it to each term unless it is a
  [MultipleReducers](reducers.md#multiplereducers).
- **embedding_regularizer**: A [regularizer](regularizers.md) applied to the
  embeddings. `None` adds no penalty.
- **embedding_reg_weight**: The weight multiplying that penalty. The default is 1.

```python
from pytorch_metric_learning import distances, losses, reducers, regularizers

loss_func = losses.ContrastiveLoss(
    neg_margin=0.5,
    distance=distances.CosineSimilarity(),
    reducer=reducers.MeanReducer(),
    embedding_regularizer=regularizers.LpRegularizer(),
    embedding_reg_weight=0.1,
)
```

Proxy losses also accept the [learnable weight arguments](#learnable-weight-arguments),
declared by `types.ProxyLossOptions`.

## Regularization terms

Losses with [learnable weights](#learnable-weight-arguments) also accept
`weight_regularizer` and `weight_reg_weight`. The total loss is

$$
L = \operatorname{reduce}(\text{objective terms})
+ w_e \, R_e(\text{embeddings}) + w_w \, R_w(\text{weights}),
$$

where each penalty appears only when its regularizer is supplied. The
[reducer](reducers.md) sees only the loss's own objective terms. Each
regularizer applies its own reducer, and the weighted result is added after
reduction.

## Unreduced loss terms

`loss_func.loss_terms(embeddings, labels, ...)` takes the same arguments as
`forward` and returns the `pytorch_metric_learning.types.LossDict` that the
reducer would receive, plus the regularization terms `embedding_reg_loss` and
`weight_reg_loss` when their regularizers are set. Regularization terms are
already reduced and already multiplied by their weights. An objective term
with the same name as a regularization term raises `ValueError` from both
`forward` and `loss_terms`. Use `loss_terms` to
implement a custom aggregation or to inspect per-pair values:

```python
terms = loss_func.loss_terms(embeddings, labels)
pos_pair_losses = terms["pos_loss"]["losses"]
```

Each term follows the [`LossTerm` contract](types.md#lossterm): its `indices`
say whether one value of `losses` is already reduced or belongs to an element, a
pair, or a triplet, and a term with a `divisor` requires
[DivisorReducer](reducers.md#divisorreducer). Each loss page lists its terms
under **Reducer input**. A loss with no valid pairs, triplets, or queries
returns `{}` or terms with empty `losses`, which
[reduce](reducers.md#reduction) to a differentiable zero; an embedding
regularizer still adds its penalty.

## Miner weights

Losses that do not consume pairs or triplets directly can still use a miner's
output. ArcFaceLoss, CosFaceLoss, DynamicArcFaceLoss, FastAPLoss,
LargeMarginSoftmaxLoss, NCALoss,
NormalizedSoftmaxLoss, ProxyAnchorLoss, ProxyNCALoss, SmoothAPLoss,
SoftTripleLoss, SphereFaceLoss, and SubCenterArcFaceLoss turn `indices_tuple`
into one weight per embedding:

$$
w_i = \frac{c_i}{\max_j c_j},
$$

where $c_i$ counts the occurrences of embedding $i$ in the tuples. Embeddings
absent from the tuples get weight 0, and the most frequent embedding gets
weight 1. With `indices_tuple=None`, every weight is 1. A supplied tuple that
contains no indices, such as an empty miner result, gives every embedding
weight 0, so the loss is a differentiable zero rather than the unmined loss.
NCALoss counts only anchors when `ref_emb` is supplied, because positives and
negatives then index the reference set. ProxyAnchorLoss multiplies each
exponential term by its weight; the other losses multiply each embedding's loss
by its weight. These losses require `labels`.

## Learnable weight arguments

[BaseProxyLoss](losses/base_proxy_loss.md) is the base class of every loss that
owns one learnable vector per class or per class center. It stores them in a
parameter named `weight` with shape `(num_weights, embedding_size)`; row `r` is
one vector, as in `torch.nn.Linear.weight`. With one vector per class, row `c`
belongs to class `c`. With `K` centers per class, rows `c * K` through
`(c + 1) * K - 1` belong to class `c`. Each subclass declares these keyword-only
arguments:

- **weight_regularizer**: A [regularizer](regularizers.md) applied to `weight`.
  If `None`, no weight penalty is added. The penalty appears in `loss_terms` as
  `weight_reg_loss`.
- **weight_reg_weight**: The weight multiplying that penalty before it is added
  to the reduced loss. The default is 1.0.
- **weight_init_func**: A callable that initializes `weight` in place, such as
  `torch.nn.init.xavier_uniform_` or
  `functools.partial(torch.nn.init.kaiming_uniform_, a=1, mode="fan_out")`. If
  `None`, the loss's `default_weight_init(weight)` is used, as listed below.

| Loss                                                                                                                                                                                     | `weight` shape | Default initialization                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------- |
| [ArcFaceLoss](losses/arcface_loss.md), [CosFaceLoss](losses/cosface_loss.md), [LargeMarginSoftmaxLoss](losses/large_margin_softmax_loss.md), [SphereFaceLoss](losses/sphereface_loss.md) | `(C, D)`       | `torch.nn.init.normal_`                                          |
| [SubCenterArcFaceLoss](losses/subcenter_arcface_loss.md), [DynamicArcFaceLoss](losses/dynamic_arcface_loss.md)                                                                           | `(C·S, D)`     | `torch.nn.init.normal_`                                          |
| [NormalizedSoftmaxLoss](losses/normalized_softmax_loss.md)                                                                                                                               | `(C, D)`       | `torch.nn.init.normal_`                                          |
| [ProxyAnchorLoss](losses/proxy_anchor_loss.md)                                                                                                                                           | `(C, D)`       | `torch.nn.init.kaiming_normal_(mode="fan_out")`                  |
| [ProxyNCALoss](losses/proxy_nca_loss.md)                                                                                                                                                 | `(C, D)`       | `torch.nn.init.normal_`                                          |
| [SoftTripleLoss](losses/soft_triple_loss.md)                                                                                                                                             | `(C·K, D)`     | `torch.nn.init.kaiming_uniform_(a=math.sqrt(5), mode="fan_out")` |

`D` is the embedding size, `C` the number of classes, `S` the sub-centers per
class, and `K` the centers per class.

These parameters belong to the loss module. Move the loss with
`loss_func.to(device, dtype)` and include `loss_func.parameters()` in an
optimizer:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.ArcFaceLoss(num_classes=10, embedding_size=128)
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)

embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 10, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
loss_optimizer.step()
```

Outside autocast, the parameters must have the embeddings' device and dtype;
otherwise the loss raises `ValueError` naming both placements. Under autocast,
parameters may keep a higher precision than the embeddings. The loss never
moves or casts its own parameters during a forward pass.

[P2SGradLoss](losses/p2s_grad_loss.md) owns a `weight` parameter and
[ManifoldLoss](losses/manifold_loss.md) owns `proxies` under the same placement
rule. They do not derive from BaseProxyLoss and accept only the embedding
regularization options.

## Classification logits

These losses provide `loss_func.get_logits(embeddings)`, which returns a
`(N, num_classes)` tensor with one score per embedding and class. The scores
omit the training margin, so they suit prediction and evaluation:

| Loss                                     | `get_logits` returns                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| ArcFaceLoss, CosFaceLoss                 | `scale` times the cosine similarity to each class weight                       |
| SubCenterArcFaceLoss, DynamicArcFaceLoss | `scale` times the cosine similarity to each class's closest sub-center         |
| LargeMarginSoftmaxLoss                   | `scale` times the weight norm, the embedding norm, and the cosine              |
| SphereFaceLoss                           | `scale` times the embedding norm and the cosine                                |
| NormalizedSoftmaxLoss                    | The similarity to each normalized class weight divided by `temperature`        |
| ProxyAnchorLoss                          | The distance-matrix value to each proxy, cosine similarity by default          |
| ProxyNCALoss                             | The distance-matrix value to each proxy, squared Euclidean distance by default |
| SoftTripleLoss                           | The relaxed similarity $\mathcal{S}'_{i,c}$ over each class's centers          |

For every loss except ProxyNCALoss with its default distance, a larger value
means a closer class. ProxyNCALoss returns distances unless it is configured
with a similarity, so its closest class is the smallest value. P2SGradLoss and
ManifoldLoss do not provide `get_logits`.

## Choose an objective

Each loss has a separate contract page with its signature, equation, default
distance and reducer, the arguments it supports, and the paper's settings when
the library default differs. Use the guidance below to pick a starting point.

### Pair and triplet objectives

These compare embeddings with each other and need several examples per class in
each batch, such as batches from [MPerClassSampler](samplers.md#mperclasssampler).

- [AngularLoss](losses/angular_loss.md): triplets constrained by the angle at
  the negative, which is invariant to the scale of the distances.
- [CircleLoss](losses/circle_loss.md): pairs weighted by how far each
  similarity is from its optimum, with one margin and one scale.
- [ContrastiveLoss](losses/contrastive_loss.md): the simplest pair objective;
  pull positives within one margin and push negatives beyond another.
- [DynamicSoftMarginLoss](losses/dynamic_soft_margin_loss.md): local
  descriptors and patch matching, where the margin adapts to a running
  histogram of distances.
- [GeneralizedLiftedStructureLoss](losses/generalized_lifted_structure_loss.md):
  lifted structure per anchor, with support for a separate reference set.
- [HistogramLoss](losses/histogram_loss.md): no margin to tune; it penalizes the
  overlap of the positive and negative similarity distributions.
- [IntraPairVarianceLoss](losses/intra_pair_variance_loss.md): an auxiliary term
  that reduces the variance of pair similarities, combined with TupletMarginLoss
  through [MultipleLosses](losses/multiple_losses.md) as in its paper.
- [LiftedStructureLoss](losses/lifted_structure_loss.md): each positive pair
  against every negative of both of its members in the batch.
- [MarginLoss](losses/margin_loss.md): a boundary between positive and negative
  distances, optionally learned per class, from the same paper as
  [DistanceWeightedMiner](miners.md#distanceweightedminer).
- [MultiSimilarityLoss](losses/multi_similarity_loss.md): a strong pair
  baseline that weights pairs by their relative similarities; pair it with
  [MultiSimilarityMiner](miners.md#multisimilarityminer).
- [NCALoss](losses/nca_loss.md): a softmax over distances to every reference,
  with no margin.
- [NPairsLoss](losses/n_pairs_loss.md): batches of one positive pair per class,
  scored as classification among the batch's positives.
- [RankedListLoss](losses/ranked_list_loss.md): each anchor's ranked list of
  references, separating positives from negatives by a margin.
- [SignalToNoiseRatioContrastiveLoss](losses/signal_to_noise_ratio_contrastive_loss.md):
  the contrastive objective measured with the signal-to-noise ratio distance.
- [TripletMarginLoss](losses/triplet_margin_loss.md): the standard triplet
  baseline; combine it with a miner such as
  [TripletMarginMiner](miners.md#tripletmarginminer).
- [TupletMarginLoss](losses/tuplet_margin_loss.md): one positive against all
  negatives of an anchor with scaled angular margins.

### Classification and proxy objectives

These learn one vector per class, or per class center, so they need only labels
and work with ordinary random batches. Train their parameters with the model;
see [learnable weight arguments](#learnable-weight-arguments).

- [ArcFaceLoss](losses/arcface_loss.md): an additive angular margin, widely
  used for face recognition and other tasks with many classes.
- [CosFaceLoss](losses/cosface_loss.md): an additive cosine margin.
- [DynamicArcFaceLoss](losses/dynamic_arcface_loss.md): SubCenterArcFace with
  larger margins for smaller classes, for long-tailed class distributions.
- [LargeMarginSoftmaxLoss](losses/large_margin_softmax_loss.md): a
  multiplicative angular margin that keeps the weight and embedding norms.
- [ManifoldLoss](losses/manifold_loss.md): proxies compared through a
  random-walk similarity over the batch and the proxies.
- [NormalizedSoftmaxLoss](losses/normalized_softmax_loss.md): a cosine softmax
  with a temperature and no margin; a strong, simple baseline.
- [P2SGradLoss](losses/p2s_grad_loss.md): cosine classification without margin
  or scale hyperparameters.
- [ProxyAnchorLoss](losses/proxy_anchor_loss.md): proxies as anchors against all
  embeddings of the batch; fast to converge.
- [ProxyNCALoss](losses/proxy_nca_loss.md): NCA against one learned proxy per
  class.
- [SoftTripleLoss](losses/soft_triple_loss.md): several centers per class, for
  classes with multiple modes.
- [SphereFaceLoss](losses/sphereface_loss.md): a multiplicative angular margin
  on normalized weights.
- [SubCenterArcFaceLoss](losses/subcenter_arcface_loss.md): ArcFace with
  sub-centers per class, for noisy labels; it can report outliers.

### Ranking objectives

These optimize a retrieval ranking measure directly and benefit from large
batches with several positives per class.

- [FastAPLoss](losses/fast_ap_loss.md): average precision through a histogram
  of distances.
- [PNPLoss](losses/pnp_loss.md): average precision that penalizes only negatives
  ranked before positives, with paper variants.
- [SmoothAPLoss](losses/smooth_ap_loss.md): average precision with a sigmoid
  relaxation of the ranking.
- [ThresholdConsistentMarginLoss](losses/threshold_consistent_margin_loss.md):
  an auxiliary term that makes one similarity threshold work across classes;
  add it to another loss with [MultipleLosses](losses/multiple_losses.md).

### Self-supervised and supervised contrastive objectives

- [InstanceLoss](losses/instance_loss.md): classification of each instance
  against the batch, as in image-text instance retrieval.
- [MultiSupConLoss](losses/multi_supcon_loss.md): SupCon for label sets, with
  positives chosen and weighted by Jaccard similarity.
- [NTXentLoss](losses/ntxent_loss.md): the InfoNCE objective of SimCLR and MoCo;
  use [SelfSupervisedLoss](losses/self_supervised_loss.md) for unlabeled views.
- [SupConLoss](losses/supcon_loss.md): the contrastive objective with every
  same-label sample as a positive.
- [VICRegLoss](losses/vicreg_loss.md): two views without negatives, regularized
  by variance and covariance terms.

### Composition and extension

- [BaseMetricLossFunction](losses/base_metric_loss_function.md): subclass it to
  write a [custom loss](extend/losses.md).
- [BaseProxyLoss](losses/base_proxy_loss.md): subclass it for a custom loss with
  one learnable vector per class.
- [CrossBatchMemory](losses/cross_batch_memory.md): more references than one
  batch holds, from a queue of earlier embeddings.
- [GenericPairLoss](losses/generic_pair_loss.md): subclass it for a custom loss
  over positive and negative pairs.
- [MultipleLosses](losses/multiple_losses.md): a weighted sum of losses, each
  with an optional miner.
- [SelfSupervisedLoss](losses/self_supervised_loss.md): two augmented views
  without labels, with any loss that uses labels.
