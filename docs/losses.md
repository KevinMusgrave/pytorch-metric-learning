# Losses

Every loss takes embeddings and labels and returns a scalar tensor:

```python
from pytorch_metric_learning import losses
loss_func = losses.SomeLoss()
loss = loss_func(embeddings, labels)
```

With a [miner](miners.md), pass the mined pairs or triplets as the third argument:

```python
from pytorch_metric_learning import miners
miner_func = miners.SomeMiner()
loss_func = losses.SomeLoss()
miner_output = miner_func(embeddings, labels)
loss = loss_func(embeddings, labels, miner_output)
```

Tuple losses accept pair or triplet indices without labels:

```python
loss = loss_func(embeddings, indices_tuple=pairs)
loss = loss_func(embeddings, indices_tuple=pairs, ref_emb=ref_emb)
```

Tuple losses also accept a separate reference set. Anchors come from
`embeddings`, and positives and negatives come from `ref_emb`:

```python
loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
```

## Forward arguments

```python
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

`ref_emb=None` means self-comparison: each embedding is compared with the other
embeddings in the batch, and an embedding is never its own positive.

Passing `ref_emb` means a separate reference set, even when it is the same
tensor as `embeddings`. Anchors come from `embeddings`; positives and negatives
come from `ref_emb`, and every reference row is a candidate, including a row
that holds the same values as the anchor. With `ref_emb`, `labels` and
`ref_labels` must both be supplied or both be omitted; supplying only one raises
`ValueError`.

### Index tuples

`indices_tuple` takes one of two forms, matching what [miners](miners.md)
return:

- `types.PairIndices(pos_anchors, positives, neg_anchors, negatives)`: positive
  pairs `(pos_anchors[i], positives[i])` and negative pairs
  `(neg_anchors[j], negatives[j])`.
- `types.TripletIndices(anchors, positives, negatives)`: triplets
  `(anchors[i], positives[i], negatives[i])`.

Plain tuples or lists of four or three tensors are accepted and converted to
these named tuples. Every tensor must be one-dimensional, hold integer indices,
and be on the embeddings' device. Anchors index `embeddings`; positives and
negatives index `ref_emb`, or `embeddings` when `ref_emb` is `None`. A
non-tensor entry raises `TypeError`; a wrong length, dimension, or device raises
`ValueError`.

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

Every loss signature ends with `**options`. The accepted options are declared
by `pytorch_metric_learning.types.LossOptions` and are the same for every loss:

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
already reduced and already multiplied by their weights. Use `loss_terms` to
implement a custom aggregation or to inspect per-pair values:

```python
terms = loss_func.loss_terms(embeddings, labels)
pos_pair_losses = terms["pos_loss"]["losses"]
```

Each term is a dictionary with `losses`, a tensor, and `indices`, which says
what one value of `losses` is and how reducers index it:

| `indices`                                        | One value of `losses` per            | Reducers group values by |
| ------------------------------------------------ | ------------------------------------ | ------------------------ |
| `None`                                           | Nothing; `losses` is already reduced | Not reduced further      |
| A tensor of shape `(n,)`                         | Element, such as an embedding        | That tensor              |
| `(anchors, positives)` or `(anchors, negatives)` | Pair                                 | `anchors`                |
| `TripletIndices(anchors, positives, negatives)`  | Triplet                              | `anchors`                |

Each loss page lists its terms under **Reducer input**. A term may also contain
a `divisor`. Such a term must be reduced by
[DivisorReducer](reducers.md#divisorreducer), directly or through
[MultipleReducers](reducers.md#multiplereducers); any other reducer raises
`ValueError`. A loss with no valid pairs, triplets, or queries returns `{}`
from `loss_terms`, and every reducer turns an empty dictionary into a
differentiable zero. See [custom losses](extend/losses.md#loss-terms) for the
full contract.

## Miner weights

Losses that do not consume pairs or triplets directly can still use a miner's
output. ArcFaceLoss, CosFaceLoss, FastAPLoss, LargeMarginSoftmaxLoss, NCALoss,
NormalizedSoftmaxLoss, ProxyAnchorLoss, ProxyNCALoss, SmoothAPLoss,
SoftTripleLoss, SphereFaceLoss, and SubCenterArcFaceLoss turn `indices_tuple`
into one weight per embedding:

$$
w_i = \frac{c_i}{\max_j c_j},
$$

where $c_i$ counts the occurrences of embedding $i$ in the tuples. Embeddings
absent from the tuples get weight 0, and the most frequent embedding gets
weight 1. With `indices_tuple=None`, or with tuples that contain no indices,
every weight is 1. NCALoss counts only anchors when `ref_emb` is supplied,
because positives and negatives then index the reference set. ProxyAnchorLoss
multiplies each exponential term by its weight; the other losses multiply each
embedding's loss by its weight. These losses require `labels`.

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
| [SubCenterArcFaceLoss](losses/subcenter_arcface_loss.md)                                                                                                                                 | `(C·S, D)`     | `torch.nn.init.normal_`                                          |
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

| Loss                     | `get_logits` returns                                                           |
| ------------------------ | ------------------------------------------------------------------------------ |
| ArcFaceLoss, CosFaceLoss | `scale` times the cosine similarity to each class weight                       |
| SubCenterArcFaceLoss     | `scale` times the cosine similarity to each class's closest sub-center         |
| LargeMarginSoftmaxLoss   | `scale` times the weight norm, the embedding norm, and the cosine              |
| SphereFaceLoss           | `scale` times the embedding norm and the cosine                                |
| NormalizedSoftmaxLoss    | The similarity to each normalized class weight divided by `temperature`        |
| ProxyAnchorLoss          | The distance-matrix value to each proxy, cosine similarity by default          |
| ProxyNCALoss             | The distance-matrix value to each proxy, squared Euclidean distance by default |
| SoftTripleLoss           | The relaxed similarity $\mathcal{S}'_{i,c}$ over each class's centers          |

For every loss except ProxyNCALoss with its default distance, a larger value
means a closer class. ProxyNCALoss returns distances unless it is configured
with a similarity, so its closest class is the smallest value. P2SGradLoss and
ManifoldLoss do not provide `get_logits`.

## Choose an objective

Each loss has a separate contract page with its signature, equation, default
distance and reducer, the arguments it supports, and the paper's settings when
the library default differs.

### Pair and triplet objectives

- [AngularLoss](losses/angular_loss.md)
- [CircleLoss](losses/circle_loss.md)
- [ContrastiveLoss](losses/contrastive_loss.md)
- [DynamicSoftMarginLoss](losses/dynamic_soft_margin_loss.md)
- [GeneralizedLiftedStructureLoss](losses/generalized_lifted_structure_loss.md)
- [HistogramLoss](losses/histogram_loss.md)
- [IntraPairVarianceLoss](losses/intra_pair_variance_loss.md)
- [LiftedStructureLoss](losses/lifted_structure_loss.md)
- [MarginLoss](losses/margin_loss.md)
- [MultiSimilarityLoss](losses/multi_similarity_loss.md)
- [NCALoss](losses/nca_loss.md)
- [NPairsLoss](losses/n_pairs_loss.md)
- [RankedListLoss](losses/ranked_list_loss.md)
- [SignalToNoiseRatioContrastiveLoss](losses/signal_to_noise_ratio_contrastive_loss.md)
- [TripletMarginLoss](losses/triplet_margin_loss.md)
- [TupletMarginLoss](losses/tuplet_margin_loss.md)

### Classification and proxy objectives

- [ArcFaceLoss](losses/arcface_loss.md)
- [CosFaceLoss](losses/cosface_loss.md)
- [LargeMarginSoftmaxLoss](losses/large_margin_softmax_loss.md)
- [ManifoldLoss](losses/manifold_loss.md)
- [NormalizedSoftmaxLoss](losses/normalized_softmax_loss.md)
- [P2SGradLoss](losses/p2s_grad_loss.md)
- [ProxyAnchorLoss](losses/proxy_anchor_loss.md)
- [ProxyNCALoss](losses/proxy_nca_loss.md)
- [SoftTripleLoss](losses/soft_triple_loss.md)
- [SphereFaceLoss](losses/sphereface_loss.md)
- [SubCenterArcFaceLoss](losses/subcenter_arcface_loss.md)

### Ranking objectives

- [FastAPLoss](losses/fast_ap_loss.md)
- [PNPLoss](losses/pnp_loss.md)
- [SmoothAPLoss](losses/smooth_ap_loss.md)
- [ThresholdConsistentMarginLoss](losses/threshold_consistent_margin_loss.md)

### Self-supervised and supervised contrastive objectives

- [InstanceLoss](losses/instance_loss.md)
- [NTXentLoss](losses/ntxent_loss.md)
- [SupConLoss](losses/supcon_loss.md)
- [VICRegLoss](losses/vicreg_loss.md)

### Composition and extension

- [BaseMetricLossFunction](losses/base_metric_loss_function.md)
- [BaseProxyLoss](losses/base_proxy_loss.md)
- [CrossBatchMemory](losses/cross_batch_memory.md)
- [GenericPairLoss](losses/generic_pair_loss.md)
- [MultipleLosses](losses/multiple_losses.md)
- [SelfSupervisedLoss](losses/self_supervised_loss.md)
