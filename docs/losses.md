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
`embeddings` and positives and negatives from `ref_emb`:

```python
loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
```

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

These are also the losses that work inside [CrossBatchMemory](losses/cross_batch_memory.md)
and [SelfSupervisedLoss](losses/self_supervised_loss.md), which pass a separate
reference set to the wrapped loss.

## Loss options

Every loss signature ends with `**options`. The accepted options are declared
by `pytorch_metric_learning.types.LossOptions` and are the same for every loss:

- **distance**: A [distance](distances.md) module. `None` selects the loss's
  default distance.
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
reducer would receive, plus any regularization terms as `already_reduced`
entries. Use it to implement a custom aggregation or to inspect per-pair values:

```python
terms = loss_func.loss_terms(embeddings, labels)
pos_pair_losses = terms["pos_loss"]["losses"]
```

Each term is a dictionary with `losses` (a tensor), `indices`, and
`reduction_type`; see [reduction types](extend/losses.md#reduction-type).

## Learnable weight arguments

Losses that own class weights or proxies store them in a parameter named
`weight` with shape `(num_classes, embedding_size)`, or
`(num_classes * centers, embedding_size)` when a class has several centers.
Row `c` is the weight vector of class `c`, like `torch.nn.Linear.weight`. These
losses declare keyword-only arguments:

- **weight_regularizer**: A [regularizer](regularizers.md) applied to `weight`.
  If `None`, no weight penalty is added.
- **weight_reg_weight**: The weight multiplying that penalty before it is added
  to the reduced loss.
- **weight_init_func**: A callable that initializes `weight` in place, such as
  `torch.nn.init.xavier_uniform_` or
  `functools.partial(torch.nn.init.kaiming_uniform_, a=1, mode="fan_out")`. If
  `None`, the loss's default below is used.

| Loss                                                                                                                                                                                                                                                   | `weight` shape | Default initialization                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- | ---------------------------------------------------------------- |
| [ArcFaceLoss](losses/arcface_loss.md#arcfaceloss), [CosFaceLoss](losses/cosface_loss.md#cosfaceloss), [LargeMarginSoftmaxLoss](losses/large_margin_softmax_loss.md#largemarginsoftmaxloss), [SphereFaceLoss](losses/sphereface_loss.md#spherefaceloss) | `(C, D)`       | `torch.nn.init.normal_`                                          |
| [SubCenterArcFaceLoss](losses/subcenter_arcface_loss.md#subcenterarcfaceloss)                                                                                                                                                                          | `(C·S, D)`     | `torch.nn.init.normal_`                                          |
| [NormalizedSoftmaxLoss](losses/normalized_softmax_loss.md#normalizedsoftmaxloss)                                                                                                                                                                       | `(C, D)`       | `torch.nn.init.normal_`                                          |
| [ProxyAnchorLoss](losses/proxy_anchor_loss.md#proxyanchorloss)                                                                                                                                                                                         | `(C, D)`       | `torch.nn.init.kaiming_normal_(mode="fan_out")`                  |
| [ProxyNCALoss](losses/proxy_nca_loss.md#proxyncaloss)                                                                                                                                                                                                  | `(C, D)`       | `torch.nn.init.normal_`                                          |
| [SoftTripleLoss](losses/soft_triple_loss.md#softtripleloss)                                                                                                                                                                                            | `(C·K, D)`     | `torch.nn.init.kaiming_uniform_(a=math.sqrt(5), mode="fan_out")` |

`D` is the embedding size, `C` the number of classes, `S` the sub-centers per
class, and `K` the centers per class.

These parameters belong to the loss module. Move the loss with
`loss_func.to(device, dtype)` and include `loss_func.parameters()` in the
optimizer. Outside autocast, the parameters must have the embeddings' device and
dtype; otherwise the loss raises `ValueError` naming both placements. Under
autocast, parameters may keep a higher precision than the embeddings. The loss
never moves or casts its own parameters during a forward pass.
`P2SGradLoss` owns a `weight` parameter and `ManifoldLoss` owns `proxies` under
the same placement rule; their regularization arguments are only the embedding
ones.

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

For classification losses, `loss_func.get_logits(embeddings)` returns the
scaled class logits.

### Ranking objectives

- [FastAPLoss](losses/fast_ap_loss.md)
- [PNPLoss](losses/pnp_loss.md)
- [SmoothAPLoss](losses/smooth_ap.md)
- [ThresholdConsistentMarginLoss](losses/tcm_loss.md)

### Self-supervised and supervised contrastive objectives

- [InstanceLoss](losses/instance_loss.md)
- [NTXentLoss](losses/ntxent_loss.md)
- [SupConLoss](losses/supcon_loss.md)
- [VICRegLoss](losses/vicreg_loss.md)

### Composition and extension

- [BaseMetricLossFunction](losses/base_metric_loss_function.md)
- [CrossBatchMemory](losses/cross_batch_memory.md)
- [GenericPairLoss](losses/generic_pair_loss.md)
- [MultipleLosses](losses/multiple_losses.md)
- [SelfSupervisedLoss](losses/self_supervised_loss.md)
