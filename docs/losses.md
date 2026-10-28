# Losses

All loss functions are used as follows:

```python
from pytorch_metric_learning import losses
loss_func = losses.SomeLoss()
loss = loss_func(embeddings, labels) # in your training for-loop
```

Or if you are using a loss in conjunction with a [miner](miners.md):

```python
from pytorch_metric_learning import miners
miner_func = miners.SomeMiner()
loss_func = losses.SomeLoss()
miner_output = miner_func(embeddings, labels) # in your training for-loop
loss = loss_func(embeddings, labels, miner_output)
```

For some losses, you don't need to pass in labels if you are already passing in pair/triplet indices:

```python
loss = loss_func(embeddings, indices_tuple=pairs)
# it also works with ref_emb
loss = loss_func(embeddings, indices_tuple=pairs, ref_emb=ref_emb)
```

**Losses for which you can pass in `indices_tuple` without `labels`**

- CircleLoss
- ContrastiveLoss
- IntraPairVarianceLoss
- GeneralizedLiftedStructureLoss
- LiftedStructureLoss
- MarginLoss
- MultiSimilarityLoss
- NTXentLoss
- SignalToNoiseRatioContrastiveLoss
- SupConLoss
- TripletMarginLoss
- TupletMarginLoss

You can specify how losses get reduced to a single value by using a [reducer](reducers.md):

```python
from pytorch_metric_learning import reducers
reducer = reducers.SomeReducer()
loss_func = losses.SomeLoss(reducer=reducer)
loss = loss_func(embeddings, labels) # in your training for-loop
```

For tuple losses, you can separate the source of anchors and positives/negatives:

```python
loss_func = losses.SomeLoss()
# anchors will come from embeddings
# positives/negatives will come from ref_emb
loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
```

For classification losses, you can get logits using the `get_logits` function:

```python
loss_func = losses.SomeClassificationLoss()
logits = loss_func.get_logits(embeddings)
```

## Regularization terms

Every loss accepts `embedding_regularizer` and `embedding_reg_weight`. Losses
with [learnable weights](#learnable-weight-arguments) also accept
`weight_regularizer` and `weight_reg_weight`. The total loss is

$$
L = \operatorname{reduce}(\text{objective terms})
+ w_e \, R_e(\text{embeddings}) + w_w \, R_w(\text{weights}),
$$

where each penalty appears only when its regularizer is supplied. The
[reducer](reducers.md) sees only the loss's own objective terms. Each
regularizer applies its own reducer, and the weighted result is added after
reduction. With `DoNothingReducer`, the returned dictionary also contains
`"embedding_reg_loss"` and `"weight_reg_loss"` entries whose `"losses"` value is
the weighted scalar penalty and whose `"reduction_type"` is
`"already_reduced"`.

```python
from pytorch_metric_learning import losses, regularizers

loss_func = losses.ContrastiveLoss(
    embedding_regularizer=regularizers.LpRegularizer(),
    embedding_reg_weight=0.1,
)
```

## Learnable weight arguments

Losses that own class weights or proxies declare these keyword-only arguments
in their signatures:

- **weight_regularizer**: A [regularizer](regularizers.md) applied to the loss's
  learnable weights, one row per class vector or proxy. If `None`, no weight
  penalty is added.
- **weight_reg_weight**: The weight multiplying that penalty before it is added
  to the reduced loss.
- **weight_init_func**: A callable that initializes the parameter in place, such
  as `torch.nn.init.xavier_uniform_` or
  `functools.partial(torch.nn.init.kaiming_uniform_, a=1, mode="fan_out")`. If
  `None`, the loss's default below is used.

| Loss                                                                                                                                                                                                                                                                                                                                  | Parameter           | Default initialization                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------ |
| [ArcFaceLoss](losses/arcface_loss.md#arcfaceloss), [CosFaceLoss](losses/cosface_loss.md#cosfaceloss), [LargeMarginSoftmaxLoss](losses/large_margin_softmax_loss.md#largemarginsoftmaxloss), [SphereFaceLoss](losses/sphereface_loss.md#spherefaceloss), [SubCenterArcFaceLoss](losses/subcenter_arcface_loss.md#subcenterarcfaceloss) | `W`, `(D, C)`       | `torch.nn.init.normal_`                          |
| [NormalizedSoftmaxLoss](losses/normalized_softmax_loss.md#normalizedsoftmaxloss)                                                                                                                                                                                                                                                      | `W`, `(D, C)`       | `torch.nn.init.normal_`                          |
| [ProxyAnchorLoss](losses/proxy_anchor_loss.md#proxyanchorloss)                                                                                                                                                                                                                                                                        | `proxies`, `(C, D)` | `torch.nn.init.kaiming_normal_(mode="fan_out")`  |
| [ProxyNCALoss](losses/proxy_nca_loss.md#proxyncaloss)                                                                                                                                                                                                                                                                                 | `proxies`, `(C, D)` | `torch.nn.init.normal_`                          |
| [SoftTripleLoss](losses/soft_triple_loss.md#softtripleloss)                                                                                                                                                                                                                                                                           | `fc`, `(D, C·K)`    | `torch.nn.init.kaiming_uniform_(a=math.sqrt(5))` |

`D` is the embedding size, `C` the number of classes, and `K` the centers per
class. `SubCenterArcFaceLoss` stores `C·sub_centers` columns.

These parameters belong to the loss module. Move the loss with
`loss_func.to(device, dtype)` and include `loss_func.parameters()` in the
optimizer. Outside autocast, the parameters must have the embeddings' device and
dtype; otherwise the loss raises `ValueError` naming both placements. Under
autocast, parameters may keep a higher precision than the embeddings. The loss
never moves or casts its own parameters during a forward pass.
`P2SGradLoss` and `ManifoldLoss` own parameters with the same placement rule;
their regularization arguments are only the embedding ones.

## Choose an objective

Each loss has a separate contract page. Pages with an **Equation** block reproduce the stated formula; otherwise the linked paper is the mathematical reference. Default distances, reduction, and implementation notes describe the library behavior.

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
