# BaseProxyLoss

The base class of losses that own one learnable vector per class or per class
center: [ArcFaceLoss](arcface_loss.md), [CosFaceLoss](cosface_loss.md),
[LargeMarginSoftmaxLoss](large_margin_softmax_loss.md),
[NormalizedSoftmaxLoss](normalized_softmax_loss.md),
[ProxyAnchorLoss](proxy_anchor_loss.md), [ProxyNCALoss](proxy_nca_loss.md),
[SoftTripleLoss](soft_triple_loss.md), [SphereFaceLoss](sphereface_loss.md), and
[SubCenterArcFaceLoss](subcenter_arcface_loss.md).

```python {.signature}
losses.BaseProxyLoss(num_weights, embedding_size,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Implementation notes**:

BaseProxyLoss extends [BaseMetricLossFunction](base_metric_loss_function.md)
and owns three things:

- **`weight`**: A `torch.nn.Parameter` of shape `(num_weights, embedding_size)`
  in `torch.nn.Linear.weight` layout, one row per class or center. Subclasses
  with one vector per class pass `num_weights=num_classes`; subclasses with `K`
  centers per class pass `num_classes * K` and store class `c` in rows `c * K`
  through `(c + 1) * K - 1`.
- **`default_weight_init(weight)`**: Initializes `weight` in place when
  `weight_init_func` is `None`. The base implementation is
  `torch.nn.init.normal_`; override it to change a subclass's default.
- **`regularization_terms(embeddings)`**: Adds the `weight_reg_loss` term,
  `weight_reg_weight * weight_regularizer(weight)`, to the embedding
  regularization term when `weight_regularizer` is set. The penalty is added
  after the objective terms are reduced; see
  [regularization terms](../losses.md#regularization-terms).

Subclasses implement `compute_loss`, as for any
[custom loss](../extend/losses.md#losses-with-learnable-parameters). The base
class does not check where `weight` lives; built-in subclasses raise
`ValueError` when `weight` and the embeddings have different devices, or
different dtypes outside autocast. Move the loss with
`loss_func.to(device, dtype)` and include `loss_func.parameters()` in an
optimizer.

**Parameters**:

- **num_weights**: The number of rows of `weight`.
- **embedding_size**: The size of the embeddings passed to the loss, and the
  number of columns of `weight`.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance),
  inherited from BaseMetricLossFunction. Every built-in subclass overrides it.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
