# SelfSupervisedLoss

A common use case is to have `embeddings` and `ref_emb` be augmented versions of each other. For most losses, you have to create labels to indicate which `embeddings` correspond with which `ref_emb`.

`SelfSupervisedLoss` is a wrapper that takes care of this by creating labels internally. It assumes that:

- `ref_emb[i]` is an augmented version of `embeddings[i]`.
- `ref_emb[i]` is the only augmented version of `embeddings[i]` in the batch.

```python
losses.SelfSupervisedLoss(loss, symmetric=True, **kwargs)
```

**Parameters**:

- **loss**: The loss function to be wrapped.
- **symmetric**: If `True`, then the embeddings in both `embeddings` and `ref_emb` are used as anchors. If `False`, then only the embeddings in `embeddings` are used as anchors.

Example usage:

```
loss_fn = losses.TripletMarginLoss()
loss_fn = SelfSupervisedLoss(loss_fn)
loss = loss_fn(embeddings, ref_emb)
```

**Supported Loss Functions**

The listed types and subclasses that preserve their contracts are supported.
`supported_losses()` returns a tuple of types. ProxyNCA and its subclasses
are excluded because their learned proxies do not accept separate reference
embeddings. Other unsupported losses raise `TypeError` at construction.

- [AngularLoss](angular_loss.md#angularloss)
- [CircleLoss](circle_loss.md#circleloss)
- [ContrastiveLoss](contrastive_loss.md#contrastiveloss)
- [GeneralizedLiftedStructureLoss](generalized_lifted_structure_loss.md#generalizedliftedstructureloss)
- [IntraPairVarianceLoss](intra_pair_variance_loss.md#intrapairvarianceloss)
- [MultiSimilarityLoss](multi_similarity_loss.md#multisimilarityloss)
- [NCALoss](nca_loss.md#ncaloss)
- [NTXentLoss](ntxent_loss.md#ntxentloss)
- [SignalToNoiseRatioContrastiveLoss](signal_to_noise_ratio_contrastive_loss.md#signaltonoiseratiocontrastiveloss)
- [SupConLoss](supcon_loss.md#supconloss)
- [TripletMarginLoss](triplet_margin_loss.md#tripletmarginloss)
- [TupletMarginLoss](tuplet_margin_loss.md#tupletmarginloss)
