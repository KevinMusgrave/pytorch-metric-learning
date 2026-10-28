# SubCenterArcFaceLoss

[Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf){target=_blank}

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

This loss extends [ArcFaceLoss](arcface_loss.md#arcfaceloss). It uses multiple sub centers per class, instead of just a single center, hence the name Sub-center ArcFace.

```python
losses.SubCenterArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64, sub_centers=3,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Parameters**:

- **sub_centers**: The number of sub centers per class.

See [ArcFaceLoss](arcface_loss.md#arcfaceloss) for a description of the other parameters.

**Other info**:

- This loss **requires an optimizer**. See [ArcFaceLoss](arcface_loss.md#arcfaceloss) for details.
- See [ArcFaceLoss](arcface_loss.md#arcfaceloss) for default distance, reducer, and reducer input.

**Getting outliers and dominant centers**

Outliers and dominant centers can be computed as described in the paper.

```python
outliers, dominant_centers = loss_func.get_outliers(
    embeddings, labels, threshold=75, return_dominant_centers=True
)
```
