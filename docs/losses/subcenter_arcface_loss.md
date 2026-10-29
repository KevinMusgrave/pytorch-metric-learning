# SubCenterArcFaceLoss

[Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf){target=_blank}

This loss extends [ArcFaceLoss](arcface_loss.md#arcfaceloss). It uses multiple sub centers per class, instead of just a single center, hence the name Sub-center ArcFace.

```python
losses.SubCenterArcFaceLoss(num_classes, embedding_size, margin=0.5, scale=64, sub_centers=3,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **options)
```

**Equation**:

$$
\cos\theta_{j} = \max_{k \in \lbrace 1, \ldots, K \rbrace} \cos\theta_{j,k},
\qquad
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cos\left(\theta_{y_i} + m\right)}}{e^{s \cos\left(\theta_{y_i} + m\right)} + \sum_{j \neq y_i} e^{s \cos \theta_j}}
$$

Here $\theta_{j,k}$ is the angle between an embedding and the $k$-th of the $K$ sub-centers of class $j$. Each class's logit is its closest sub-center, then the [ArcFaceLoss](arcface_loss.md#arcfaceloss) margin and scale apply.

**Parameters**:

- **sub_centers**: $K$, the number of sub-centers per class. The paper uses 3.

See [ArcFaceLoss](arcface_loss.md#arcfaceloss) for the other parameters. The sub-centers are stored in `weight` with shape `(num_classes * sub_centers, embedding_size)`; rows `c * sub_centers` through `(c + 1) * sub_centers - 1` belong to class `c`.

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

`outliers` holds the indices of embeddings whose similarity to their class's
dominant sub-center is below `cos(threshold)`, with `threshold` in degrees.
`dominant_centers` has shape `(num_classes, embedding_size)`; a class's dominant
sub-center is the one nearest to the most of its embeddings in `embeddings`.
With `return_dominant_centers=False`, only `outliers` is returned.
