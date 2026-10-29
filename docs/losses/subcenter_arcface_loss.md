# SubCenterArcFaceLoss

[Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)

```python
losses.SubCenterArcFaceLoss(num_classes, embedding_size, margin=0.5, scale=64, sub_centers=3,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
\cos\theta_{j} = \max_{k \in \lbrace 1, \ldots, K \rbrace} \cos\theta_{j,k},
\qquad
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cos\left(\theta_{y_i} + m\right)}}{e^{s \cos\left(\theta_{y_i} + m\right)} + \sum_{j \neq y_i} e^{s \cos \theta_j}}
$$

Here $\theta_{j,k}$ is the angle between an embedding and the $k$-th of the $K$ sub-centers of class $j$. Each class's logit uses its closest sub-center, then the [ArcFaceLoss](arcface_loss.md) margin and scale apply.

**Implementation notes**: The target-logit treatment for
$\theta_{y_i} > \pi - m$ and the [miner weights](../losses.md#miner-weights)
are those of ArcFaceLoss. `get_logits(embeddings)` returns $s \cos\theta_j$
over the closest sub-centers, shaped `(N, num_classes)`; see
[classification logits](../losses.md#classification-logits). This loss derives
from ArcFaceLoss and [BaseProxyLoss](base_proxy_loss.md).

`get_outliers` finds outliers and dominant sub-centers as described in the paper:

```python
outliers, dominant_centers = loss_func.get_outliers(
    embeddings, labels, threshold=75, return_dominant_centers=True
)
```

`outliers` holds the indices of embeddings whose similarity to their class's
dominant sub-center is below `cos(threshold)`, with `threshold` in degrees.
`dominant_centers` has shape `(num_classes, embedding_size)`; a class's dominant
sub-center is the one nearest to the most of its embeddings in `embeddings`.
With `return_dominant_centers=False`, only `outliers` is returned. The method
runs without gradients.

**Parameters**:

- **num_classes**: The number of classes in the training set. Labels must be in
  `0..num_classes - 1`.
- **embedding_size**: The size of the embeddings passed to the loss.
- **margin**: The additive angular margin $m$ in radians. The paper uses 0.5.
- **scale**: The scale $s$. The paper uses 64.
- **sub_centers**: $K$, the number of sub-centers per class. The paper uses 3.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  sub-centers are `weight`, shaped `(num_classes * sub_centers, embedding_size)`
  and initialized with `torch.nn.init.normal_`; rows `c * sub_centers` through
  `(c + 1) * sub_centers - 1` belong to class `c`. Include
  `loss_func.parameters()` in an optimizer.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
