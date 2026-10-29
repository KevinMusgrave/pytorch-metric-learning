# SoftTripleLoss

[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf){target=_blank}

```python
losses.SoftTripleLoss(num_classes, embedding_size, centers_per_class=10, la=20, gamma=0.1, margin=0.01,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **options)
```

**Equation**:

$$
\ell_{\operatorname{SoftTriple}}(\mathbf{x}_i) = -\log \frac{\exp\left(\lambda\left(\mathcal{S}'_{i,y_i}-\delta\right)\right)}{\exp\left(\lambda\left(\mathcal{S}'_{i,y_i}-\delta\right)\right)+\sum_{j \neq y_i}\exp\left(\lambda \mathcal{S}'_{i,j}\right)}
$$

where

$$
\mathcal{S}'_{i,c} = \sum_{k} \frac{\exp\left(\frac{1}{\gamma}\mathbf{x}_i^\top \mathbf{w}_c^k\right)}{\sum_{k'}\exp\left(\frac{1}{\gamma}\mathbf{x}_i^\top \mathbf{w}_c^{k'}\right)} \mathbf{x}_i^\top \mathbf{w}_c^k
$$

Here $\mathbf{w}_c^k$ is the $k$-th of the `centers_per_class` centers of class $c$; `CosineSimilarity` L2-normalizes both $\mathbf{x}_i$ and $\mathbf{w}_c^k$.

**Implementation notes**: The paper adds a center regularizer with weight $\tau = 0.2$. Reproduce it with `weight_regularizer=regularizers.SparseCentersRegularizer(num_classes, centers_per_class)` and `weight_reg_weight=0.2`; it is not enabled by default.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **centers_per_class**: The number of weight vectors per class. (The regular cross entropy loss has 1 center per class.) The paper uses 10.
- **la**: This is lambda in the above equation.
- **gamma**: This is gamma in the above equation. The paper uses 0.1.
- **margin**: This is delta in the above equations. The paper uses 0.01.

**Other info**

- Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.SoftTripleLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)
  - The distance measure must be inverted. For example, [`DotProductSimilarity(normalize_embeddings=False)`](../distances.md#dotproductsimilarity) is also compatible.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
