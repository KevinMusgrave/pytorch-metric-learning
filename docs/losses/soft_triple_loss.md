# SoftTripleLoss

[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)

```python
losses.SoftTripleLoss(num_classes, embedding_size, centers_per_class=10, la=20, gamma=0.1, margin=0.01,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
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

**Implementation notes**: $\mathbf{x}_i^\top \mathbf{w}_c^k$ is the configured
similarity. With an `indices_tuple`, each embedding's loss is multiplied by its
[miner weight](../losses.md#miner-weights). `get_logits(embeddings)` returns
$\mathcal{S}'_{i,c}$, shaped `(N, num_classes)`, without $\lambda$ or
$\delta$; see [classification logits](../losses.md#classification-logits). This
loss derives from [BaseProxyLoss](base_proxy_loss.md).

**Paper settings**: The paper adds a center regularizer with weight
$\tau = 0.2$. Reproduce it with
`weight_regularizer=regularizers.SparseCentersRegularizer(num_classes, centers_per_class)`
and `weight_reg_weight=0.2`; it is not enabled by default. The other defaults
are the paper's values.

**Example**:

```python
import torch
from pytorch_metric_learning import losses, regularizers

loss_func = losses.SoftTripleLoss(
    num_classes=10,
    embedding_size=128,
    weight_regularizer=regularizers.SparseCentersRegularizer(10, 10),
    weight_reg_weight=0.2,
)
optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 10, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
optimizer.step()
```

**Parameters**:

- **num_classes**: The number of classes in the training set. Labels must be in
  `0..num_classes - 1`.
- **embedding_size**: The size of the embeddings passed to the loss.
- **centers_per_class**: The number of centers per class. An ordinary softmax
  has 1. The paper uses 10.
- **la**: The scale $\lambda$. The paper uses 20.
- **gamma**: The temperature $\gamma$ of the softmax over a class's centers.
  The paper uses 0.1.
- **margin**: The margin $\delta$. The paper uses 0.01.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  centers are `weight`, shaped `(num_classes * centers_per_class, embedding_size)`
  and initialized with
  `torch.nn.init.kaiming_uniform_(a=math.sqrt(5), mode="fan_out")`; rows
  `c * centers_per_class` through `(c + 1) * centers_per_class - 1` belong to
  class `c`. Include `loss_func.parameters()` in an optimizer.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). The distance must
  be a similarity (`is_inverted=True`), such as
  [`DotProductSimilarity()`](../distances.md#dotproductsimilarity); otherwise
  the constructor raises `ValueError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
