# SphereFaceLoss

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

```python
losses.SphereFaceLoss(num_classes, embedding_size, margin=4, scale=1,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
L_i = -\log\left(\frac{e^{\lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})}}{e^{\lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})} + \sum_{j \neq y_i} e^{\lVert \mathbf{x}_i \rVert \cos(\theta_j)}}\right)
$$

with $\psi$ as in [LargeMarginSoftmaxLoss](large_margin_softmax_loss.md). The weight vectors are L2-normalized, so only the embedding norm scales the logits.

**Implementation notes**: The paper's annealing with a decreasing $\lambda$ is
not implemented. Every logit is multiplied by `scale`. With an `indices_tuple`,
each embedding's loss is multiplied by its
[miner weight](../losses.md#miner-weights). `get_logits(embeddings)` returns
the logits with $\cos\theta_j$ in place of $\psi$; see
[classification logits](../losses.md#classification-logits). This loss derives
from LargeMarginSoftmaxLoss and [BaseProxyLoss](base_proxy_loss.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.SphereFaceLoss(num_classes=10, embedding_size=128)
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
- **margin**: The integer angular margin $m$. A value that is not a positive
  `int` raises `ValueError`. The paper uses 4.
- **scale**: A multiplier of every logit. The paper uses 1.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  class weights are `weight`, shaped `(num_classes, embedding_size)` and
  initialized with `torch.nn.init.normal_`. Include `loss_func.parameters()` in
  an optimizer.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity). This is the only
  compatible distance; another distance class raises `TypeError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
