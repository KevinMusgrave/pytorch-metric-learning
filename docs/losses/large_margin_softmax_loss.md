# LargeMarginSoftmaxLoss

[Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/abs/1612.02295)

```python
losses.LargeMarginSoftmaxLoss(num_classes, embedding_size, margin=4, scale=1,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
L_i = -\log\left(\frac{e^{\lVert \mathbf{W}_{y_i} \rVert \lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})}}{e^{\lVert \mathbf{W}_{y_i} \rVert \lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})} + \sum_{j \neq y_i} e^{\lVert \mathbf{W}_j \rVert \lVert \mathbf{x}_i \rVert \cos(\theta_j)}}\right)
$$

where

$$
\psi(\theta) = (-1)^k \cos(m\theta) - 2k, \quad \theta \in \left[\frac{k\pi}{m}, \frac{(k+1)\pi}{m}\right], \quad k \in \lbrace 0, 1, \ldots, m - 1 \rbrace
$$

$$
\cos(m\theta) = \sum_{n=0}^{\lfloor m/2 \rfloor} (-1)^n \binom{m}{2n} \cos^{m-2n}(\theta) \left(1 - \cos^2(\theta)\right)^n
$$

Here $\theta_j$ is the angle between $\mathbf{W}_j$ and $\mathbf{x}_i$. The implementation computes $\cos(m\theta)$ from $\cos\theta$ with this expansion, equations 6 and 7 of the paper, and treats $k$ as a constant for gradients.

**Implementation notes**: The paper's annealing of the target logit with a
decreasing $\lambda$ is not implemented; schedule a combination with an ordinary
softmax loss in application code if needed. Every logit is multiplied by
`scale`. With an `indices_tuple`, each embedding's loss is multiplied by its
[miner weight](../losses.md#miner-weights). `get_logits(embeddings)` returns
the logits with $\cos\theta_j$ in place of $\psi$; see
[classification logits](../losses.md#classification-logits). This loss derives
from [BaseProxyLoss](base_proxy_loss.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.LargeMarginSoftmaxLoss(num_classes=10, embedding_size=128)
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
  `int` raises `ValueError`. The paper finds `m=4` works best.
- **scale**: A multiplier of every logit. The paper uses 1, so it does not
  appear in the equation.
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
