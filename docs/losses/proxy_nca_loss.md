# ProxyNCALoss

[No Fuss Distance Metric Learning using Proxies](https://arxiv.org/abs/1703.07464)

```python
losses.ProxyNCALoss(num_classes, embedding_size, softmax_scale=1,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} w_i \log \frac{\exp\left(-s \, d(\mathbf{x}_i, \mathbf{p}_{y_i})\right)}{\sum_{z} \exp\left(-s \, d(\mathbf{x}_i, \mathbf{p}_z)\right)}
$$

Here $\mathbf{p}_z$ is the proxy of class $z$, $d$ is the squared Euclidean distance between L2-normalized vectors, $s$ is `softmax_scale`, and $w_i$ is the [miner weight](../losses.md#miner-weights) of embedding $i$, 1 without an `indices_tuple`.

**Implementation notes**: The denominator sums over every proxy, including the
target proxy, so the loss is the softmax cross-entropy over proxies. The paper's
equation excludes the target proxy from the denominator. The loss is computed
in log space with log-sum-exp; float16 and bfloat16 inputs accumulate in
float32. With a similarity distance, the exponents are $s \cdot
\mathrm{sim}(\mathbf{x}_i, \mathbf{p}_z)$. One proxy per class and normalized
embeddings and proxies are the paper's settings and the defaults here.

Under autocast, embeddings are converted to the proxies' dtype before
normalization and distance computation. This applies to both the loss and
`get_logits`, preserves encoder gradients, and leaves parameter placement and
optimizer state unchanged. `get_logits(embeddings)` returns the
`(N, num_classes)` distance matrix between embeddings and proxies, so with the
default distance the closest class has the smallest value; see
[classification logits](../losses.md#classification-logits). This loss derives
from [BaseProxyLoss](base_proxy_loss.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.ProxyNCALoss(num_classes=10, embedding_size=128)
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
- **softmax_scale**: The multiplier $s$ of the distances in the softmax. The
  paper uses 1.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  proxies are `weight`, shaped `(num_classes, embedding_size)` and initialized
  with `torch.nn.init.normal_`. Include `loss_func.parameters()` in an
  optimizer.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
