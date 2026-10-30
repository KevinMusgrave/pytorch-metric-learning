# NormalizedSoftmaxLoss

[Classification is a Strong Baseline for Deep Metric Learning](https://arxiv.org/abs/1811.12649)

```python {.signature}
losses.NormalizedSoftmaxLoss(num_classes, embedding_size, temperature=0.05,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
L_{\mathrm{norm}} = -\log\left(\frac{\exp\left(x^{T} p_{y} / \sigma\right)}{\sum_{z \in Z} \exp\left(x^{T} p_{z} / \sigma\right)}\right)
$$

Here $x$ is the L2-normalized embedding with label $y$, $p_z$ is the L2-normalized weight vector of class $z$, and $Z$ is the set of all classes.

**Implementation notes**: The class weights are L2-normalized at every forward
pass; the embeddings are normalized by the default distance. With a distance
where smaller means closer, the logits are negated distances. With an
`indices_tuple`, each embedding's loss is multiplied by its
[miner weight](../losses.md#miner-weights). `get_logits(embeddings)` returns
the logits $x^T p_z / \sigma$; see
[classification logits](../losses.md#classification-logits). This loss derives
from [BaseProxyLoss](base_proxy_loss.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.NormalizedSoftmaxLoss(num_classes=10, embedding_size=128)
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
- **temperature**: The temperature $\sigma$. The paper uses 0.05.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  class weights are `weight`, shaped `(num_classes, embedding_size)` and
  initialized with `torch.nn.init.normal_`. Include `loss_func.parameters()` in
  an optimizer.

**Default distance**:

- [`DotProductSimilarity()`](../distances.md#dotproductsimilarity), which
  L2-normalizes the embeddings.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per embedding, indexed by embedding position.
