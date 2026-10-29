# CosFaceLoss

[CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)

```python
losses.CosFaceLoss(num_classes, embedding_size, margin=0.35, scale=64,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
L_{lmc} = \frac{1}{N} \sum_{i} -\log \frac{e^{s \left(\cos\left(\theta_{y_i,i}\right) - m\right)}}{e^{s \left(\cos\left(\theta_{y_i,i}\right) - m\right)} + \sum_{j \neq y_i} e^{s \cos\left(\theta_{j,i}\right)}}
$$

Here $N$ is the batch size and $\theta_{j,i}$ is the angle between embedding $i$ and the weight vector of class $j$.

**Implementation notes**: With an `indices_tuple`, each embedding's loss is
multiplied by its [miner weight](../losses.md#miner-weights).
`get_logits(embeddings)` returns $s \cos\theta_{j,i}$ without the margin; see
[classification logits](../losses.md#classification-logits). This loss derives
from [BaseProxyLoss](base_proxy_loss.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.CosFaceLoss(num_classes=10, embedding_size=128)
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
- **margin**: The cosine margin $m$. The paper uses values between 0.25 and 0.45.
- **scale**: The scale $s$. The paper uses 64.
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
