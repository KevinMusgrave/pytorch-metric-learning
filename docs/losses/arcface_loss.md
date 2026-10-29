# ArcFaceLoss

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

```python {.signature}
losses.ArcFaceLoss(num_classes, embedding_size, margin=0.5, scale=64,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cos\left(\theta_{y_i} + m\right)}}{e^{s \cos\left(\theta_{y_i} + m\right)} + \sum_{j=1, j \neq y_i}^{n} e^{s \cos \theta_j}}
$$

Here $N$ is the batch size, $n$ is `num_classes`, and $\theta_j$ is the angle between an embedding and the weight vector of class $j$.

**Implementation notes**: When $\theta_{y_i} > \pi - m$, the target logit is
$s(\cos\theta_{y_i} - m \sin m)$, which keeps it decreasing in $\theta$. With an
`indices_tuple`, each embedding's loss is multiplied by its
[miner weight](../losses.md#miner-weights). `get_logits(embeddings)` returns
$s \cos\theta_j$ without the margin; see
[classification logits](../losses.md#classification-logits). This loss derives
from [BaseProxyLoss](base_proxy_loss.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.ArcFaceLoss(num_classes=10, embedding_size=128)
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
- **margin**: The additive angular margin $m$ in radians. The paper uses 0.5.
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
