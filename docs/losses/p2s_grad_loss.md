# P2SGradLoss

[P2SGrad: Refined Gradients for Optimizing Deep Face Models](https://arxiv.org/abs/1905.02479)

```python {.signature}
losses.P2SGradLoss(
    num_classes,
    embedding_size,
    *,
    embedding_regularizer=None,
    embedding_reg_weight=1.0,
)
```

**Equation**:

$$
L = \frac{1}{NC} \sum_{i=1}^{N} \sum_{j=1}^{C} \left(\cos\theta_{i,j} - \mathbf{1}_{j = y_i}\right)^2
$$

Here $\cos\theta_{i,j}$ is the cosine between embedding $i$ and the unit-norm weight vector of class $j$, and $C$ is `num_classes`.

**Implementation notes**: The paper specifies the gradient
$\partial L / \partial \mathbf{x}_i$ directly; this mean squared error between
the cosines and the one-hot target produces it. The cosines are computed
directly from L2-normalized embeddings and weights. The weight rows are
renormalized to unit length in place, under `torch.no_grad()`, at every forward
pass. Labels are required; `indices_tuple` and `ref_emb` raise `ValueError`. An
empty batch gives a differentiable zero.

The class weights are the parameter `weight`, shaped
`(num_classes, embedding_size)` and initialized uniformly in $[-1, 1]$, then
normalized. Include `loss_func.parameters()` in an optimizer. The
[placement rule](../losses.md#learnable-weight-arguments) of learnable weights
applies; this loss does not derive from BaseProxyLoss, has no weight
regularization arguments, and has no `get_logits`.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.P2SGradLoss(num_classes=10, embedding_size=128)
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
- **embedding_regularizer**, **embedding_reg_weight**: See
  [loss options](../losses.md#loss-options).

**Distance**:

- None. The loss computes cosines directly, so it has no `distance` option;
  passing `distance` raises `TypeError`.

**Reducer**:

- None. The loss reduces its own value, so it has no `reducer` option; passing
  `reducer` raises `TypeError`.

**Loss terms**:

- **loss**: The value of $L$, with `indices=None`.
