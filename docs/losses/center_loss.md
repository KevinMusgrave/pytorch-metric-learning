# CenterLoss

[A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)
(Wen et al.) keeps one center per class and pulls each embedding toward its class's
center. It reduces variation within a class and does not separate classes, so it is an
auxiliary term alongside a classification loss.

```python {.signature}
losses.CenterLoss(
    num_classes,
    embedding_size,
    alpha=0.5,
    *,
    reducer=None,
    embedding_regularizer=None,
    embedding_reg_weight=1.0,
)
```

**Equation**:

For embeddings $x_i$ with labels $y_i$ (equation 2):

$$
\mathcal{L}_C = \frac{1}{2} \sum_{i} \lVert x_i - c_{y_i} \rVert_2^2
$$

**Updating the centers**: The centers are not trained by an optimizer or by the loss's
gradient. They move only when you call `update_centers`:

```python {.signature}
loss_func.update_centers(embeddings, labels)
```

Call it once per batch. Without it, the centers keep their random initial values, and the
loss pulls every class toward a meaningless point. It applies equation 4 and step 6 of
algorithm 1 of the paper to each class $j$ in the batch, and leaves other classes alone:

$$
\Delta c_j = \frac{\sum_i \delta(y_i = j)\,(c_j - x_i)}{1 + \sum_i \delta(y_i = j)},
\qquad c_j \leftarrow c_j - \alpha \Delta c_j
$$

The denominator's $1 +$ makes this different from a gradient step on $\mathcal{L}_C$, and
it is why a class seen rarely moves less: one sample moves its center by $\alpha / 2$ of
the offset. The update runs under `torch.no_grad()`, so the embeddings keep their graph, and
it is safe before or after `backward()`. `forward` never changes the centers, so evaluation
and repeated forward passes leave them alone.

The centers are the buffer `loss_func.centers`, shaped `(num_classes, embedding_size)`. It
is saved in `state_dict`, moves with `.to()`, and is initialized from a standard normal
distribution, because the paper does not specify one. `loss_func.parameters()` is empty,
so the optimizer in the example does not include the loss. To start from known class
means, copy them into `loss_func.centers`.

**Implementation notes**: Labels must be integers in `[0, num_classes)`, and embeddings
must have `embedding_size` columns; otherwise the loss raises `ValueError`. Embeddings are
compared as passed, with no normalization and no `distance` argument, because equation 2
is defined on squared Euclidean distance. float16 and bfloat16 inputs are computed in
float32, and the loss is returned in float32, because a squared distance easily exceeds
the float16 range.

Each process updates its own centers, and neither `DistributedDataParallel` nor
[DistributedLossWrapper](../distributed.md) synchronizes them. To keep them equal, pass
the same gathered embeddings and labels to `update_centers` on every process.

The loss takes `labels` only. `indices_tuple` and `ref_emb` raise `ValueError`.

**Paper settings**: The joint objective is $\mathcal{L}_S + \lambda \mathcal{L}_C$
(equation 5). The paper fixes $\alpha = 0.5$ while varying $\lambda$ and fixes $\lambda = 0.003$
while varying $\alpha$. The paper sums both losses over the batch, while PyTorch's
`cross_entropy` and the default [MeanReducer](../reducers.md#meanreducer) both average, so the
same $\lambda$ applies, and the total is scaled by the batch size.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

model = torch.nn.Linear(16, 8)
classifier = torch.nn.Linear(8, 4)
center_loss = losses.CenterLoss(num_classes=4, embedding_size=8)
optimizer = torch.optim.SGD([*model.parameters(), *classifier.parameters()], lr=0.1)
data = torch.randn(32, 16)
labels = torch.arange(32) % 4

embeddings = model(data)
logits = classifier(embeddings)
loss = torch.nn.functional.cross_entropy(logits, labels)
loss = loss + 0.003 * center_loss(embeddings, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
center_loss.update_centers(embeddings, labels)
```

**Parameters**:

- **num_classes**: The number of class centers.
- **embedding_size**: The embedding dimension.
- **alpha**: The center learning rate $\alpha$. It must be in `[0, 1]`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
