# ProxyAnchorLoss

[Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/abs/2003.13911)

```python
losses.ProxyAnchorLoss(num_classes, embedding_size, margin=0.1, alpha=32,
    *, weight_regularizer=None, weight_reg_weight=1.0, weight_init_func=None, **options)
```

**Equation**:

$$
\begin{aligned}
\ell(X) &= \frac{1}{|P^+|} \sum_{p \in P^+} \log\left(1 + \sum_{x \in X_p^+} w_x \, e^{-\alpha\left(s(x, p) - \delta\right)}\right) \\
&\quad + \frac{1}{|P|} \sum_{p \in P} \log\left(1 + \sum_{x \in X_p^-} w_x \, e^{\alpha\left(s(x, p) + \delta\right)}\right)
\end{aligned}
$$

Here $P$ is the set of all proxies, $P^+$ the proxies whose class is in the batch, $X_p^+$ and $X_p^-$ the batch embeddings with and without proxy $p$'s class, $s(x, p)$ the cosine similarity, and $w_x$ the [miner weight](../losses.md#miner-weights) of embedding $x$. Without an `indices_tuple`, every $w_x = 1$ and the equation is the paper's.

**Implementation notes**: Each sum is computed as a log-sum-exp with
$\log w_x$ added to the exponent, so a miner weight multiplies the
embedding's exponential term and an embedding with weight 0 is excluded from
both sums. A proxy with no remaining embeddings contributes 0. With a distance
where smaller means closer, $s(x, p) - \delta$ and $s(x, p) + \delta$ become
$\delta - d(x, p)$ and $-\delta - d(x, p)$.

Version 2 of this library multiplied each exponential term by $e^{w_x - 1}$
instead of $w_x$. That kept an unmined embedding in the sums with factor
$e^{-1}$ and compressed the ratio between frequently and rarely mined
embeddings. Multiplying by $w_x$ makes each embedding's contribution
proportional to its mining frequency and makes mined-out embeddings contribute
nothing, while unweighted batches give the same value in both versions.

`get_logits(embeddings)` returns the `(N, num_classes)` similarity matrix
between embeddings and proxies; see
[classification logits](../losses.md#classification-logits). This loss derives
from [BaseProxyLoss](base_proxy_loss.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.ProxyAnchorLoss(num_classes=10, embedding_size=128)
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
- **margin**: The margin $\delta$. The paper uses 0.1.
- **alpha**: The scale $\alpha$. The paper uses 32.
- **weight_regularizer**, **weight_reg_weight**, **weight_init_func**: See
  [learnable weight arguments](../losses.md#learnable-weight-arguments). The
  proxies are `weight`, shaped `(num_classes, embedding_size)` and initialized
  with `torch.nn.init.kaiming_normal_(mode="fan_out")`. Include
  `loss_func.parameters()` in an optimizer.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [DivisorReducer](../reducers.md#divisorreducer). Both terms carry a divisor,
  so they must reach DivisorReducer, directly or through MultipleReducers; any
  other reducer raises `ValueError`.

**Reducer input**:

- **pos_loss**: One loss per proxy, indexed by proxy (class) index, with
  `divisor` $|P^+|$, the number of proxies whose class is in the batch.
- **neg_loss**: One loss per proxy, indexed by proxy (class) index, with
  `divisor` `num_classes`.
