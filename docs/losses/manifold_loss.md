# ManifoldLoss

[Ensemble Deep Manifold Similarity Learning using Hard Proxies](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aziere_Ensemble_Deep_Manifold_Similarity_Learning_Using_Hard_Proxies_CVPR_2019_paper.pdf)

```python
losses.ManifoldLoss(
    embedding_size, num_proxies=50, lambda_c=1.0, alpha=0.8, margin=5e-4, **options
)
```

**Equation**:

$$
F = (1 - \alpha)\left(I - \alpha \bar S\right)^{-1}, \qquad \bar S = D^{-1/2} S D^{-1/2}, \qquad S_{ij} = e^{\max(\cos(\mathbf{x}_i, \mathbf{x}_j), 0) / 0.5} \; (i \neq j)
$$

$$
L_{int} = \frac{1}{N} \sum_{i=1}^{N} \log\left(1 + \sum_{k \neq c(i)} e^{\left[F_{i,k} - F_{i,c(i)} + m\right]_+}\right),
\qquad
L_{ctx} = \frac{1}{N} \sum_{i=1}^{N} \log\left(1 + \sum_{k \neq c(i)} e^{\left[\cos(F_i, F_{p_k}) - \cos(F_i, F_{p_{c(i)}}) + m\right]_+}\right)
$$

$$
L = L_{int} + \lambda_C L_{ctx}
$$

Here $S$ is the similarity graph over the $N$ embeddings and $K$ proxies, $D$ is its degree matrix, $F_{i,k}$ is the random-walk affinity between embedding $i$ and proxy $k$, $c(i)$ is the proxy assigned to embedding $i$, and only violating terms with $[\cdot]_+ > 0$ enter the sums.

**Implementation notes**: `labels` holds the proxy assignments $c(i)$, integers
in `0..num_proxies - 1`; they are not class labels. With `labels=None`, every
proxy is assigned to at least one embedding and the remaining embeddings
receive random proxies, which requires a batch of at least `num_proxies`
embeddings; a smaller unlabeled batch raises `ValueError`. Labels must lie in
`[0, num_proxies)`. `indices_tuple`, `ref_emb`, and `ref_labels` raise `ValueError`,
as does an embedding dimension other than `embedding_size`. The matrix inverse
is computed in float32.

The proxies are the parameter `proxies`, shaped `(num_proxies, embedding_size)`
and initialized with `torch.randn`. Include `loss_func.parameters()` in an
optimizer. The [placement rule](../losses.md#learnable-weight-arguments) of
learnable weights applies; this loss does not derive from BaseProxyLoss and has
no weight regularization arguments.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.ManifoldLoss(embedding_size=128, num_proxies=10)
optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
embeddings = torch.randn(32, 128, requires_grad=True)
loss = loss_func(embeddings)
loss.backward()
proxy_ids = torch.randint(0, 10, (32,))
loss = loss_func(embeddings, proxy_ids)
loss.backward()
optimizer.step()
```

**Parameters**:

- **embedding_size**: The embedding size, $l$ in the paper.
- **num_proxies**: The number of proxies, $K$ in the paper.
- **lambda_c**: The weight $\lambda_C$ of the context loss. It must be
  non-negative; a negative value raises `ValueError`. `lambda_c=0` uses only the
  intrinsic loss, and `lambda_c=math.inf` uses only the context loss.
- **alpha**: The random-walk parameter $\alpha$, in the open interval `(0, 1)`.
  It controls how much similarity propagates between neighboring nodes.
- **margin**: The margin $m$.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: Already reduced: the value of $L$. Every reducer returns it unchanged.
