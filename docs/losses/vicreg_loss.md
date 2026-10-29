# VICRegLoss

[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)

```python
losses.VICRegLoss(invariance_lambda=25, variance_mu=25, covariance_v=1, eps=1e-4)
```

**Equation**:

$$
\ell(Z, Z') = \lambda s(Z, Z') + \mu\left[v(Z) + v(Z')\right] + \nu\left[c(Z) + c(Z')\right]
$$

Here $\lambda$, $\mu$, and $\nu$ are `invariance_lambda`, `variance_mu`, and `covariance_v`; $n$ is the batch size and $d$ the embedding size.

where

$$
s(Z, Z') = \frac{1}{n} \sum_{i} \left\lVert z_i - z'_i \right\rVert_2^2
$$

$$
v(Z) = \frac{1}{d} \sum_{j=1}^{d} \max\left(0, \gamma - S\left(z^j, \epsilon\right)\right)
$$

Here $z^j$ is the $j$-th embedding dimension across the batch, $\gamma = 1$, and $\epsilon$ is `eps`.

$$
S(x, \epsilon) = \sqrt{\operatorname{Var}(x) + \epsilon}
$$

$$
c(Z) = \frac{1}{d} \sum_{i \neq j} \left[C(Z)\right]_{i,j}^2
$$

Here $C(Z)$ is the covariance matrix of $Z$, normalized by $n - 1$.

**Implementation notes**: `VICRegLoss` is a plain `torch.nn.Module` that takes
two embedding batches of equal shape `(n, d)`, where `ref_emb[i]` is a second
view of `embeddings[i]`. It has no labels, indices, distance, reducer, or
regularizer options. Inputs that are not two-dimensional or that differ in
shape raise `ValueError`. The invariance term is a mean squared error over both
rows and dimensions, $\frac{1}{nd}\sum_i \lVert z_i - z'_i \rVert_2^2$, as in
the paper's pseudocode and the authors' implementation. The variance term is
$\frac{\mu}{2}\left[v(Z) + v(Z')\right]$, also following the authors'
implementation; $\gamma$ is fixed at 1.

**Paper settings**: The defaults $\lambda = 25$, $\mu = 25$, $\nu = 1$ are the
paper's.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.VICRegLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
ref_emb = torch.randn(32, 128, requires_grad=True)
loss = loss_func(embeddings, ref_emb)
loss.backward()
```

**Parameters**:

- **invariance_lambda**: The weight $\lambda$ of the invariance term.
- **variance_mu**: The weight $\mu$ of the variance term.
- **covariance_v**: The weight $\nu$ of the covariance term.
- **eps**: The constant $\epsilon$ added to the variance for numerical stability.
