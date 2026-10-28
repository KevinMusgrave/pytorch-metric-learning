# VICRegLoss

[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/pdf/2105.04906.pdf){target=_blank}

```python
losses.VICRegLoss(invariance_lambda=25,
                variance_mu=25,
                covariance_v=1,
                eps=1e-4,
                **kwargs)
```

**Usage**:

Unlike other loss functions, `VICRegLoss` does not accept `labels` or `indices_tuple`:

```python
loss_fn = VICRegLoss()
loss = loss_fn(embeddings, ref_emb=ref_emb)
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

**Implementation notes**: The invariance term is a mean squared error over both rows and dimensions, $\frac{1}{nd}\sum_i \lVert z_i - z'_i \rVert_2^2$, as in the paper's pseudocode and the authors' implementation. The variance term is $\frac{\mu}{2}\left[v(Z) + v(Z')\right]$, also following the authors' implementation; $\gamma$ is fixed at 1.

**Parameters**:

- **invariance_lambda**: The weight of the invariance term.
- **variance_mu**: The weight of the variance term.
- **covariance_v**: The weight of the covariance term.
- **eps**: Small scalar to prevent numerical instability.

**Default distance**:

- Not applicable. You cannot pass in a distance function.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **invariance_loss**: The MSE loss between `embeddings[i]` and `ref_emb[i]`. Reduction type is `"element"`.
- **variance_loss1**: The variance loss for `embeddings`. Reduction type is `"element"`.
- **variance_loss2**: The variance loss for `ref_emb`. Reduction type is `"element"`.
- **covariance_loss**: The covariance loss. This loss is already reduced to a single value.
