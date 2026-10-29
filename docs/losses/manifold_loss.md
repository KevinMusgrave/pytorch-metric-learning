# ManifoldLoss

[Ensemble Deep Manifold Similarity Learning using Hard Proxies](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aziere_Ensemble_Deep_Manifold_Similarity_Learning_Using_Hard_Proxies_CVPR_2019_paper.pdf)

```python
losses.ManifoldLoss(
        embedding_size: int,
        num_proxies: int = 50,
        lambda_c: float = 1.0,
        alpha: float = 0.8,
        margin: float = 5e-4,
        **options
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

**Implementation notes**: The matrix inverse is computed in float32. With `indices_tuple=None`, every proxy is assigned to at least one embedding and the remaining embeddings receive random proxies.

**Parameters**

- **embedding_size**: embedding size, $l$ in the paper.

- **num_proxies**: number of proxies, $K$ in the paper.

- **lambda_c**: regularization weight $\lambda_C$ in `loss = intrinsic_loss + lambda_c * context_loss`. `lambda_c=0` uses only the intrinsic loss, and `lambda_c=math.inf` uses only the context loss.

- **alpha**: parameter of the Random Walk. Must be in the range `(0,1)`. It specifies the amount of similarity between neighboring nodes.

- **margin**: margin used in the calculation of the loss.

Example usage:

```python
loss_fn = ManifoldLoss(128)

# use random cluster centers
loss = loss_fn(embeddings)
# or specify indices of embeddings to use as cluster centers
loss = loss_fn(embeddings, indices_tuple=indices)
```

**Important notes**

`labels`, `ref_emb`, and `ref_labels` are not supported for this loss function.

In addition, `indices_tuple` is **not** for the output of miners. Instead, it is for a list of indices of embeddings to be used as cluster centers.

**Default reducer**:

- This loss returns an **already reduced** loss.
