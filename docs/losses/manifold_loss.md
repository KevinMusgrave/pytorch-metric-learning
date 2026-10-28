# ManifoldLoss

[Ensemble Deep Manifold Similarity Learning using Hard Proxies](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aziere_Ensemble_Deep_Manifold_Similarity_Learning_Using_Hard_Proxies_CVPR_2019_paper.pdf)

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.ManifoldLoss(
        embedding_size: int,
        num_proxies: int = 50,
        lambda_c: float = 1.0,
        alpha: float = 0.8,
        margin: float = 5e-4,
        **kwargs
    )
```

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
