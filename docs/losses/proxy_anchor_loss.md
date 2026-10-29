# ProxyAnchorLoss

[Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/pdf/2003.13911.pdf){target=_blank}

```python
losses.ProxyAnchorLoss(num_classes, embedding_size, margin = 0.1, alpha = 32,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **options)
```

**Equation**:

$$
\begin{aligned}
\ell(X) &= \frac{1}{|P^+|} \sum_{p \in P^+} \log\left(1 + \sum_{x \in X_p^+} e^{-\alpha\left(s(x, p) - \delta\right)}\right) \\
&\quad + \frac{1}{|P|} \sum_{p \in P} \log\left(1 + \sum_{x \in X_p^-} e^{\alpha\left(s(x, p) + \delta\right)}\right)
\end{aligned}
$$

Here $P$ is the set of all proxies, $P^+$ the proxies whose class is in the batch, $X_p^+$ and $X_p^-$ the batch embeddings with and without proxy $p$'s class, and $s(x, p)$ the cosine similarity.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: This is delta in the above equation. The paper uses 0.1.
- **alpha**: This is alpha in the above equation. The paper uses 32.

**Other info**

- The proxies are the `weight` parameter, shaped `(num_classes, embedding_size)`. Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.ProxyAnchorLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [DivisorReducer](../reducers.md#divisorreducer)

**Reducer input**:

- **pos_loss**: The positive pair loss per proxy. Reduction type is `"element"`.
- **neg_loss**: The negative pair loss per proxy. Reduction type is `"element"`.
