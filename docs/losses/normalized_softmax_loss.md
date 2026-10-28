# NormalizedSoftmaxLoss

[Classification is a Strong Baseline for Deep Metric Learning](https://arxiv.org/pdf/1811.12649.pdf){target=_blank}

```python
losses.NormalizedSoftmaxLoss(num_classes, embedding_size, temperature=0.05,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
L_{\mathrm{norm}} = -\log\left(\frac{\exp\left(x^{T} p_{y} / \sigma\right)}{\sum_{z \in Z} \exp\left(x^{T} p_{z} / \sigma\right)}\right)
$$

Here $x$ is the L2-normalized embedding with label $y$, $p_z$ is the L2-normalized weight vector of class $z$, and $Z$ is the set of all classes.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **temperature**: This is sigma in the above equation. The paper uses 0.05.

**Other info**

- Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.NormalizedSoftmaxLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`DotProductSimilarity()`](../distances.md#dotproductsimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
