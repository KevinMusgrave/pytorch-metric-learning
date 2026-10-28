# ProxyNCALoss

[No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf){target=_blank}

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.ProxyNCALoss(num_classes, embedding_size, softmax_scale=1,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **softmax_scale**: See [NCALoss](nca_loss.md#ncaloss)

**Other info**

- Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.ProxyNCALoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

Under autocast, embeddings are converted to the proxies' dtype before
normalization and distance computation. This applies to both the loss and
`get_logits`, preserves encoder gradients, and leaves parameter placement and
optimizer state unchanged.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch, that results in a non zero exponent in the cross entropy expression. Reduction type is `"element"`.
