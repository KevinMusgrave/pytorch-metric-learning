# SphereFaceLoss

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf){target=_blank}

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.SphereFaceLoss(num_classes, embedding_size, margin=4, scale=1,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Parameters**:

See [LargeMarginSoftmaxLoss](large_margin_softmax_loss.md#largemarginsoftmaxloss)

**Other info**

- Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.SphereFaceLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
