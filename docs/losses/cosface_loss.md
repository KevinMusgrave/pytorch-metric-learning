# CosFaceLoss

[CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf){target=_blank}

```python
losses.CosFaceLoss(num_classes, embedding_size, margin=0.35, scale=64,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
L_{lmc} = \frac{1}{N} \sum_{i} -\log \frac{e^{s \left(\cos\left(\theta_{y_i,i}\right) - m\right)}}{e^{s \left(\cos\left(\theta_{y_i,i}\right) - m\right)} + \sum_{j \neq y_i} e^{s \cos\left(\theta_{j,i}\right)}}
$$

Here $N$ is the batch size and $\theta_{j,i}$ is the angle between embedding $i$ and the weight vector of class $j$.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: The cosine margin penalty (m in the above equation). The paper used values between 0.25 and 0.45.
- **scale**: This is `s` in the above equation. The paper uses 64.

**Other info**:

- Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.CosFaceLoss(...).to(torch.device('cuda'))
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
