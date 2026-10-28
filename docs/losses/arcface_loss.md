# ArcFaceLoss

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf){target=_blank}

```python
losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cos\left(\theta_{y_i} + m\right)}}{e^{s \cos\left(\theta_{y_i} + m\right)} + \sum_{j=1, j \neq y_i}^{n} e^{s \cos \theta_j}}
$$

Here $N$ is the batch size, $n$ is `num_classes`, and $\theta_j$ is the angle between an embedding and the weight vector of class $j$.

**Implementation notes**: When $\theta_{y_i} > \pi - m$, the target logit is $\cos\theta_{y_i} - m \sin m$, which keeps it decreasing in $\theta$. The default margin of 28.6 degrees is 0.4992 radians; pass `margin=28.648` for the paper's 0.5. With an `indices_tuple`, each element's loss is weighted by its relative mining frequency.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: The angular margin penalty in degrees. In the above equation, `m = radians(margin)`. The paper uses 0.5 radians, which is 28.6 degrees.
- **scale**: This is `s` in the above equation. The paper uses 64.

**Other info**:

- Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.ArcFaceLoss(...).to(torch.device('cuda'))
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
