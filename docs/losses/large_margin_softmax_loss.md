# LargeMarginSoftmaxLoss

[Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf){target=_blank}

```python
losses.LargeMarginSoftmaxLoss(num_classes, embedding_size, margin=4, scale=1,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
L_i = -\log\left(\frac{e^{\lVert \mathbf{W}_{y_i} \rVert \lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})}}{e^{\lVert \mathbf{W}_{y_i} \rVert \lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})} + \sum_{j \neq y_i} e^{\lVert \mathbf{W}_j \rVert \lVert \mathbf{x}_i \rVert \cos(\theta_j)}}\right)
$$

where

$$
\psi(\theta) = (-1)^k \cos(m\theta) - 2k, \quad \theta \in \left[\frac{k\pi}{m}, \frac{(k+1)\pi}{m}\right], \quad k \in \lbrace 0, 1, \ldots, m - 1 \rbrace
$$

$$
\cos(m\theta) = \sum_{n=0}^{\lfloor m/2 \rfloor} (-1)^n \binom{m}{2n} \cos^{m-2n}(\theta) \left(1 - \cos^2(\theta)\right)^n
$$

Here $\theta_j$ is the angle between $\mathbf{W}_j$ and $\mathbf{x}_i$. The implementation computes $\cos(m\theta)$ from $\cos\theta$ with this expansion, equations 6 and 7 of the paper, and treats $k$ as a constant for gradients.

**Implementation notes**: The paper's annealing of the target logit with a decreasing $\lambda$ is not implemented; schedule a combination with ordinary softmax in application code if needed.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: An integer which dictates the size of the angular margin. This is `m` in the above equation. The paper finds `m=4` works best.
- **scale**: The exponent multiplier in the loss's softmax expression. The paper uses `scale = 1 `, which is why it does not appear in the above equation.

**Other info**:

- Accepts the keyword-only [learnable weight arguments](../losses.md#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.LargeMarginSoftmaxLoss(...).to(torch.device('cuda'))
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
