# P2SGradLoss

[P2SGrad: Refined Gradients for Optimizing Deep Face Models](https://arxiv.org/abs/1905.02479)

```python
losses.P2SGradLoss(num_classes, embedding_size, **options)
```

**Equation**:

$$
L = \frac{1}{NC} \sum_{i=1}^{N} \sum_{j=1}^{C} \left(\cos\theta_{i,j} - \mathbf{1}_{j = y_i}\right)^2
$$

Here $\cos\theta_{i,j}$ is the cosine between embedding $i$ and the unit-norm weight vector of class $j$, and $C$ is `num_classes`.

**Implementation notes**: The paper specifies the gradient
$\partial L / \partial \mathbf{x}_i$ directly; this mean squared error between
the cosines and the one-hot target produces it. The cosines are computed
directly from L2-normalized embeddings and weights, so the `distance` option
has no effect. The weight rows are renormalized to unit length in place, under
`torch.no_grad()`, at every forward pass. Labels are required; `indices_tuple`
and `ref_emb` raise `ValueError`.

The class weights are the parameter `weight`, shaped
`(num_classes, embedding_size)` and initialized uniformly in $[-1, 1]$, then
normalized. Include `loss_func.parameters()` in an optimizer. The
[placement rule](../losses.md#learnable-weight-arguments) of learnable weights
applies; this loss does not derive from BaseProxyLoss, has no weight
regularization arguments, and has no `get_logits`.

**Parameters**:

- **num_classes**: The number of classes in the training set. Labels must be in
  `0..num_classes - 1`.
- **embedding_size**: The size of the embeddings passed to the loss.

**Default distance**:

- Not used. The loss computes cosines directly, and a `distance` option has no
  effect.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: Already reduced: the value of $L$. Every reducer returns it unchanged.

```python
loss_func = losses.P2SGradLoss(num_classes=10, embedding_size=128)
loss = loss_func(embeddings, labels)
```
