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

**Implementation notes**: The paper specifies the gradient $\partial L / \partial \mathbf{x}_i$ directly; this mean squared error between the cosines and the one-hot target produces it. The weight rows are renormalized to unit length under `torch.no_grad()` at every forward.

**Parameters**

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function.

Example usage:

```python
loss_fn = P2SGradLoss(10, 128)
loss = loss_fn(embeddings, labels)
```

The class weights are the `weight` parameter, shaped `(num_classes, embedding_size)`. This loss **requires an optimizer** that includes `loss_fn.parameters()`.

**Important notes**

`indices_tuple`, `ref_emb`, and `ref_labels` are not supported for this loss function.

**Default reducer**:

- This loss returns an **already reduced** loss.
