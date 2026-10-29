# NCALoss

[Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf){target=_blank}

```python
losses.NCALoss(softmax_scale=1, **options)
```

**Equation**:

$$
g(A) = \sum_i \log\left(\sum_{j \in C_i} p_{ij}\right) = \sum_i \log(p_i)
$$

where

$$
p_i = \sum_{j \in C_i} p_{ij}
$$

$$
p_{ij} = \frac{\exp\left(-\lVert A x_i - A x_j \rVert^2\right)}{\sum_{k \neq i} \exp\left(-\lVert A x_i - A x_k \rVert^2\right)}
$$

Here $A x_i$ is the embedding of element $i$, $C_i$ is the set of elements with the same label as $i$, and $p_{ii} = 0$.

In this implementation, we use `-g(A)` as the loss.

**Implementation notes**: The loss is averaged over elements with $p_i > 0$; elements with no same-label reference are skipped. The default distance L2-normalizes embeddings, unlike the paper's linear map.

**Parameters**:

- **softmax_scale**: The exponent multiplier in the loss's softmax expression. The paper uses `softmax_scale = 1 `, which is why it does not appear in the above equations.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch, that results in a non zero exponent in the cross entropy expression. Reduction type is `"element"`.
