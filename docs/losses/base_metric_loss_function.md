# BaseMetricLossFunction

All loss functions extend this class and therefore inherit its `__init__` parameters.

```python
losses.BaseMetricLossFunction(reducer = None,
							distance = None,
							embedding_regularizer = None,
							embedding_reg_weight = 1)
```

**Parameters**:

- **reducer**: A [reducer](../reducers.md) object. If None, then the default reducer will be used.
- **distance**: A [distance](../distances.md) object. If None, then the default distance will be used.
- **embedding_regularizer**: A [regularizer](../regularizers.md) object that will be applied to embeddings. If None, then no embedding regularization will be used.
- **embedding_reg_weight**: If an embedding regularizer is used, then its loss will be multiplied by this amount before being added to the reduced loss.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Regularization**: `embedding_regularizer` and `embedding_reg_weight` are
described in [regularization terms](../losses.md#regularization-terms).

**Required Implementations**:

```python
def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
    raise NotImplementedError
```
