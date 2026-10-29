# BaseMetricLossFunction

All loss functions extend this class. Its keyword-only parameters are the
[loss options](../losses.md#loss-options) that every loss accepts as `**options`.

```python
losses.BaseMetricLossFunction(*, distance=None, reducer=None, embedding_regularizer=None, embedding_reg_weight=1.0)
```

**Parameters**:

- **distance**: A [distance](../distances.md) object. If None, then the default distance will be used.
- **reducer**: A [reducer](../reducers.md) object. If None, then the default reducer will be used.
- **embedding_regularizer**: A [regularizer](../regularizers.md) object that will be applied to embeddings. If None, then no embedding regularization will be used.
- **embedding_reg_weight**: If an embedding regularizer is used, then its loss will be multiplied by this amount before being added to the reduced loss.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Regularization**: `embedding_regularizer` and `embedding_reg_weight` are
described in [regularization terms](../losses.md#regularization-terms).

**Forward**:

```python
loss_func(embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None)
```

Returns a scalar tensor. `loss_func.loss_terms(...)` takes the same arguments
and returns the [unreduced terms](../losses.md#unreduced-loss-terms).

**Required Implementations**:

```python
def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
    raise NotImplementedError
```

Override `default_distance()`, `default_reducer()`, and the `sub_loss_names`
class attribute to change the defaults; see [custom losses](../extend/losses.md).
