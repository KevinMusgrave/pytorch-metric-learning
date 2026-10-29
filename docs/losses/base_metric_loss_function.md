# BaseMetricLossFunction

The base class of every loss that accepts the
[loss options](../losses.md#loss-options). Its keyword-only parameters are the
options that every loss accepts as `**options`.

```python
losses.BaseMetricLossFunction(*, distance=None, reducer=None, embedding_regularizer=None, embedding_reg_weight=1.0)
```

**Implementation notes**:

`forward` and `loss_terms` take the
[forward arguments](../losses.md#forward-arguments):

```python
loss_func(embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None)
loss_func.loss_terms(
    embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
)
```

Both call `objective_terms`, which validates shapes, devices, and the
`ref_emb`/`ref_labels` combination, converts `indices_tuple` to
`types.PairIndices` or `types.TripletIndices`, and calls `compute_loss`.
`forward` reduces the objective terms with the reducer, adds the terms from
`regularization_terms(embeddings)`, and returns a scalar tensor. `loss_terms`
returns the objective and regularization terms unreduced; see
[unreduced loss terms](../losses.md#unreduced-loss-terms).

Subclasses implement
`compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels)`,
which receives validated inputs. `labels` and `ref_labels` may be
`None`; `indices_tuple` is `None`, a `PairIndices`, or a `TripletIndices`; and
`ref_emb` is `None` for self-comparison. It returns a `types.LossDict`, or `{}`
when there is nothing to compute. Override `default_distance()` and
`default_reducer()` to change the defaults, and `regularization_terms` to add a
penalty that is applied after reduction. See
[custom losses](../extend/losses.md).

**Example**:

```python
import torch
from pytorch_metric_learning import losses


class UnitNormLoss(losses.BaseMetricLossFunction):
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        values = (embeddings.norm(dim=1) - 1).square()
        indices = torch.arange(len(embeddings), device=embeddings.device)
        return {"loss": {"losses": values, "indices": indices}}


loss_func = UnitNormLoss()
embeddings = torch.randn(32, 128, requires_grad=True)
loss = loss_func(embeddings)
loss.backward()
```

**Parameters**:

- **distance**: A [distance](../distances.md) object. If `None`,
  `default_distance()` is used.
- **reducer**: A [reducer](../reducers.md) object. If `None`,
  `default_reducer()` is used.
- **embedding_regularizer**: A [regularizer](../regularizers.md) object applied
  to the embeddings. If `None`, no embedding penalty is added.
- **embedding_reg_weight**: The weight multiplying the embedding penalty before
  it is added to the reduced loss. See
  [regularization terms](../losses.md#regularization-terms).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
