# How to write custom loss functions

1. Extend [BaseMetricLossFunction](../losses/base_metric_loss_function.md), or
   [BaseProxyLoss](../losses/base_proxy_loss.md) for a loss with class weights.
2. Implement `compute_loss`.
3. Inside `compute_loss`, return a dictionary of [loss terms](#loss-terms).

## The simplest possible loss function

```python
import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.types import LossDict


class BarebonesLoss(BaseMetricLossFunction):
    def compute_loss(
        self, embeddings, labels, indices_tuple, ref_emb, ref_labels
    ) -> LossDict:
        return {"loss": {"losses": embeddings.mean(), "indices": None}}


loss_func = BarebonesLoss()
loss = loss_func(torch.randn(8, 16))
```

`compute_loss` receives validated inputs:

- `labels` is `None` or a `(N,)` tensor on the embeddings' device.
- `indices_tuple` is `None`, a `types.PairIndices`, or a `types.TripletIndices`.
  Callers may pass plain tuples; they are validated and converted before
  `compute_loss` runs.
- `ref_emb` is `None` when the caller supplied no reference set. `None` means
  self-comparison: compare `embeddings` with themselves and never treat an
  embedding as its own positive. A tensor means a separate reference set, even
  when it is the same object as `embeddings`.
- `ref_labels` is `None` whenever `ref_emb` is `None`. With `ref_emb`, `labels`
  and `ref_labels` are both tensors or both `None`.

`compute_loss` returns a `types.LossDict`, a dictionary from term names to loss
terms. The base class reduces the terms with the reducer, adds the
regularization penalties, and returns a tensor.

## Compatibility with distances and reducers

A loss becomes more useful when it supports distance metrics and reducers:

```python
from typing import Unpack

import torch
from pytorch_metric_learning.distances import BaseDistance, CosineSimilarity
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer, BaseReducer
from pytorch_metric_learning.types import LossDict, LossOptions
from pytorch_metric_learning.utils import indexing


class FullFeaturedLoss(BaseMetricLossFunction):
    def __init__(self, margin: float = 0.1, **options: Unpack[LossOptions]) -> None:
        super().__init__(**options)
        self.margin = margin

    def compute_loss(
        self, embeddings, labels, indices_tuple, ref_emb, ref_labels
    ) -> LossDict:
        if labels is None and indices_tuple is None:
            raise ValueError("FullFeaturedLoss requires labels or indices_tuple")
        triplets = indexing.convert_to_triplets(indices_tuple, labels, ref_labels)
        anchors, positives, negatives = triplets
        if len(anchors) == 0:
            return {}

        mat = self.distance(embeddings, ref_emb)
        ap = mat[anchors, positives]
        an = mat[anchors, negatives]

        return {
            "triplet_loss": {
                "losses": torch.relu(self.distance.margin(ap, an) + self.margin),
                "indices": triplets,
            },
            "pos_loss": {
                "losses": torch.relu(self.distance.margin(ap, 0.5)),
                "indices": (anchors, positives),
            },
            "norm_loss": {
                "losses": (embeddings.norm(dim=1) - 1).square(),
                "indices": torch.arange(len(embeddings), device=embeddings.device),
            },
            "mean_loss": {
                "losses": embeddings.mean().abs(),
                "indices": None,
            },
        }

    def default_reducer(self) -> BaseReducer:
        return AvgNonZeroReducer()

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()


loss_func = FullFeaturedLoss(margin=0.2)
embeddings = torch.randn(32, 16, requires_grad=True)
labels = torch.randint(0, 4, (32,))
loss = loss_func(embeddings, labels)
loss.backward()

terms = loss_func.loss_terms(embeddings, labels)
assert set(terms) == {"triplet_loss", "pos_loss", "norm_loss", "mean_loss"}
assert loss_func(embeddings, torch.arange(32)).item() == 0
```

Here are a few details about this loss function:

- It operates on triplets, so `convert_to_triplets` turns labels or a supplied
  `indices_tuple` into a `TripletIndices`. Without `ref_emb`, `ref_labels` is
  `None` and the conversion excludes each embedding from its own positives.
- `self.distance(embeddings, ref_emb)` returns the `(N, N)` matrix of
  `embeddings` with themselves when `ref_emb` is `None`, or the `(N, M)` matrix
  against `ref_emb`.
- When there are no triplets, it returns `{}`. Every reducer turns an empty
  dictionary into a differentiable zero, so the call with all-distinct labels,
  which have no positive pairs, returns zero.
- Each term's `indices` tell the reducer what one value of `losses` is; see
  [loss terms](#loss-terms). Regularization penalties are not objective terms;
  the base class adds them after reduction.
- `default_reducer` and `default_distance` select the components used when the
  caller passes none. The user can override either through the
  [loss options](../losses.md#loss-options).
- Constructor arguments are declared before `**options: Unpack[LossOptions]`,
  and `**options` is forwarded to the base class, so the loss accepts the
  keyword-only options `distance`, `reducer`, `embedding_regularizer`, and
  `embedding_reg_weight`.

## Pair-based losses

[GenericPairLoss](../losses/generic_pair_loss.md) converts labels or mined
indices into pairs, returns `{}` when there are none, computes the distance
matrix, and calls `pair_loss(mat, pairs)`. Use `pair_values` for the distances
of the selected pairs or `pair_masks` for boolean positive and negative masks
over the whole matrix:

```python
import torch
from pytorch_metric_learning.losses import GenericPairLoss


class HingePairLoss(GenericPairLoss):
    def pair_loss(self, mat, pairs):
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        a1, p, a2, n = pairs
        return {
            "pos_loss": {
                "losses": torch.relu(self.distance.margin(pos_pairs, 0.2)),
                "indices": (a1, p),
            },
            "neg_loss": {
                "losses": torch.relu(self.distance.margin(0.8, neg_pairs)),
                "indices": (a2, n),
            },
        }


loss_func = HingePairLoss()
embeddings = torch.randn(16, 8)
labels = torch.randint(0, 3, (16,))
loss = loss_func(embeddings, labels)
loss = loss_func(
    embeddings,
    labels,
    ref_emb=torch.randn(10, 8),
    ref_labels=torch.randint(0, 3, (10,)),
)
```

## Losses with learnable parameters

Extend [BaseProxyLoss](../losses/base_proxy_loss.md) for a loss with one
learnable vector per class. It owns the `weight` parameter, shaped
`(num_weights, embedding_size)`, the keyword-only `weight_regularizer`,
`weight_reg_weight`, and `weight_init_func` arguments, and the `weight_reg_loss`
term that is added after reduction. Override `default_weight_init` to change
the initialization used when the caller passes no `weight_init_func`:

```python
from typing import Unpack

import torch
from pytorch_metric_learning import regularizers
from pytorch_metric_learning.losses import BaseProxyLoss
from pytorch_metric_learning.types import ProxyLossOptions


class CenterLoss(BaseProxyLoss):
    def __init__(
        self, num_classes: int, embedding_size: int, **options: Unpack[ProxyLossOptions]
    ) -> None:
        super().__init__(num_classes, embedding_size, **options)

    def default_weight_init(self, weight):
        torch.nn.init.xavier_uniform_(weight)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        if labels is None:
            raise ValueError("CenterLoss requires labels")
        if indices_tuple is not None or ref_emb is not None:
            raise ValueError("CenterLoss does not support indices_tuple or ref_emb")
        if self.weight.device != embeddings.device or (
            not torch.is_autocast_enabled(embeddings.device.type)
            and self.weight.dtype != embeddings.dtype
        ):
            raise ValueError("Move the loss with .to(device, dtype) before calling it")
        losses = (embeddings - self.weight[labels]).square().sum(dim=1)
        return {
            "loss": {
                "losses": losses,
                "indices": torch.arange(len(embeddings), device=embeddings.device),
            }
        }


loss_func = CenterLoss(
    num_classes=4,
    embedding_size=16,
    weight_regularizer=regularizers.LpRegularizer(),
    weight_reg_weight=0.01,
)
optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.1)
embeddings = torch.randn(32, 16)
labels = torch.randint(0, 4, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
optimizer.step()
assert set(loss_func.loss_terms(embeddings, labels)) == {"loss", "weight_reg_loss"}
```

Raise an error for unsupported arguments instead of ignoring them, so callers
learn which combinations the loss supports. Do not move or cast parameters
inside `forward`; that would silently change optimizer state.

A loss with learnable parameters that are not one vector per class, such as a
temperature, can declare them on a `BaseMetricLossFunction` subclass and add a
penalty by overriding `regularization_terms(embeddings)`. Penalty term names
must differ from the names `compute_loss` returns; a shared name raises
`ValueError` from `forward` and `loss_terms`.

## More on distances

To make your loss compatible with both distances and similarities, check
`self.distance.is_inverted`, which is `True` for similarities, where larger
means closer. Two helpers cover the common cases:

- `self.distance.margin(x, y)` returns `x - y` for distances and `y - x` for
  similarities, so `margin(anchor_positive, anchor_negative)` is positive when a
  triplet is violated in either orientation.
- `self.distance.pairwise(query, ref)` returns the distance between
  corresponding rows, applying the distance's normalization and power.

To select the closest value, pick `torch.max` or `torch.min` from
`is_inverted`. To require a specific distance, check it in the constructor:

```python
from typing import Unpack

import torch
from pytorch_metric_learning.distances import BaseDistance, CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.types import LossOptions


class NearestNeighborLoss(BaseMetricLossFunction):
    def __init__(self, **options: Unpack[LossOptions]) -> None:
        super().__init__(**options)
        if not isinstance(self.distance, CosineSimilarity):
            raise TypeError("NearestNeighborLoss requires CosineSimilarity")

    def default_distance(self) -> BaseDistance:
        return CosineSimilarity()

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        if ref_emb is None:
            mat = self.distance(embeddings)
            mat = mat.masked_fill(
                torch.eye(len(mat), dtype=torch.bool, device=mat.device), -torch.inf
            )
        else:
            mat = self.distance(embeddings, ref_emb)
        closest = torch.amax if self.distance.is_inverted else torch.amin
        nearest = closest(mat, dim=1)
        return {
            "loss": {
                "losses": 1 - nearest,
                "indices": torch.arange(len(embeddings), device=embeddings.device),
            }
        }


loss = NearestNeighborLoss()(torch.randn(8, 4))
try:
    NearestNeighborLoss(distance=LpDistance())
except TypeError:
    pass
```

## Using `indices_tuple`

This optional argument comes from the caller, usually from a miner. Inside
`compute_loss` it has one of three forms:

- `None`
- `types.PairIndices(pos_anchors, positives, neg_anchors, negatives)`, the
  indices of mined pairs
- `types.TripletIndices(anchors, positives, negatives)`, the indices of mined
  triplets

Anchors index `embeddings`; positives and negatives index `ref_emb`, or
`embeddings` when `ref_emb` is `None`. Use the conversion function for the
form your loss needs; each accepts any of the three inputs, as well as plain
tuples:

```python
import torch
from pytorch_metric_learning.types import PairIndices, TripletIndices
from pytorch_metric_learning.utils import indexing

labels = torch.tensor([0, 0, 1, 1, 2])
pairs = indexing.convert_to_pairs(None, labels)
triplets = indexing.convert_to_triplets(pairs, labels)
weights = indexing.convert_to_weights(triplets, labels, dtype=torch.float32)
plain = (torch.tensor([0, 2]), torch.tensor([1, 3]), torch.tensor([4, 4]))

assert isinstance(pairs, PairIndices)
assert isinstance(triplets, TripletIndices)
assert isinstance(indexing.convert_to_pairs(plain, labels), PairIndices)
assert weights.shape == (5,) and weights.max() == 1
```

`convert_to_pairs` reads a triplet `(a, p, n)` as the pairs `(a, p)` and
`(a, n)`. `convert_to_triplets` combines each anchor's positive pairs with its
negative pairs, or enumerates triplets from labels; its `t_per_anchor` defaults
to `"all"`. With `indices_tuple=None`, both use labels, and `ref_labels=None`
means self-comparison. `convert_to_weights` returns one weight per element:
its count in the tuples divided by the largest count. With
`indices_tuple=None` every weight is 1; a tuple with no indices gives every
element weight 0, so a loss that multiplies by the weights reduces to a
differentiable zero. See [miner weights](../losses.md#miner-weights).

## Loss terms

`compute_loss` returns `types.LossDict`, a dictionary from term names to
`types.LossTerm` dictionaries. Each term has `losses`, always a tensor, and
`indices`, whose form tells the reducer what one value of `losses` is:

| `indices`                                                        | `losses`                                   | Reducers group by |
| ---------------------------------------------------------------- | ------------------------------------------ | ----------------- |
| `None`                                                           | One value; the term is already reduced     | Not reduced       |
| A tensor of shape `(n,)`                                         | `(n,)`, one value per element              | That tensor       |
| `(anchors, others)`, two tensors of shape `(n,)`                 | `(n,)`, one value per (anchor, other) pair | `anchors`         |
| `TripletIndices(anchors, positives, negatives)` or three tensors | `(n,)`, one value per triplet              | `anchors`         |

An element can be an embedding, a query, or a proxy; the indices only need to
be meaningful to the reducer in use. Reducers that look up labels, such as
[ClassWeightedReducer](../reducers.md#classweightedreducer), index `labels`
with the anchors. An empty `losses` tensor, and an empty `LossDict`, reduce to a
differentiable zero. A term without indices must contain exactly one value, and
a term with indices must have one-dimensional `losses` and index tensors of
equal length; otherwise reducers raise `ValueError`. A tuple of indices holds
two or three tensors; use a single tensor, not a one-element tuple, for
per-element losses. Index tensors must be `int64` or `int32`, and reducers raise
`TypeError` for other dtypes or structures.

A term may also contain an optional `divisor`, a tensor or number. Only
[DivisorReducer](../reducers.md#divisorreducer) applies it, by dividing the sum
of the term's losses by the divisor; every other reducer raises `ValueError`
for such a term. Pass a [MultipleReducers](../reducers.md#multiplereducers) to
use DivisorReducer for some terms and another reducer for the rest.
[ProxyAnchorLoss](../losses/proxy_anchor_loss.md) and
[MarginLoss](../losses/margin_loss.md) use divisors.

`PairIndices`, `TripletIndices`, `LossTerm`, and `LossDict` describe ordinary
tuples and dictionaries; there is no runtime wrapper. `forward` returns a
tensor, and `loss_terms` returns the `LossDict` before reduction, including
regularization terms, for custom aggregation; see
[unreduced loss terms](../losses.md#unreduced-loss-terms).

## Some useful examples to look at

- [ContrastiveLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/contrastive_loss.py)
- [MultiSimilarityLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/multi_similarity_loss.py)
- [NormalizedSoftmaxLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/normalized_softmax_loss.py)
- [ProxyAnchorLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/proxy_anchor_loss.py)
