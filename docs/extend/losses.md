# How to write custom loss functions

## The simplest possible loss function

```python
import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.types import LossDict


class BarebonesLoss(BaseMetricLossFunction):
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels) -> LossDict:
        some_loss = torch.mean(embeddings)
        return {
            "loss": {
                "losses": some_loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
```

`compute_loss` receives prepared inputs: `labels` is on the embeddings'
device, and `ref_emb` and `ref_labels` equal `embeddings` and `labels` when the
caller supplied none. It returns a `LossDict`, a dictionary from term names to
loss terms. The base class reduces the terms, adds the regularization penalty,
and returns a tensor.

## Compatibility with distances and reducers

A loss becomes more useful when it supports distance metrics and reducers:

```python
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import indexing


class FullFeaturedLoss(BaseMetricLossFunction):
    sub_loss_names = ("loss1", "loss2", "loss3")

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        anchors, positives, negatives = indexing.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor="all"
        )
        if len(anchors) == 0:
            return self.zero_losses(embeddings)

        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchors, positives]
        an_dists = mat[anchors, negatives]

        losses1 = ap_dists - an_dists
        losses2 = ap_dists * 5
        losses3 = torch.mean(embeddings)

        return {
            "loss1": {
                "losses": losses1,
                "indices": (anchors, positives, negatives),
                "reduction_type": "triplet",
            },
            "loss2": {
                "losses": losses2,
                "indices": (anchors, positives),
                "reduction_type": "pos_pair",
            },
            "loss3": {
                "losses": losses3,
                "indices": None,
                "reduction_type": "already_reduced",
            },
        }

    def default_reducer(self):
        return AvgNonZeroReducer()

    def default_distance(self):
        return CosineSimilarity()
```

Here are a few details about this loss function:

- It operates on triplets, so `convert_to_triplets` turns labels or a supplied
  `indices_tuple` into triplet form.
- `self.distance` returns a pairwise distance matrix between the anchors and
  the reference embeddings.
- The output dictionary contains several terms. `sub_loss_names` is a class
  attribute listing their names; `zero_losses(embeddings)` returns one
  differentiable zero per name when there is nothing to compute. Regularization
  terms are not sub losses; do not list them.
- `default_reducer` and `default_distance` select the components used when the
  caller passes none. The user can override either through the
  [loss options](../losses.md#loss-options).

A custom loss inherits the keyword-only options `distance`, `reducer`,
`embedding_regularizer`, and `embedding_reg_weight`. To add your own
constructor arguments, accept `**options` and forward it:

```python
from typing import Unpack

from pytorch_metric_learning.types import LossOptions


class MarginAwareLoss(FullFeaturedLoss):
    def __init__(self, margin: float = 0.1, **options: Unpack[LossOptions]) -> None:
        super().__init__(**options)
        self.margin = margin
```

## Pair-based losses

[GenericPairLoss](../losses/generic_pair_loss.md) converts labels or mined
indices into pairs, skips batches without pairs, computes the distance matrix,
and calls `pair_loss(mat, pairs)`. Use `pair_values` for the distances of the
selected pairs or `pair_masks` for boolean positive and negative masks over the
whole matrix:

```python
import torch
from pytorch_metric_learning.losses import GenericPairLoss


class HingePairLoss(GenericPairLoss):
    sub_loss_names = ("pos_loss", "neg_loss")

    def pair_loss(self, mat, pairs):
        pos_pairs, neg_pairs = self.pair_values(mat, pairs)
        a1, p, a2, n = pairs
        return {
            "pos_loss": {
                "losses": torch.relu(self.distance.margin(pos_pairs, 0.2)),
                "indices": (a1, p),
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": torch.relu(self.distance.margin(0.8, neg_pairs)),
                "indices": (a2, n),
                "reduction_type": "neg_pair",
            },
        }
```

## Losses with learnable parameters

Declare learnable weights as a parameter named `weight` with one row per class,
and expose their regularization explicitly. Add the weight penalty in
`regularization_terms`, not in `compute_loss`, so it is added after reduction
like the embedding penalty:

```python
import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.losses.base_metric_loss_function import (
    regularization_term,
)


class CenterLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes,
        embedding_size,
        *,
        weight_regularizer=None,
        weight_reg_weight=1.0,
        **options,
    ):
        super().__init__(**options)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.weight = torch.nn.Parameter(torch.randn(num_classes, embedding_size))

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
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
                "reduction_type": "element",
            }
        }

    def regularization_terms(self, embeddings):
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.weight
            )
        return terms
```

Do not move or cast parameters inside `forward`; that would silently change
optimizer state.

## More on distances

To make your loss compatible with inverted distances such as cosine similarity,
check `self.distance.is_inverted` and write the logic that makes sense in that
context. Two helpers cover the common cases: `self.distance.smallest_dist(*args, **kwargs)`
calls `torch.max` for similarities and `torch.min` for distances, and
`self.distance.margin(x, y)` returns `y - x` for similarities and `x - y` for
distances. To require a specific distance, check it in the constructor:

```python
from pytorch_metric_learning.distances import CosineSimilarity

if not isinstance(self.distance, CosineSimilarity):
    raise TypeError("MyLoss requires CosineSimilarity")
```

## Using `indices_tuple`

This optional argument comes from the caller, usually from a miner. It has three
possible forms:

- `None`
- A tuple of four tensors, the indices of mined pairs `(anchors, positives, anchors, negatives)`
- A tuple of three tensors, the indices of mined triplets `(anchors, positives, negatives)`

Use the conversion function for the form your loss needs; it accepts any of the
three inputs:

```python
from pytorch_metric_learning.utils import indexing

pairs = indexing.convert_to_pairs(indices_tuple, labels, ref_labels)
triplets = indexing.convert_to_triplets(indices_tuple, labels, ref_labels, t_per_anchor="all")
miner_weights = indexing.convert_to_weights(indices_tuple, labels, dtype=torch.float32)
```

`convert_to_weights` returns one weight per element, the element's relative
frequency in the mined tuples; classification losses multiply per-element
losses by it.

## Reduction type

Reduction types tell the reducer what each entry of `losses` represents:

| Reduction type    | Meaning                                                                                     | Shape of "indices"                                                       |
| ----------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| "triplet"         | Each entry in "losses" represents a triplet.                                                | A tuple of 3 tensors (anchors, positives, negatives), each of size (N,). |
| "pos_pair"        | Each entry in "losses" represents a positive pair.                                          | A tuple of 2 tensors (anchors, positives), each of size (N,).            |
| "neg_pair"        | Each entry in "losses" represents a negative pair.                                          | A tuple of 2 tensors (anchors, negatives), each of size (N,).            |
| "element"         | Each entry in "losses" represents something other than a tuple, e.g. an element in a batch. | A tensor of size (N,)                                                    |
| "already_reduced" | "losses" is a single value, i.e. the loss has already been reduced.                         | `None`                                                                   |

An empty `losses` tensor reduces to a differentiable zero.

## Typed return contract

`compute_loss` returns `pytorch_metric_learning.types.LossDict`, a dictionary
from term names to `LossTerm` dictionaries. Each term has `losses` (always a
tensor), `indices`, and `reduction_type`, plus an optional `divisor` for
`DivisorReducer`. `PairIndices` contains four tensors and `TripletIndices`
contains three; their union, `IndicesTuple`, describes the optional input to
loss computation and the output of mining. These are annotations for ordinary
dictionaries and tuples; there is no runtime wrapper or conversion.

`forward` returns a tensor. `loss_terms` returns the `LossDict` before
reduction, including regularization terms, for custom aggregation.

## Some useful examples to look at

- [ContrastiveLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/contrastive_loss.py)
- [MultiSimilarityLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/multi_similarity_loss.py)
- [NormalizedSoftmaxLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/normalized_softmax_loss.py)
