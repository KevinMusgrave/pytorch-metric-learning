# How to write custom loss functions

## The simplest possible loss function

```python
from pytorch_metric_learning.losses import BaseMetricLossFunction
import torch

class BarebonesLoss(BaseMetricLossFunction):
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        # perform some calculation #
        some_loss = torch.mean(embeddings)

        # put into dictionary #
        return {
            "loss": {
                "losses": some_loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
```

## Compatability with distances and reducers

You can make your loss function a lot more powerful by adding support for distance metrics and reducers:

```python
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch

class FullFeaturedLoss(BaseMetricLossFunction):
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels)
        anchors, positives, negatives = indices_tuple
        if len(anchors) == 0:
            return self.zero_losses()

        mat = self.distance(embeddings)
        ap_dists = mat[anchors, positives]
        an_dists = mat[anchors, negatives]

        # perform some calculations #
        losses1 = ap_dists - an_dists
        losses2 = ap_dists * 5
        losses3 = torch.mean(embeddings)

        # put into dictionary #
        return {
            "loss1": {
                "losses": losses1,
                "indices": indices_tuple,
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

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return CosineSimilarity()

    def sub_loss_names(self):
        return ["loss1", "loss2", "loss3"]
```

Here are a few details about this loss function:

- It operates on triplets, so `convert_to_triplets` is used to convert `indices_tuple` to triplet form.
- `self.distance` returns a pairwise distance matrix
- The output of the loss function is a dictionary that contains multiple sub losses. It overrides `sub_loss_names` to return their names. The method must return constants: the default reducer is copied once per name while the loss is constructed, and `zero_losses()` returns one zero entry per name. Regularization terms are not sub losses; do not list them.
- `get_default_reducer` is overriden to use `AvgNonZeroReducer` by default, rather than `MeanReducer`.
- `get_default_distance` is overriden to use `CosineSimilarity` by default, rather than `LpDistances(p=2)`.

## Losses with learnable parameters

Declare learnable weights as parameters of the loss and expose their
regularization explicitly. Add the weight penalty in `regularization_terms`, not
in `compute_loss`, so it is added after reduction like the embedding penalty:

```python
import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.losses.base_metric_loss_function import (
    regularization_term,
)
from pytorch_metric_learning.utils import common_functions as c_f


class CenterLoss(BaseMetricLossFunction):
    def __init__(
        self,
        num_classes,
        embedding_size,
        *,
        weight_regularizer=None,
        weight_reg_weight=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight_regularizer = weight_regularizer
        self.weight_reg_weight = weight_reg_weight
        self.centers = torch.nn.Parameter(torch.randn(num_classes, embedding_size))

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.check_parameter_placement(self, self.centers, embeddings)
        losses = (embeddings - self.centers[labels]).square().sum(dim=1)
        return {
            "loss": {
                "losses": losses,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }

    def regularization_terms(self, embeddings):
        terms = super().regularization_terms(embeddings)
        if self.weight_regularizer is not None:
            terms["weight_reg_loss"] = regularization_term(
                self.weight_regularizer, self.weight_reg_weight, self.centers
            )
        return terms
```

`check_parameter_placement` raises a clear `ValueError` when the caller forgot
to move the loss to the embeddings' device or dtype. Do not move or cast
parameters inside `forward`; that would silently change optimizer state.

## More on distances

To make your loss compatible with inverted distances (like cosine similarity), you can check `self.distance.is_inverted`, and write whatever logic necessary to make your loss make sense in that context.

There are also a few functions in `self.distance` that provide some of this logic, specifically `self.distance.smallest_dist`, `self.distance.largest_dist`, and `self.distance.margin`. The function definitions are pretty straightforward, and you can find them [here](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/distances/base_distance.py#L39-L53).

## Using `indices_tuple`

This is an optional argument passed in from the outside. (See the [loss usage examples](../losses.md) for an example.) It currently has 3 possible forms:

- `None`
- A tuple of size 4, representing the indices of mined pairs (anchors, positives, anchors, negatives)
- A tuple of size 3, representing the indices of mined triplets (anchors, positives, negatives)

To use `indices_tuple`, use the appropriate conversion function. You don't need to know what type will be passed in, as the conversion function takes care of that:

```python
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

# For a pair based loss
# After conversion, indices_tuple will be a tuple of size 4
indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)

# For a triplet based loss
# After conversion, indices_tuple will be a tuple of size 3
indices_tuple = lmu.convert_to_triplets(indices_tuple, labels)

# For a classification based loss
# miner_weights.shape == labels.shape
# You can use these to weight your loss
miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=torch.float32)
```

## Reduction type

The purpose of reduction types is to provide extra information to the reducer, if it needs it. For example, you could write a reducer that behaves differently depending on what kind of loss it receives. Here's a summary of each reduction type:

| Reduction type    | Meaning                                                                                     | Shape of "indices"                                                       |
| ----------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| "triplet"         | Each entry in "losses" represents a triplet.                                                | A tuple of 3 tensors (anchors, positives, negatives), each of size (N,). |
| "pos_pair"        | Each entry in "losses" represents a positive pair.                                          | A tuple of 2 tensors (anchors, positives), each of size (N,).            |
| "neg_pair"        | Each entry in "losses" represents a negative pair.                                          | A tuple of 2 tensors (anchors, negatives), each of size (N,).            |
| "element"         | Each entry in "losses" represents something other than a tuple, e.g. an element in a batch. | A tensor of size (N,)                                                    |
| "already_reduced" | "losses" is a single number, i.e. the loss has already been reduced.                        | Should be `None`                                                         |

## Some useful examples to look at

Here are some existing loss functions that might be useful for reference:

- [ContrastiveLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/contrastive_loss.py)
- [MultiSimilarityLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/multi_similarity_loss.py)
- [NormalizedSoftmaxLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/normalized_softmax_loss.py)
