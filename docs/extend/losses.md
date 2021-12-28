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

    def _sub_loss_names(self):
        return ["loss1", "loss2", "loss3"]
```

Here are a few details about this loss function:

 - It operates on triplets, so ```convert_to_triplets``` is used to convert ```indices_tuple``` to triplet form.
 - ```self.distance``` returns a pairwise distance matrix
 - The output of the loss function is a dictionary that contains multiple sub losses. This is why it overrides the ```_sub_loss_names``` function.
 - ```get_default_reducer``` is overriden to use ```AvgNonZeroReducer``` by default, rather than ```MeanReducer```.
 - ```get_default_distance``` is overriden to use ```CosineSimilarity``` by default, rather than ```LpDistances(p=2)```.


## More on distances
To make your loss compatible with inverted distances (like cosine similarity), you can check ```self.distance.is_inverted```, and write whatever logic necessary to make your loss make sense in that context. 

There are also a few functions in ```self.distance``` that provide some of this logic, specifically ```self.distance.smallest_dist```, ```self.distance.largest_dist```, and ```self.distance.margin```. The function definitions are pretty straightforward, and you can find them [here](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/distances/base_distance.py#L39-L53).

## Using ```indices_tuple```

This is an optional argument passed in from the outside. (See the [overview](../../#using-losses-and-miners-in-your-training-loop) for an example.) It currently has 3 possible forms: 

 - ```None```
 - A tuple of size 4, representing the indices of mined pairs (anchors, positives, anchors, negatives)
 - A tuple of size 3, representing the indices of mined triplets (anchors, positives, negatives)

To use ```indices_tuple```, use the appropriate conversion function. You don't need to know what type will be passed in, as the conversion function takes care of that:

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

| Reduction type | Meaning | Shape of "indices" |
|--|--|--|
| "triplet" | Each entry in "losses" represents a triplet. | A tuple of 3 tensors (anchors, positives, negatives), each of size (N,). |
| "pos_pair" | Each entry in "losses" represents a positive pair. | A tuple of 2 tensors (anchors, positives), each of size (N,). |
| "neg_pair" | Each entry in "losses" represents a negative pair. | A tuple of 2 tensors (anchors, negatives), each of size (N,). |
| "element" | Each entry in "losses" represents something other than a tuple, e.g. an element in a batch. | A tensor of size (N,) |  
| "already_reduced" | "losses" is a single number, i.e. the loss has already been reduced. | Should be ```None``` |

## Some useful examples to look at
Here are some existing loss functions that might be useful for reference:

- [ContrastiveLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/contrastive_loss.py)
- [MultiSimilarityLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/multi_similarity_loss.py)
- [NormalizedSoftmaxLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/normalized_softmax_loss.py)