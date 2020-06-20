# Inference Models

utils.inference contains classes that make it convenient to find matching pairs within a batch, or from a set of pairs. Take a look at [this notebook](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/Inference.ipynb) to see example usage.

## InferenceModel
```python
from pytorch_metric_learning.utils.inference import InferenceModel
InferenceModel(trunk, embedder=None, match_finder=None, normalize_embeddings=True)
```
**Parameters**:

* **trunk**: Your trained model for computing embeddings.
* **embedder**: Optional. This is if your model is split into two components (trunk and embedder). If None, then the embedder will simply return the trunk's output.
* **match_finder**: A [MatchFinder](inference_model.md#matchfinder) object.
* **normalize_embeddings**: If True, embeddings will be normalized to have Euclidean norm of 1.


## MatchFinder
```python
from pytorch_metric_learning.utils.inference import MatchFinder
MatchFinder(mode="dist", threshold=None)
```

**Parameters**:

* **mode**: One of:
	* ```dist```: Use the Euclidean distance between vectors
	* ```squared_dist```: Use the squared Euclidean distance between vectors
	* ```sim```: Use the dot product of vectors
* **threshold**: Optional. Pairs will be a match if they fall under this threshold for distance modes, or over this value for the similarity mode. If not provided, then a threshold must be provided during function calls.