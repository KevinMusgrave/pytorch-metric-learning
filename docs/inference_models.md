# Inference Models

utils.inference contains classes that make it convenient to find matching pairs within a batch, or from a set of pairs. Take a look at [this notebook](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/Inference.ipynb) to see example usage.

## InferenceModel
```python
from pytorch_metric_learning.utils.inference import InferenceModel
InferenceModel(trunk, 
				embedder=None, 
				match_finder=None, 
				indexer=None,
				normalize_embeddings=True,
				batch_size=64)
```
**Parameters**:

* **trunk**: Your trained model for computing embeddings.
* **embedder**: Optional. This is if your model is split into two components (trunk and embedder). If None, then the embedder will simply return the trunk's output.
* **match_finder**: A [MatchFinder](inference_models.md#matchfinder) object. If ```None```, it will be set to ```MatchFinder(mode="sim", threshold=0.9)```.
* **indexer**: The object used for computing k-nearest-neighbors. If ```None```, it will be set to ```FaissIndexer()```.
* **normalize_embeddings**: If True, embeddings will be normalized to have Euclidean norm of 1.
* **batch_size**: The batch size used to compute embeddings, when training the indexer for k-nearest-neighbor retrieval.



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


## FaissIndexer
This will create a faiss index, specifically ```IndexFlatL2```, which can be used for nearest neighbor retrieval.
```python
from pytorch_metric_learning.utils.inference import FaissIndexer
FaissIndexer()
```

