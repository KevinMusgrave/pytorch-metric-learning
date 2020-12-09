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
* **match_finder**: A [MatchFinder](inference_models.md#matchfinder) object. If ```None```, it will be set to ```MatchFinder(distance=CosineSimilarity(), threshold=0.9)```.
* **indexer**: The object used for computing k-nearest-neighbors. If ```None```, it will be set to ```FaissIndexer()```.
* **normalize_embeddings**: If True, embeddings will be normalized to have Euclidean norm of 1.
* **batch_size**: The batch size used to compute embeddings, when training the indexer for k-nearest-neighbor retrieval.



## MatchFinder
```python
from pytorch_metric_learning.utils.inference import MatchFinder
MatchFinder(distance=None, threshold=None)
```

**Parameters**:

* **distance**: A [distance](distances.md) object.
* **threshold**: Optional. Pairs will be a match if they fall under this threshold for non-inverted distances, or over this value for inverted distances. If not provided, then a threshold must be provided during function calls.


## FaissIndexer
This will create a faiss index, specifically ```IndexFlatL2```, which can be used for nearest neighbor retrieval.
```python
from pytorch_metric_learning.utils.inference import FaissIndexer
FaissIndexer()
```


## LogitGetter
This class makes it easier to extract logits from classifier loss functions. Although "metric learning" usually means that you use embeddings during inference, there might be cases where you want to use the class logits instead of the embeddings.
```python
from pytorch_metric_learning.utils.inference import LogitGetter
LogitGetter(
        classifier,
        layer_name=None,
        transpose=None,
        distance=None,
        copy_weights=True,
    ):
```

**Parameters**:

* **classifier**: The classifier loss function that you want to extract logits from.
* **layer_name**: Optional. The attribute name of the weight matrix inside the classifier. If not specified, each of the following will be tried: ```["fc", "proxies", "W"]```.
* **transpose**: Optional. Whether or not to transpose the weight matrix. If the weight matrix is of size ```(embedding_size, num_classes)```, then it should be transposed. If not specified, then transposing will be done automatically during the ```forward``` call if necessary, based on the shapes of the input embeddings and the weight matrix. (Note that it is only guaranteed to make the correct transposing decision if ```num_classes != embedding_size```.)
* **distance**: Optional. A [distance](distances.md) object which will compute the distance or similarity matrix, i.e. the logits. If not specified, then ```classifier.distance``` will be used.
* **copy_weights**: If True, then LogitGetter will contain a copy of (instead of a reference to) the classifier weights, so that if you update the classifier weights, the LogitGetter remains unchanged.


Example usage:
```python
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.utils.inference import LogitGetter

loss_fn = ArcFaceLoss(num_classes = 100, embedding_size = 512)
LG = LogitGetter(loss_fn)
logits = LG(embeddings)
```