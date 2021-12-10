# Inference Models

utils.inference contains classes that make it convenient to find matching pairs within a batch, or from a set of pairs. Take a look at [this notebook](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/Inference.ipynb) to see example usage.

## InferenceModel
```python
from pytorch_metric_learning.utils.inference import InferenceModel
InferenceModel(trunk,
				embedder=None,
				match_finder=None,
				normalize_embeddings=True,
				knn_func=None,
				data_device=None,
				dtype=None)
```
**Parameters**:

* **trunk**: Your trained model for computing embeddings.
* **embedder**: Optional. This is if your model is split into two components (trunk and embedder). If None, then the embedder will simply return the trunk's output.
* **match_finder**: A [MatchFinder](inference_models.md#matchfinder) object. If ```None```, it will be set to ```MatchFinder(distance=CosineSimilarity(), threshold=0.9)```.
* **normalize_embeddings**: If True, embeddings will be normalized to have Euclidean norm of 1.
* **knn_func**: The function used for computing k-nearest-neighbors. If ```None```, it will be set to ```FaissKNN()```.
* **data_device**: The device that you want to put batches of data on. If not specified, GPUs will be used if available.
* **dtype**: The datatype to cast data to. If None, no casting will be done.

**Methods**:
```python
# initialize with a model
im = InferenceModel(model)

# pass in a dataset to serve as the search space for k-nn
im.train_knn(dataset)

# add another dataset to the index
im.add_to_knn(dataset2)

# get the 10 nearest neighbors of a query
distances, indices = im.get_nearest_neighbors(query, k=10)

# determine if inputs are close to each other
is_match = im.is_match(x, y)

# determine "is_match" pairwise for all elements in a batch
match_matrix = im.get_matches(x)

# save and load the knn function (which is a faiss index by default)
im.save_knn_func("filename.index")
im.load_knn_func("filename.index")
```


## MatchFinder
```python
from pytorch_metric_learning.utils.inference import MatchFinder
MatchFinder(distance=None, threshold=None)
```

**Parameters**:

* **distance**: A [distance](distances.md) object.
* **threshold**: Optional. Pairs will be a match if they fall under this threshold for non-inverted distances, or over this value for inverted distances. If not provided, then a threshold must be provided during function calls.


## FaissKNN

Uses the faiss library to compute k-nearest-neighbors

```python
from pytorch_metric_learning.utils.inference import FaissKNN
FaissKNN(reset_before=True,
			reset_after=True, 
			index_init_fn=None, 
			gpus=None)
```

**Parameters**:

* **reset_before**: Reset the faiss index before knn is computed.
* **reset_after**: Reset the faiss index after knn is computed (good for clearing memory).
* **index_init_fn**: A callable that takes in the embedding dimensionality and returns a faiss index. The default is ```faiss.IndexFlatL2```.
* **gpus**: A list of gpu indices to move the faiss index onto. The default is to use all available gpus, if the input tensors are also on gpus.

Example:
```python
# use faiss.IndexFlatIP on 3 gpus
knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, gpus=[0,1,2])

# query = query embeddings 
# k = the k in k-nearest-neighbors
# reference = the embeddings to search
# last argument is whether or not query and reference share datapoints
distances, indices = knn_func(query, k, references, False)
```

## FaissKMeans

Uses the faiss library to do k-means clustering.

```python
from pytorch_metric_learning.utils.inference import FaissKMeans
FaissKMeans(**kwargs)
```

**Parameters**:

* **kwargs**: Keyword arguments that will be passed to the ```faiss.Kmeans``` constructor.

Example:
```python
kmeans_func = FaissKMeans(niter=100, verbose=True, gpu=True)

# cluster into 10 groups
cluster_assignments = kmeans_func(embeddings, 10)
```

## CustomKNN

Uses a [distance function](distances.md) to determine similarity between datapoints, and then computes k-nearest-neighbors.

```python
from pytorch_metric_learning.utils.inference import CustomKNN
CustomKNN(distance)
```

**Parameters**:

* **distance**: A [distance function](distances.md)

Example:
```python
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.utils.inference import CustomKNN

knn_func = CustomKNN(SNRDistance())
distances, indices = knn_func(query, k, references, False)
```