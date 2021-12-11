# Accuracy Calculation

The AccuracyCalculator class computes several accuracy metrics given a query and reference embeddings. It can be easily extended to create custom accuracy metrics.

```python
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
AccuracyCalculator(include=(),
                    exclude=(),
                    avg_of_avgs=False,
                    return_per_class=False,
                    k=None,
                    label_comparison_fn=None,
                    device=None,
                    knn_func=None,
                    kmeans_func=None)
```
### Parameters

* **include**: Optional. A list or tuple of strings, which are the names of metrics you want to calculate. If left empty, all default metrics will be calculated.
* **exclude**: Optional. A list or tuple of strings, which are the names of metrics you **do not** want to calculate.
* **avg_of_avgs**: If True, the average accuracy per class is computed, and then the average of those averages is returned. This can be useful if your dataset has unbalanced classes. If False, the global average will be returned.
* **return_per_class**: If True, the average accuracy per class is computed and returned.
* **k**: The number of nearest neighbors that will be retrieved for metrics that require k-nearest neighbors. The allowed values are:
    * ```None```. This means k will be set to the total number of reference embeddings.
    * An integer greater than 0. This means k will be set to the input integer.
    * ```"max_bin_count"```. This means k will be set to ```max(bincount(reference_labels)) - self_count``` where ```self_count == 1``` if the query and reference embeddings come from the same source.
* **label_comparison_fn**: A function that compares two torch arrays of labels and returns a boolean array. The default is ```torch.eq```. If a custom function is used, then you must exclude clustering based metrics ("NMI" and "AMI"). The following is an example of a custom function for two-dimensional labels. It returns ```True``` if the 0th column matches, and the 1st column does **not** match:
* **device**: The device to move input tensors to. If ```None```, will default to GPUs if available.
* **knn_func**: A callable that takes in 4 arguments (```query, k, reference, embeddings_come_from_same_source```) and returns ```distances, indices```. Default is ```pytorch_metric_learning.utils.inference.FaissKNN```.
* **kmeans_func**: A callable that takes in 2 arguments (```x, nmb_clusters```) and returns a 1-d tensor of cluster assignments. Default is ```pytorch_metric_learning.utils.inference.FaissKMeans```.
```python
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.utils.inference import CustomKNN

def example_label_comparison_fn(x, y):
    return (x[:, 0] == y[:, 0]) & (x[:, 1] != y[:, 1])

knn_func = CustomKNN(SNRDistance())
AccuracyCalculator(exclude=("NMI", "AMI"), 
                    label_comparison_fn=example_label_comparison_fn,
                    knn_func=knn_func)
```

### Getting accuracy

Call the ```get_accuracy``` method to obtain a dictionary of accuracies.
```python
def get_accuracy(self, 
	query, 		
	reference, 
	query_labels, 
	reference_labels, 
	embeddings_come_from_same_source, 
	include=(),
	exclude=()
):
# returns a dictionary mapping from metric names to accuracy values
# The default metrics are:
# "NMI" (Normalized Mutual Information)
# "AMI" (Adjusted Mutual Information)
# "precision_at_1"
# "r_precision"
# "mean_average_precision_at_r"
```

* **query**: A 2D torch or numpy array of size ```(Nq, D)```, where Nq is the number of query samples. For each query sample, nearest neighbors are retrieved and accuracy is computed.
* **reference**: A 2D torch or numpy array of size ```(Nr, D)```, where Nr is the number of reference samples. This is where nearest neighbors are retrieved from.
* **query_labels**: A 1D torch or numpy array of size ```(Nq)```. Each element should be an integer representing the sample's label.
* **reference_labels**: A 1D torch or numpy array of size ```(Nr)```. Each element should be an integer representing the sample's label. 
* **embeddings_come_from_same_source**: Set to True if ```query``` is a subset of ```reference``` or if ```query is reference```. Set to False otherwise.
* **include**: Optional. A list or tuple of strings, which are the names of metrics you want to calculate. If left empty, all metrics specified during initialization will be calculated.
* **exclude**: Optional. A list or tuple of strings, which are the names of metrics you do not want to calculate.

Note that labels can be 2D if a [custom label comparison function](#using-a-custom-label-comparison-function) is used.


### CPU/GPU usage

* If you installed ```faiss-cpu``` then the CPU will always be used.
* If you installed ```faiss-gpu```, then the GPU will be used if ```k <= 1024``` for CUDA < 9.5, and ```k <= 2048``` for CUDA >= 9.5. If this condition is not met, then the CPU will be used. 

If your dataset is large, you might find the k-nn search is very slow. This is because the default behavior is to set k to ```len(reference_embeddings)```. To avoid this, you can set k to a number, like ```k = 1000```, or try ```k = "max_bin_count"```.


### Explanations of the default accuracy metrics

- **AMI**: 

     - [scikit-learn article](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)
     - [Wikipedia](https://en.wikipedia.org/wiki/Adjusted_mutual_information)

- **NMI**:

     - [scikit-learn article](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
     - [Slides from Northeastern University](https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf)

- **mean_average_precision**:

    - [Slides from Stanford](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf)

- **mean_average_precision_at_r**:

    - [See section 3.2 of A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)

- **precision_at_1**:

    - Fancy way of saying "is the 1st nearest neighbor correct?"

- **r_precision**:

    - [See chapter 8 (page 161) of Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)


**Important note**

AccuracyCalculator's ```mean_average_precision_at_r``` and ```r_precision``` are correct only if ```k = None```, **or** ```k = "max_bin_count"```, **or** ```k >= max(bincount(reference_labels))```


### Adding custom accuracy metrics

Let's say you want to use the existing metrics but also compute precision @ 2, and a fancy mutual info method. You can extend the existing class, and write methods that start with the keyword ```calculate_```

```python
from pytorch_metric_learning.utils import accuracy_calculator

class YourCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_2(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 2)

    def calculate_fancy_mutual_info(self, query_labels, cluster_labels, **kwargs):
        return fancy_computations

    def requires_clustering(self):
        return super().requires_clustering() + ["fancy_mutual_info"] 

    def requires_knn(self):
    	return super().requires_knn() + ["precision_at_2"] 
```

Any method that starts with "calculate_" will be passed the following kwargs:
```python
kwargs = {"query": query,                    # query embeddings
    "reference": reference,                  # reference embeddings
    "query_labels": query_labels,        
    "reference_labels": reference_labels,
    "embeddings_come_from_same_source": e}  # True if query is reference, or if query is a subset of reference.
```

If your method requires a k-nearest neighbors search, then append your method's name to the ```requires_knn``` list, as shown in the above example. If any of your accuracy methods require k-nearest neighbors, they will also receive the following kwargs:

```python
	{"label_counts": label_counts,           # A dictionary mapping from reference labels to the number of times they occur
    "knn_labels": knn_labels,                # A 2d array where each row is the labels of the nearest neighbors of each query. The neighbors are retrieved from the reference set
    "knn_distances": knn_distances           # The euclidean distance corresponding to each k-nearest neighbor in knn_labels
    "lone_query_labels": lone_query_labels   # The set of labels (in the form of a torch array) that have only 1 occurrence in reference_labels
    "not_lone_query_mask": not_lone_query_mask} # A boolean mask, where True means that a query element has at least 1 possible neighbor in reference.           
```

If your method requires cluster labels, then append your method's name to the ```requires_clustering``` list, as shown in the above example. Then, if any of your methods need cluster labels, ```self.get_cluster_labels()``` will be called, and the kwargs will include:

```python
    {"cluster_labels": cluster_labels} # A 1D array with a cluster label for each element in the query embeddings.
```

Now when ```get_accuracy``` is called, the returned dictionary will contain ```precision_at_2``` and ```fancy_mutual_info```:
```python
calculator = YourCalculator()
acc_dict = calculator.get_accuracy(query_embeddings,
    reference_embeddings,
    query_labels,
    reference_labels,
    embeddings_come_from_same_source=True
)
# Now acc_dict contains the metrics "precision_at_2" and "fancy_mutual_info"
# in addition to the original metrics from AccuracyCalculator
```

You can use your custom calculator with the [tester](testers.md) classes as well, by passing it in as an init argument. (By default, the testers use AccuracyCalculator.)
```python
from pytorch_metric_learning import testers
t = testers.GlobalEmbeddingSpaceTester(..., accuracy_calculator=YourCalculator())
```


### Using a custom label comparison function
If you define your own ```label_comparison_fn```, then ```query_labels``` and ```reference_labels``` can be 1D or 2D, and consist of integers or floating point numbers, as long as your ```label_comparison_fn``` can process them.

Example of 2D labels:
```python
def label_comparison_fn(x, y):
    return (x[..., 0] == y[..., 0]) & (x[..., 1] != y[..., 1])

# these are valid labels
labels = torch.tensor([
    (1, 3),
    (7, 4),
    (1, 4),
    (1, 5),
    (1, 6),
])
```

Example of floating point labels:
```python
def label_comparison_fn(x, y):
    return torch.abs(x - y) < 1

# these are valid labels
labels = torch.tensor([
    10.0,
    0.03,
    0.04,
    0.05,
])
```


### Warning for versions <= 0.9.97

The behavior of the ```k``` parameter described in the [Parameters](#parameters) section is for versions >= 0.9.98.

For versions <= 0.9.97, the behavior was:

* If ```k = None```, then ```k = min(1023, max(bincount(reference_labels)))```
* Otherwise ```k = k```