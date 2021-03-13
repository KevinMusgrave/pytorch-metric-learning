# Accuracy Calculation

The AccuracyCalculator class computes several accuracy metrics given a query and reference embeddings. It can be easily extended to create custom accuracy metrics.

```python
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
AccuracyCalculator(include=(),
                    exclude=(),
                    avg_of_avgs=False,
                    k=None,
                    label_comparison_fn=None)
```
### Parameters

* **include**: Optional. A list or tuple of strings, which are the names of metrics you want to calculate. If left empty, all default metrics will be calculated.
* **exclude**: Optional. A list or tuple of strings, which are the names of metrics you **do not** want to calculate.
* **avg_of_avgs**: If True, the average accuracy per class is computed, and then the average of those averages is returned. This can be useful if your dataset has unbalanced classes. If False, the global average will be returned.
* **k**: If set, this number of nearest neighbors will be retrieved for metrics that require k-nearest neighbors. If None, the value of k will be determined as follows:
    * First, count the number of occurrences of each label in ```reference_labels```, and set k to the maximum value. For example, if ```reference_labels``` is ```[0, 0, 1, 1, 1, 1, 2]```, then ```k = 4```.
    * Then set ```k = min(1023, k)```. This is done because faiss (the library that computes k-nn) has a limit on the value of k.
    * After k is set, the ```k+1``` nearest neighbors are found for every query sample. When the query and reference come from the same source, the 1st nearest neighbor is discarded since that "neighbor" is actually the query sample. When the query and reference come from different sources, the ```k+1``` neighbor is discarded.
* **label_comparison_fn**: A function that compares two torch arrays of labels and returns a boolean array. The default is ```torch.eq```. If a custom function is used, then you must exclude clustering based metrics ("NMI" and "AMI"). The following is an example of a custom function for two-dimensional labels. It returns ```True``` if the 0th column matches, and the 1st column does **not** match:
```python
def example_label_comparison_fn(x, y):
    return (x[:, 0] == y[:, 0]) & (x[:, 1] != y[:, 1])

AccuracyCalculator(exclude=("NMI", "AMI"), 
                    label_comparison_fn=example_label_comparison_fn)
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

    - [See section 3.2 of A Metric Learning Reality Check](https://arxiv.org/pdf/2003.08505.pdf)



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