# Utils
Utils contains various helpful functions and classes. Some of the more useful features are listed here.

## Accuracy Calculations
The ```accuracy_calculator``` module contains functions for determining the quality of an embedding space.
### AccuracyCalculator

This class computes several accuracy metrics given a query and reference embeddings. It can be easily extended to create custom accuracy metrics.

```python
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
AccuracyCalculator(include=(), exclude=(), average_per_class=False)
```
**Parameters**:

* **include**: Optional. A list or tuple of strings, which are the names of metrics you want to calculate. If left empty, all default metrics will be calculated.
* **exclude**: Optional. A list or tuple of strings, which are the names of metrics you **do not** want to calculate.
* **average_per_class**: If True, the average accuracy per class is computed, and then the average of those averages is returned. This can be useful if your dataset has unbalanced classes. If False, the global average will be returned.

**Getting accuracy**:

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
* **query**: A 2D numpy array of size ```(Nq, D)```, where Nq is the number of query samples. For each query sample, nearest neighbors are retrieved and accuracy is computed.
* **reference**: A 2D numpy array of size ```(Nr, D)```, where Nr is the number of reference samples. This is where nearest neighbors are retrieved from.
* **query_labels**: A 1D numpy array of size ```(Nq)```. Each element should be an integer representing the sample's label.
* **reference_labels**: A 1D numpy array of size ```(Nr)```. Each element should be an integer representing the sample's label. 
* **embeddings_come_from_same_source**: Set to True if ```query``` is a subset of ```reference``` or if ```query is reference```. Set to False otherwise.
* **include**: Optional. A list or tuple of strings, which are the names of metrics you want to calculate. If left empty, all metrics specified during initialization will be calculated.
* **exclude**: Optional. A list or tuple of strings, which are the names of metrics you do not want to calculate.

**Adding custom accuracy metrics**

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
    "embeddings_come_from_same_source": e,  # True if query is reference, or if query is a subset of reference.
    "label_counts": label_counts,           # a dictionary mapping from reference labels to the number of times they occur
    "knn_labels": knn_labels}               # A 2d array where each row is the labels of the nearest neighbors of each query. The neighbors are retrieved from the reference set
```

If your method requires cluster labels, then append your method's name to the ```requires_clustering``` list, via ```super()```. Then, if any of your methods need cluster labels, ```self.get_cluster_labels()``` will be called, and the kwargs will include ```cluster_labels```. Likewise for computing k-nearest-neighbors.

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

## Logging Presets
The ```logging_presets``` module contains ready-to-use hooks for logging data, validating and saving your models, and early stoppage during training. It requires the record-keeper and tensorboard packages, which can be installed with pip:

```pip install record-keeper tensorboard```

Here's how you can use it in conjunction with a trainer and tester:
```python
import pytorch_metric_learning.utils.logging_presets as LP
log_folder, tensorboard_folder = "example_logs", "example_tensorboard"
record_keeper, _, _ = LP.get_record_keeper(log_folder, tensorboard_folder)
hooks = LP.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "example_saved_models"

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook)
end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
trainer = trainers.MetricLossOnly(models,
								optimizers,
								batch_size,
								loss_funcs,
								mining_funcs,
								train_dataset,
								sampler=sampler,
								end_of_iteration_hook=hooks.end_of_iteration_hook,
								end_of_epoch_hook=end_of_epoch_hook)

trainer.train(num_epochs=num_epochs)
```
With the provided hooks, data from both the training and validation stages will be saved in csv, sqlite, and tensorboard format, and models and optimizers will be saved in the specified model folder. See [the example notebooks](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples) for complete examples. Read the next section to learn more about the provided hooks.

### HookContainer
This class contains ready-to-use hooks to be used by trainers and testers.

```python
import pytorch_metric_learning.utils.logging_presets as LP
LP.HookContainer(record_keeper, 
	record_group_name_prefix=None, 
	primary_metric="mean_average_precision_at_r", 
	validation_split_name="val",
	save_custom_figures=False)
```

**Parameters**:

* **record_keeper**: A ```record-keeper``` object. Install: ```pip install record-keeper tensorboard```.
* **record_group_name_prefix**: A string which will be prepended to all record names and tensorboard tags.
* **primary_metric**: A string that specifies the accuracy metric which will be used to determine the best checkpoint. Must be one of:
    * mean_average_precision_at_r
	* r_precision
	* precision_at_1
	* NMI
* **validation_split_name**: Optional. Default value is "val". The name of your validation set in ```dataset_dict```.
* **save_custom_figures**: Optional. If True, records that consist of a tensor at each iteration (rather than just a scalar), will be plotted on tensorboard.

**Important functions**:

* **end_of_iteration_hook**: This function records data about models, optimizers, and loss and mining functions. You can pass this function directly into a trainer object.
* **end_of_epoch_hook**: This function runs validation and saves models. This function returns the actual hook, i.e. you must pass in the following arguments to obtain the hook.
	* **tester**: A [tester](testers.md) object.
	* **dataset_dict**: A dictionary mapping from split names to PyTorch datasets. For example: ```{"train": train_dataset, "val": val_dataset}```
	* **model_folder**: A string which is the folder path where models, optimizers etc. will be saved. 
	* **test_interval**: Optional. Default value is 1. Validation will be run every ```test_interval``` epochs.
	* **patience**: Optional. Default value is None. If not None, training will end early if ```epoch - best_epoch > patience```.
	* **test_collate_fn**: Optional. Default value is None. This is the collate function used by the dataloader during testing. 
* **end_of_testing_hook**: This function records accuracy metrics. You can pass this function directly into a tester object.

**Useful methods**:

Getting loss history:
```python
# Get a dictionary mapping from loss names to lists
loss_histories = hooks.get_loss_history() 

# You can also specify which loss histories you want
# It will still return a dictionary. In this case, the dictionary will contain only "total_loss"
loss_histories = hooks.get_loss_history(loss_names=["total_loss"])
```

Getting accuracy history
```python
# The first argument is the tester object. The second is the split name.
# Get a dictionary containing the keys "epoch" and the primary metric
# The values are lists
acc_histories = hooks.get_accuracy_history(tester, "val")

# Get all accuracy histories
acc_histories = hooks.get_accuracy_history(tester, "val", return_all_metrics=True)

# Get a specific set of accuracy histories
acc_histories = hooks.get_accuracy_history(tester, "val", metrics=["AMI", "NMI"])
``` 