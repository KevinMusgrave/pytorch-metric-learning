# Testers
Testers take your model and dataset, and compute nearest-neighbor based accuracy metrics. Note that the testers require the [faiss package](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md), which you can install with conda.

In general, testers are used as follows:
```python
from pytorch_metric_learning import testers
t = testers.SomeTestingFunction(*args, **kwargs)
dataset_dict = {"train": train_dataset, "val": val_dataset}
tester.test(dataset_dict, epoch, model)

# Or if your model is composed of a trunk + embedder
tester.test(dataset_dict, epoch, trunk, embedder)
```
You can perform custom actions by writing an end-of-testing hook (see the documentation for [BaseTester](#basetester)), and you can access the test results directly via the ```all_accuracies``` attribute:
```python
print(tester.all_accuracies)
```
This will print out a dictionary of accuracy metrics, per dataset split. You'll see something like this:
```python
{"train": {"AMI_level0": 0.53, ...}, "val": {"AMI_level0": 0.44, ...}}
```
Each of the accuracy metric names is appended with ```level0```, which refers to the 0th label hierarchy level (see the documentation for [BaseTester](#basetester)). This is only relevant if you're dealing with multi-label datasets. 

For an explanation of the default accuracy metrics, see the [AccuracyCalculator documentation](accuracy_calculation.md#explanations-of-the-default-accuracy-metrics).


## BaseTester
All trainers extend this class and therefore inherit its ```__init__``` arguments.
```python
testers.BaseTester(reference_set="compared_to_self", 
					normalize_embeddings=True, 
					use_trunk_output=False, 
                    batch_size=32, 
                    dataloader_num_workers=32, 
                    pca=None, 
                    data_device=None, 
					data_and_label_getter=None,
                    label_hierarchy_level=0,
                    end_of_testing_hook=None,
			        dataset_labels=None,
			        set_min_label_to_zero=False,
			        accuracy_calculator=None,
			        visualizer=None,
        			visualizer_hook=None)
```

**Parameters**:

* **reference_set**: Must be one of the following:
	* "compared_to_self": each dataset split will refer to itself to find nearest neighbors.
	* "compared_to_sets_combined": each dataset split will refer to all provided splits to find nearest neighbors.
 	* "compared_to_training_set": each dataset will refer to the training set to find nearest neighbors.
* **normalize_embeddings**: If True, embeddings will be normalized to Euclidean norm of 1 before nearest neighbors are computed.
* **use_trunk_output**: If True, the output of the trunk_model will be used to compute nearest neighbors, i.e. the output of the embedder model will be ignored.
* **batch_size**: How many dataset samples to process at each iteration when computing embeddings.
* **dataloader_num_workers**: How many processes the dataloader will use.
* **pca**: The number of dimensions that your embeddings will be reduced to, using PCA. The default is None, meaning PCA will not be applied.
* **data_device**: Which gpu to use for the loaded dataset samples. If None, then the gpu or cpu will be used (whichever is available).
* **data_and_label_getter**: A function that takes the output of your dataset's ```__getitem__``` function, and returns a tuple of (data, labels). If None, then it is assumed that ```__getitem__``` returns (data, labels). 
* **label_hierarchy_level**: If each sample in your dataset has multiple labels, then this integer argument can be used to select which "level" to use. This assumes that your labels are "2-dimensional" with shape (num_samples, num_hierarchy_levels). Leave this at the default value, 0, if your data does not have multiple labels per sample.
* **end_of_testing_hook**: This is an optional function that has one input argument (the tester object) and performs some action (e.g. logging data) at the end of testing.
	* You'll probably want to access the accuracy metrics, which are stored in ```tester.all_accuracies```. This is a nested dictionary with the following format: ```tester.all_accuracies[split_name][metric_name] = metric_value```
	* If you want ready-to-use hooks, take a look at the [logging_presets module](logging_presets.md).
* **dataset_labels**: The labels for your dataset. Can be 1-dimensional (1 label per datapoint) or 2-dimensional, where each row represents a datapoint, and the columns are the multiple labels that the datapoint has. Labels can be integers or strings. **This option needs to be specified only if ```set_min_label_to_zero``` is True.**
* **set_min_label_to_zero**: If True, labels will be mapped such that they represent their rank in the label set. For example, if your dataset has labels 5, 10, 12, 13, then at each iteration, these would become 0, 1, 2, 3. You should also set this to True if you want to use string labels. In that case, 'dog', 'cat', 'monkey' would get mapped to 1, 0, 2. If True, you must pass in ```dataset_labels``` (see above). The default is False.
* **accuracy_calculator**: Optional. An object that extends [AccuracyCalculator](accuracy_calculation.md). This will be used to compute the accuracy of your model. By default, AccuracyCalculator is used.
* **visualizer**: Optional. An object that has implemented the ```fit_transform``` method, as done by [UMAP](https://github.com/lmcinnes/umap) and many scikit-learn functions. For example, you can set ```visualizer = umap.UMAP()```. The object's ```fit_transform``` function should take in a 2D array of embeddings, and reduce the dimensionality, such that calling ```visualizer.fit_transform(embeddings)``` results in a 2D array of size (N, 2).
* **visualizer_hook**: Optional. This function will be passed the following args. You can do whatever you want in this function, but the reason it exists is to allow you to save a plot of the embeddings etc.
	* visualizer: The visualizer object that you passed in.
	* embeddings: The dimensionality reduced embeddings.
	* label: The corresponding labels for each embedding.
	* split_name: The name of the split (train, val, etc.)
	* keyname: The name of the dictionary key where the embeddings and labels are stored. (The dictionary is self.dim_reduced_embeddings[split_name][keyname].)
	* epoch: The epoch for which the embeddings are being computed.


## GlobalEmbeddingSpaceTester
Computes nearest neighbors by looking at all points in the embedding space (rather than a subset). This is probably the tester you are looking for. To see it in action, check one of the [example notebooks](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples)
```python
testers.GlobalEmbeddingSpaceTester(*args, **kwargs)
```

## WithSameParentLabelTester
This assumes there is a label hierarchy. For each sample, the search space is narrowed by only looking at sibling samples, i.e. samples with the same parent label. For example, consider a dataset with 4 fine-grained classes {cat, dog, car, truck}, and 2 coarse-grained classes {animal, vehicle}. The nearest neighbor search for cats and dogs will consist of animals, and the nearest-neighbor search for cars and trucks will consist of vehicles.
```python
testers.WithSameParentLabelTester(*args, **kwargs)
``` 

## GlobalTwoStreamEmbeddingSpaceTester
This is the corresponding tester for [TwoStreamMetricLoss](trainers.md#twostreammetricloss). The supplied **dataset** must return ```(anchor, positive, label)```.
```python
testers.GlobalTwoStreamEmbeddingSpaceTester(*args, **kwargs)
``` 
**Requirements**:

* **reference_set**: This must be left at the default value of ```"compared_to_self"```.