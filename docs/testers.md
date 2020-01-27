# Testers
Testers take your model and dataset, and compute nearest-neighbor based accuracy metrics. Note that the testers require the [faiss package](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md), which you can install with conda.

In general, testers are used as follows:
```python
from pytorch_metric_learning import testers
t = testers.SomeTestingFunction(**kwargs)
dataset_dict = {"train": train_dataset, "val": val_dataset}
tester.test(dataset_dict, epoch, trunk, embedder)
```

## BaseTester
All trainers extend this class and therefore inherit its _\_\_init\_\__ arguments.
```python
testers.BaseTester(reference_set="compared_to_self", 
					normalize_embeddings=True, 
					use_trunk_output=False, 
                    batch_size=32, 
                    dataloader_num_workers=32, 
                    metric_for_best_epoch="mean_average_r_precision", 
                    pca=None, 
                    data_device=None, 
					size_of_tsne=0, 
					data_and_label_getter=None,
                    label_hierarchy_level=0,
                    end_of_testing_hook=None)
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
* **metric_for_best_epoch**: The performance metric that will be used to determine which model is best. Must be one of the following:
	* "NMI"
	* "precision_at_1"
	* "r_precision"
	* "mean_average_r_precision"
	* **Note**: this assumes you call the "test" method multiple times with different models, and that the record_keeper argument is not None. The input to record_keeper must be a [record-keeper object](https://github.com/KevinMusgrave/record-keeper).
* **pca**: The number of dimensions that your embeddings will be reduced to, using PCA. The default is None, meaning PCA will not be applied.
* **data_device**: Which gpu to use for the loaded dataset samples. If None, then the gpu or cpu will be used (whichever is available).
* **size_of_tsne**: The number of samples to use for a t-sne plot (requires a record_keeper). If 0, then no t-sne plot will be created.
* **data_and_label_getter**: A function that takes the output of your dataset's _\_\_getitem\_\__ function, and returns a tuple of (data, labels). If None, then it is assumed that _\_\_getitem\_\__ returns (data, labels). 
* **label_hierarchy_level**: If each sample in your dataset has multiple labels, then this integer argument can be used to select which "level" to use. This assumes that your labels are "2-dimensional" with shape (num_samples, num_hierarchy_levels). Leave this at the default value, 0, if your data does not have multiple labels per sample.
* **end_of_testing_hook**: This is a function that has one input argument, the tester object, and performs some action (e.g. logging data) at the end of testing. If None, then this parameter has no effect. If you want ready-to-use hooks, take a look at the HookContainer class in [logging_presets](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/pytorch_metric_learning/utils/logging_presets.py), and see [the end of this script](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/example_MetricLossOnly.py) for an example of how to use the hooks.

## GlobalEmbeddingSpaceTester
Computes nearest neighbors by looking at all points in the embedding space. This is probably the tester you are looking for.
```python
testers.GlobalEmbeddingSpaceTester(**kwargs)
```

## WithSameParentLabelTester
This assumes there is a label hierarchy. For each sample, the search space is narrowed by only looking at sibling samples, i.e. samples with the same parent label. For example, consider a dataset with 4 fine-grained classes {cat, dog, car, truck}, and 2 coarse-grained classes {animal, vehicle}. The nearest neighbor search for cats and dogs will consist of animals, and the nearest-neighbor search for cars and trucks will consist of vehicles.
```python
testers.WithSameParentLabelTester(**kwargs)
``` 