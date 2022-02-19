# Trainers
Trainers exist in this library because some metric learning algorithms are more than just loss or mining functions. Some algorithms require additional networks, data augmentations, learning rate schedules etc. The goal of the trainers module is to provide access to these type of metric learning algorithms. 

In general, trainers are used as follows:
```python
from pytorch_metric_learning import trainers
t = trainers.SomeTrainingFunction(*args, **kwargs)
t.train(num_epochs=10)
```


## BaseTrainer
All trainers extend this class and therefore inherit its ```__init__``` arguments.
```python
trainers.BaseTrainer(models,
					optimizers,
					batch_size,
					loss_funcs,
					mining_funcs,
					dataset,
					iterations_per_epoch=None,
					data_device=None,
					dtype=None,
					loss_weights=None,
					sampler=None,
					collate_fn=None,
					lr_schedulers=None,
					gradient_clippers=None,
					freeze_these=(),
					freeze_trunk_batchnorm=False,
					label_hierarchy_level=0,
					dataloader_num_workers=2,
					data_and_label_getter=None,
					dataset_labels=None,
					set_min_label_to_zero=False,
					end_of_iteration_hook=None,
					end_of_epoch_hook=None)
```

**Parameters**:

* **models**: A dictionary of the form: 
	* {"trunk": trunk_model, "embedder": embedder_model}
	* The "embedder" key is optional.
* **optimizers**: A dictionary mapping strings to optimizers. The base class does not require any specific keys. For example, you could provide an empty dictionary, in which case no optimization will happen. Or you could provide just an optimizer for your trunk_model. But most likely, you'll want to pass in: 
	* {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}.
* **batch_size**: The number of elements that are retrieved at each iteration.
* **loss_funcs**: A dictionary mapping strings to loss functions. The required keys depend on the training method, but all methods are likely to require at least: 
	* {"metric_loss": loss_func}.
* **mining_funcs**: A dictionary mapping strings to mining functions. Pass in an empty dictionary, or one or more of the following keys: 
	* {"subset_batch_miner": mining_func1, "tuple_miner": mining_func2}
* **dataset**: The dataset you want to train on. Note that training methods do not perform validation, so do not pass in your validation or test set.
* **data_device**: The device that you want to put batches of data on. If not specified, the trainer will put the data on any available GPUs.
* **dtype**: The type that the dataset output will be converted to, e.g. ```torch.float16```. If set to ```None```, then no type casting will be done.
* **iterations_per_epoch**: Optional. 
	* If you don't specify ```iterations_per_epoch```:
		* 1 epoch = 1 pass through the dataloader iterator. If ```sampler=None```, then 1 pass through the iterator is 1 pass through the dataset. 
		* If you use a sampler, then 1 pass through the iterator is 1 pass through the iterable returned by the sampler.
	* For samplers like ```MPerClassSampler``` or some offline mining method, the iterable returned might be very long or very short etc, and might not be related to the length of the dataset. The length of the epoch might vary each time the sampler creates a new iterable. In these cases, it can be useful to specify ```iterations_per_epoch``` so that each "epoch" is just a fixed number of iterations. The definition of epoch matters because there's various things like LR schedulers and hooks that depend on an epoch ending.
* **loss_weights**: A dictionary mapping loss names to numbers. Each loss will be multiplied by the corresponding value in the dictionary. If not specified, then no loss weighting will occur.
If not specified, then the original labels are used.
* **sampler**: The sampler used by the dataloader. If not specified, then random sampling will be used.
* **collate_fn**: The collate function used by the dataloader.
* **lr_schedulers**: A dictionary of PyTorch learning rate schedulers. Your keys should be strings of the form ```<model>_<step_type>```, where ```<model>``` is a key that comes from either the ```models``` or ```loss_funcs``` dictionary, and ```<step_type>``` is one of the following:
	* "scheduler_by_iteration" (will be stepped at every iteration)
	* "scheduler_by_epoch" (will be stepped at the end of each epoch)
	* "scheduler_by_plateau" (will step if accuracy plateaus. This requires you to write your own end-of-epoch hook, compute validation accuracy, and call ```trainer.step_lr_plateau_schedulers(validation_accuracy)```. Or you can use [HookContainer](logging_presets.md).)
	* Here are some example valid ```lr_scheduler``` keys: 
		* ```trunk_scheduler_by_iteration```
		* ```metric_loss_scheduler_by_epoch```
		* ```embedder_scheduler_by_plateau```
* **gradient_clippers**: A dictionary of gradient clipping functions. Each function will be called before the optimizers.
* **freeze_these**: Optional. A list or tuple of the names of models or loss functions that should have their parameters frozen during training. These models will have ```requires_grad``` set to False, and their optimizers will not be stepped. 
* **freeze_trunk_batchnorm**: If True, then the BatchNorm parameters of the trunk model will be frozen during training.
* **label_hierarchy_level**: If each sample in your dataset has multiple labels, then this integer argument can be used to select which "level" to use. This assumes that your labels are "2-dimensional" with shape (num_samples, num_hierarchy_levels). Leave this at the default value, 0, if your data does not have multiple labels per sample.
* **dataloader_num_workers**: The number of processes your dataloader will use to load data.
* **data_and_label_getter**: A function that takes the output of your dataset's ```__getitem__``` function, and returns a tuple of (data, labels). If None, then it is assumed that ```__getitem__``` returns (data, labels). 
* **dataset_labels**: The labels for your dataset. Can be 1-dimensional (1 label per datapoint) or 2-dimensional, where each row represents a datapoint, and the columns are the multiple labels that the datapoint has. Labels can be integers or strings. **This option needs to be specified only if ```set_min_label_to_zero``` is True.**
* **set_min_label_to_zero**: If True, labels will be mapped such that they represent their rank in the label set. For example, if your dataset has labels 5, 10, 12, 13, then at each iteration, these would become 0, 1, 2, 3. You should also set this to True if you want to use string labels. In that case, 'dog', 'cat', 'monkey' would get mapped to 1, 0, 2. If True, you must pass in ```dataset_labels``` (see above). The default is False.
* **end_of_iteration_hook**: This is an optional function that has one input argument (the trainer object), and performs some action (e.g. logging data) at the end of every iteration. Here are some things you might want to log:
	* ```trainer.losses```: this dictionary contains all loss values at the current iteration. 
	* ```trainer.loss_funcs``` and ```trainer.mining_funcs```: these dictionaries contain the loss and mining functions. 
		* Some loss and mining functions have attributes called ```_record_these``` or ```_record_these_stats```. These are lists of names of other attributes that might be useful to log. (The list of attributes might change depending on the value of [COLLECT_STATS](common_functions.md#collect_stats).) For example, the ```_record_these_stats``` list for ```BaseTupleMiner``` is ```["num_pos_pairs", "num_neg_pairs", "num_triplets"]```, so at each iteration you could log the value of ```trainer.mining_funcs["tuple_miner"].num_pos_pairs```. To accomplish this programmatically, you can use [record-keeper](https://github.com/KevinMusgrave/record-keeper). Or you can do it yourself: first check if the object has ```_record_these``` or ```_record_these_stats```, and use the python function ```getattr``` to retrieve the specified attributes. 
	* If you want ready-to-use hooks, take a look at the [logging_presets module](logging_presets.md).
* **end_of_epoch_hook**: This is an optional function that operates like ```end_of_iteration_hook```, except this occurs at the end of every epoch, so this might be a suitable place to run validation and save models. 
	* To end training early, your hook should return the boolean value False. Note, it must specifically ```return False```, not ```None```, ```0```, ```[]``` etc.
	* For this hook, you might want to access the following dictionaries: ```trainer.models```, ```trainer.optimizers```, ```trainer.lr_schedulers```, ```trainer.loss_funcs```, and ```trainer.mining_funcs```.
	* If you want ready-to-use hooks, take a look at the [logging_presets module](logging_presets.md).

## MetricLossOnly
This trainer just computes a metric loss from the output of your embedder network. See [the example notebook](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb).
```python
trainers.MetricLossOnly(*args, **kwargs)
```

**Requirements**:

* **models**: Must have the following form:
	* {"trunk": trunk_model}
	* Optionally include "embedder": embedder_model
* **loss_funcs**: Must have the following form:
	* {"metric_loss": loss_func}


## TrainWithClassifier
This trainer is for the case where your architecture is trunk -> embedder -> classifier. It applies a metric loss to the output of the embedder network, and a classification loss to the output of the classifier network. See [the example notebook](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TrainWithClassifier.ipynb).
```python
trainers.TrainWithClassifier(*args, **kwargs)
```
**Requirements**:

* **models**: Must have the following form: 
	* {"trunk": trunk_model, "classifier": classifier_model}
	* Optionally include "embedder": embedder_model
* **loss_funcs**: Must have the following form:
	* {"metric_loss": loss_func1, "classifier_loss": loss_func2}

## CascadedEmbeddings

This trainer is a generalization of [Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf). It splits the output of your embedder network, computing a separate loss for each section. In other words, the output of your embedder should be the concatenation of your cascaded models. See [the example notebook](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/CascadedEmbeddings.ipynb).

```python
trainers.CascadedEmbeddings(embedding_sizes, *args, **kwargs)
``` 

**Parameters**:

* embedding_sizes: A list of integers, which represent the size of the output of each cascaded model.

**Requirements**:

* **models**: Must have the following form:

	* {"trunk": trunk_model}
		* Optionally include "embedder": embedder_model
		* Optionally include key:values of the form "classifier_%d": classifier_model_%d. The integer appended to "classifier_" represents the cascaded model that the classifier will be appended to. For example, if the dictionary has classifier_0 and classifier_2, then the 0th and 2nd cascaded models will have classifier_model_0 and classifier_model_2 appended to them.

* **loss_funcs**: Must have the following form:
	* {"metric_loss_%d": metric_loss_func_%d}
		* Optionally include key:values of the form "classifier_loss_%d": classifier_loss_func_%d. The appended integer represents which cascaded model the loss applies to.

* **mining_funcs**: Must have the following form:
	* {"tuple_miner_%d": mining_func_%d}
		* Optionally include "subset_batch_miner": subset_batch_miner

## DeepAdversarialMetricLearning
This is an implementation of [Deep Adversarial Metric Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf). See [the example notebook](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DeepAdversarialMetricLearning.ipynb).
```python
trainers.DeepAdversarialMetricLearning(metric_alone_epochs=0,
		        					g_alone_epochs=0,
		        					g_triplets_per_anchor=100,
		        					*args,
		        					**kwargs):
``` 

**Parameters**:

* **metric_alone_epochs**: At the beginning of training, this many epochs will consist of only the metric_loss.
* **g_alone_epochs**: After metric_alone_epochs, this many epochs will consist of only the adversarial generator loss.
* **g_triplets_per_anchor**: The number of real triplets per sample that should be passed into the generator. For each real triplet, the generator will output a synthetic triplet.

**Requirements**:

* **models**: Must have the following form:
	* {"trunk": trunk_model, "generator": generator_model}
		* Optionally include "embedder": embedder_model
		* Optionally include "classifier": classifier_model
		* The input size to the generator must be 3\*(size of trunk_model output). The output size must be (size of trunk_model output).

* **loss_funcs**: Must have the following form:
	* {"metric_loss": metric_loss, "g_adv_loss": g_adv_loss, "synth_loss": synth_loss}
		* Optionally include "classifier_loss": classifier_loss
		* metric_loss applies to the embeddings of real data.
		* g_adv_loss is the adversarial generator loss. **Currently, only TripletMarginLoss is supported**
		* synth_loss applies to the embeddings of the synthetic generator triplets.

* **loss_weights**: Must be one of the following:
	* None
	* {"metric_loss": weight1, "g_adv_loss": weight2, "synth_loss": weight3, "g_reg_loss": weight4, "g_hard_lss": weight5}
		* Optionally include "classifier_loss": classifier_loss
		* "g_reg_loss" and "g_hard_loss" refer to the regularization losses described in the paper.
		

## UnsupervisedEmbeddingsUsingAugmentations
This is an implementation of a general approach that has been used in recent unsupervised learning papers, e.g. [Unsupervised Embedding Learning via Invariant and Spreading
Instance Feature
](https://arxiv.org/pdf/1904.03436.pdf) and [Unsupervised Deep Metric Learning via Auxiliary Rotation Loss](https://arxiv.org/abs/1911.07072). The idea is that augmented versions of a datapoint should be close to each other in the embedding space.
```python
trainers.UnsupervisedEmbeddingsUsingAugmentations(transforms, data_and_label_setter=None, *args, **kwargs)
```

**Parameters**:

* **transforms**: A list of transforms. For every sample in a batch, each transform will be applied to the sample. If there are N transforms and the batch size is B, then there will be a total of B*N augmented samples. 
* **data_and_label_setter**: A function that takes in a tuple of (augmented_data, pseudo_labels) and outputs whatever is expected by self.data_and_label_getter.


## TwoStreamMetricLoss
This trainer is the same as [MetricLossOnly](trainers.md#metriclossonly) but operates on separate streams of anchors and positives/negatives.
The supplied **dataset** must return ```(anchor, positive, label)```.
Given a batch of ```(anchor, positive, label)```, triplets are formed using ```anchor``` as the anchor, and ```positive``` as either the positive or negative. See [the example notebook](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TwoStreamMetricLoss.ipynb).
```python
trainers.TwoStreamMetricLoss(*args, **kwargs)
```
**Requirements**:

* **models**: Must have the following form:
	* {"trunk": trunk_model}
	* Optionally include "embedder": embedder_model
* **loss_funcs**: Must have the following form:
	* {"metric_loss": loss_func}
* **mining_funcs**: Only tuple miners are supported