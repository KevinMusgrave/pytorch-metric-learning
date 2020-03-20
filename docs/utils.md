# Utils
Utils contains various helpful functions and classes. Some of the more useful features are listed here.


## logging_presets
This module contains ready-to-use hooks for logging data, validating and saving your models, and early stoppage during training. It requires the record-keeper and tensorboard packages, which can be installed with pip:

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
								iterations_per_epoch,
								train_dataset,
								sampler=sampler,
								end_of_iteration_hook=hooks.end_of_iteration_hook,
								end_of_epoch_hook=end_of_epoch_hook)

trainer.train(num_epochs=num_epochs)
```
With the provided hooks, data from both the training and validation stages will be saved in csv, sqlite, and tensorboard format, and models and optimizers will be saved in the specified model folder. See [this script](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/example_MetricLossOnly.py) for a complete example. Read the next section to learn more about the provided hooks.

### HookContainer
This class contains ready-to-use hooks to be used by trainers and testers.

```python
import pytorch_metric_learning.utils.logging_presets as LP
LP.HookContainer(record_keeper, record_group_name_prefix=None, primary_metric="mean_average_precision_at_r", validation_split_name="val")
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