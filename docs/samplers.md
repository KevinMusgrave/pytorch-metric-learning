# Samplers
Samplers are just extensions of the torch.utils.data.Sampler class, i.e. they are passed to a PyTorch Dataloader (specifically as _sampler_ argument, unless otherwise mentioned). 
The purpose of samplers is to determine how batches should be formed. This is also where any offline pair or triplet miners should exist.


## MPerClassSampler
At every iteration, this will return ```m``` samples per class, assuming that the batch size is a multiple of ```m```. For example, if your dataloader's batch size is 100, and ```m``` = 5, then 20 classes with 5 samples each will be returned. Note that if ```batch_size``` is not specified, then most batches will have ```m``` samples per class, but it's not guaranteed for every batch.
```python
samplers.MPerClassSampler(labels, m, batch_size=None, length_before_new_iter=100000)
```
**Parameters**:

* **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
* **m**: The number of samples per class to fetch at every iteration. If a class has less than m samples, then there will be duplicates in the returned batch.
* **batch_size**: Optional. If specified, then every batch is guaranteed to have ```m``` samples per class. There are a few restrictions on this value:
	* ```batch_size``` must be a multiple of ```m```
	* ```length_before_new_iter >= batch_size``` must be true
	* ```m * (number of unique labels) >= batch_size``` must be true
* **length_before_new_iter**: How many iterations will pass before a new iterable is created.


## HierarchicalSampler
Implementation of the sampler used in [Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf).

It will do the following per batch:

 - Randomly select X super classes.
 - For each super class, randomly select Y samples from Z classes, such that Y * Z equals the batch size divided by X.

(X, Y, and the batch size are controllable parameters. See below for details.)

This is a [BatchSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler), so you should pass it into your dataloader as the ```batch_sampler``` parameter.

```python
samplers.HierarchicalSampler(
		labels,
        batch_size,
        samples_per_class,
        batches_per_super_tuple=4,
        super_classes_per_batch=2,
        inner_label=0,
        outer_label=1,
    )
```
**Parameters**:

* **labels**: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels.
* **batch_size**: because this is a BatchSampler the batch size must be specified.
	* ```batch_size``` must be a multiple of ```super_classes_per_batch``` and ```samples_per_class```
* **samples_per_class**: number of samples per class per batch. Corresponds to Y in the above explanation. You can also set this to "all" to use all elements of a class, but this is suitable only for few-shot datasets.
* **batches_per_super_tuple**: number of batches to create for each tuple of super classes. This affects the length of the iterator returned by the sampler.
* **super_classes_per_batch**: the number of super classes per batch. Corresponds to X in the above explanation.
* **inner_label**: column index of ```labels``` corresponding to classes.
* **outer_label**: column index of ```labels``` corresponding to super classes.


## TuplesToWeightsSampler
This is a simple offline miner. It does the following:

1. Take a random subset of the dataset, if you provide ```subset_size```.
2. Use a specified miner to mine tuples from the subset dataset.
3. Compute weights based on how often each element appears in the mined tuples.
4. Randomly sample, using the weights as probabilities.

```python
samplers.TuplesToWeightsSampler(model, 
								miner, 
								dataset, 
								subset_size=None, 
								**tester_kwargs)
```

**Parameters**:

* **model**: This model will be used to compute embeddings.
* **miner**: This miner will find hard tuples from the computed embeddings.
* **dataset**: The dataset you want to sample from.
* **subset_size**: Optional. If ```None```, then the entire dataset will be mined, and the iterable will have length ```len(dataset)```. Most likely though, you will run out of memory if you do this. So to avoid that, set ```subset_size``` to a number of embeddings that can be passed to the miner without running out of memory. Then a random subset of ```dataset``` of size ```subset_size``` will be used for mining. The iterable will also have length ```subset_size```.
* **tester_kwargs**: Any other keyword options will be passed to [BaseTester](testers.md#basetester), which is used internally to compute embeddings. This allows you to set things like ```dataloader_num_workers``` etc, if you want to.


## FixedSetOfTriplets
When initialized, this class creates a fixed set of triplets. This is useful for determining the performance of algorithms in cases where the only ground truth data is a set of triplets.
```python
samplers.FixedSetOfTriplets(labels, num_triplets)
```

**Parameters**:

* **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
* **num_triplets**: The number of triplets to create.
