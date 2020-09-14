# Samplers
Samplers are just extensions of the torch.utils.data.Sampler class, i.e. they are passed to a PyTorch Dataloader. The purpose of samplers is to determine how batches should be formed. This is also where any offline pair or triplet miners should exist.


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