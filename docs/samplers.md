# Samplers
Samplers are just extensions of the torch.utils.data.Sampler class, i.e. they are passed to a PyTorch Dataloader. The purpose of samplers is to determine how batches should be formed. This is also where any offline pair or triplet miners should exist.


## MPerClassSampler
At every iteration, this will return ```m``` samples per class, assuming that the batch size is a multiple of ```m```. For example, if your dataloader's batch size is 100, and ```m``` = 5, then 20 classes with 5 samples each will be returned.
```python
samplers.MPerClassSampler(labels, m, length_before_new_iter=100000)
```
**Parameters**:

* **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
* **m**: The number of samples per class to fetch at every iteration. If a class has less than m samples, then there will be duplicates in the returned batch.
* **length_before_new_iter**: How many iterations will pass before a new iterable is created.

## FixedSetOfTriplets
When initialized, this class creates a fixed set of triplets. This is useful for determining the performance of algorithms in cases where the only ground truth data is a set of triplets.
```python
samplers.FixedSetOfTriplets(labels, num_triplets)
```

**Parameters**:

* **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
* **num_triplets**: The number of triplets to create.