# Samplers
Samplers are just extensions of the torch.utils.data.Sampler class, i.e. they are passed to a PyTorch Dataloader. The purpose of samplers is to determine how batches should be formed. This is also where any offline pair or triplet miners should exist.


## MPerClassSampler
At every iteration, this will return _m_ samples per class, assuming that the batch size is a multiple of _m_. For example, if your dataloader's batch size is 100, and _m_ = 5, then 20 classes with 5 samples each will be returned.
```python
samplers.MPerClassSampler(labels_to_indices, m, hierarchy_level=0)
```
**Parameters**:

* **labels_to_indices**: A list of dictionaries, where each dictionary maps labels to lists of indices that have that label. If your dataset has 1 label per element, then the list of dictionaries should contain just 1 dictionary.
* **m**: The number of samples per class to fetch at every iteration. If a class has less than m samples, then there will be duplicates in the returned batch.
* **hierarchy_level**: This is for multi-label datasets, and it indiates which level of labels will be used to form each batch. The default is 0, because most use-cases will have 1 label per datapoint. But for example, the iNaturalist dataset has 7 labels per datapoint, in which case _hierarchy_level_ could be set to a number between 0 and 6.


## FixedSetOfTriplets
When initialized, this class creates a fixed set of triplets. This is useful for determining the performance of algorithms in cases where the only ground truth data is a set of triplets.
```python
samplers.FixedSetOfTriplets(labels_to_indices, num_triplets, hierarchy_level=0)
```

**Parameters**:

* **labels_to_indices**: A list of dictionaries, where each dictionary maps labels to lists of indices that have that label. If your dataset has N labels per datapoint, then the list of dictionaries should contain N dictionaries.
* **num_triplets**: The number of triplets to create.
* **hierarchy_level**: This is for multi-label datasets, and it indiates which level of labels will be used to form each batch. The default is 0, because most use-cases will have 1 label per datapoint. But for example, the iNaturalist dataset has 7 labels per datapoint, in which case _hierarchy_level_ could be set to a number between 0 and 6.
