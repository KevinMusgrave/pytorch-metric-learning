# Samplers

Samplers are just extensions of the torch.utils.data.Sampler class, i.e. they are passed to a PyTorch Dataloader (specifically as the `sampler` argument, unless otherwise mentioned).
The purpose of samplers is to determine how batches should be formed. This is also where any offline pair or triplet miners should exist.

## MPerClassSampler

At every iteration, this will return `m` samples per class, assuming that the batch size is a multiple of `m`. For example, if your dataloader's batch size is 100, and `m` = 5, then 20 classes with 5 samples each will be returned. Note that if `batch_size` is not specified, then most batches will have `m` samples per class, but it's not guaranteed for every batch.

```python
samplers.MPerClassSampler(labels, m, batch_size=None, length_before_new_iter=100000)
```

**Parameters**:

- **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
- **m**: The number of samples per class to fetch at every iteration. If a class has less than m samples, then there will be duplicates in the returned batch.
- **batch_size**: Optional. If specified, then every batch is guaranteed to have `m` samples per class. There are a few restrictions on this value:
  - `batch_size` must be a multiple of `m`
  - `length_before_new_iter >= batch_size` must be true
  - `m * (number of unique labels) >= batch_size` must be true
- **length_before_new_iter**: How many iterations will pass before a new iterable is created.

## HierarchicalSampler

Implementation of the sampler used in [Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf).

It will do the following per batch:

- Randomly select X super classes.
- For each super class, randomly select Y samples from Z classes, such that Y * Z equals the batch size divided by X.

(X, Y, and the batch size are controllable parameters. See below for details.)

This is a [BatchSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler), so you should pass it into your dataloader as the `batch_sampler` parameter.

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

- **labels**: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels.
- **batch_size**: because this is a BatchSampler the batch size must be specified.
  - `batch_size` must be a multiple of `super_classes_per_batch` and `samples_per_class`
- **samples_per_class**: number of samples per class per batch. Corresponds to Y in the above explanation. You can also set this to "all" to use all elements of a class, but this is suitable only for few-shot datasets.
- **batches_per_super_tuple**: number of batches to create for each tuple of super classes. This affects the length of the iterator returned by the sampler.
- **super_classes_per_batch**: the number of super classes per batch. Corresponds to X in the above explanation.
- **inner_label**: column index of `labels` corresponding to classes.
- **outer_label**: column index of `labels` corresponding to super classes.

## Mining-weighted sampling

Use the numerical tuple-frequency utility with PyTorch's `WeightedRandomSampler`.
The [agent-facing recipe](recipes/training.md#mining-weighted-sampling-contract)
shows embedding extraction, subset-index mapping, and seeded sampling.

## FixedSetOfTriplets

When initialized, this class creates a fixed set of triplets. This is useful for determining the performance of algorithms in cases where the only ground truth data is a set of triplets.

```python
samplers.FixedSetOfTriplets(labels, num_triplets)
```

**Parameters**:

- **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
- **num_triplets**: The number of triplets to create.
