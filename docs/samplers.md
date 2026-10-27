# Samplers

Samplers decide which dataset indices form each batch. `MPerClassSampler` and
`FixedSetOfTriplets` yield one dataset index at a time; pass them to
`DataLoader(..., sampler=...)`. `HierarchicalSampler` yields whole batches; pass
it as `DataLoader(..., batch_sampler=...)` and omit `batch_size` and `shuffle`.

## Randomness and coverage

Every sampler takes a keyword-only `generator: torch.Generator | None`. With
`None`, sampling uses PyTorch's global random number generator, so
`torch.manual_seed` makes it repeatable; a dedicated generator isolates sampler
randomness from model initialization and dropout. Each new iterator draws fresh
random batches, and two samplers given generators with the same seed produce the
same sequence of epochs. Samplers run in the main process, so `DataLoader`
worker seeding does not affect them.

These samplers balance labels within batches. They sample classes and examples
independently for each batch, with replacement across batches, so an iterator
is not an epoch that visits every example or class exactly once. A class with
fewer members than requested is sampled with replacement within its batch.
Use `length_before_new_iter` to choose how many indices one `MPerClassSampler`
iterator yields.

## MPerClassSampler

At every iteration, this will return `m` samples per class, assuming that the batch size is a multiple of `m`. For example, if your dataloader's batch size is 100, and `m` = 5, then 20 classes with 5 samples each will be returned. Note that if `batch_size` is not specified, then most batches will have `m` samples per class, but it's not guaranteed for every batch.

```python
samplers.MPerClassSampler(
    labels, m, batch_size=None, length_before_new_iter=100000, *, generator=None
)
```

**Parameters**:

- **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
- **m**: The number of samples per class to fetch at every iteration. If a class has less than m samples, then there will be duplicates in the returned batch.
- **batch_size**: Optional. If specified, then every batch is guaranteed to have `m` samples per class. There are a few restrictions on this value:
  - `batch_size` must be a multiple of `m`
  - `length_before_new_iter >= batch_size` must be true
  - `m * (number of unique labels) >= batch_size` must be true
- **length_before_new_iter**: The number of indices one iterator yields, rounded down to a multiple of `batch_size` (or of `m * number of classes` when that is smaller and `batch_size` is `None`). `len(sampler)` equals the number of indices yielded.
- **generator**: Optional `torch.Generator`; see [randomness and coverage](#randomness-and-coverage).

To keep the per-batch guarantee, give the `DataLoader` the same `batch_size`
and do not set `shuffle`:

```python
sampler = samplers.MPerClassSampler(labels, m=4, batch_size=32)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)
```

## HierarchicalSampler

Implementation of the sampler used in [Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf).

It will do the following per batch:

- Randomly select X super classes.
- For each super class, randomly select Y samples from Z classes, such that Y * Z equals the batch size divided by X.

(X, Y, and the batch size are controllable parameters. See below for details.)

It yields lists of dataset indices, so pass it as the `batch_sampler` argument:

```python
sampler = samplers.HierarchicalSampler(labels, batch_size=32, samples_per_class=4)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
```

```python
samplers.HierarchicalSampler(
		labels,
        batch_size,
        samples_per_class,
        batches_per_super_tuple=4,
        super_classes_per_batch=2,
        inner_label=0,
        outer_label=1,
        *,
        generator=None,
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
- **generator**: Optional `torch.Generator`; see [randomness and coverage](#randomness-and-coverage).

`len(sampler)` is `batches_per_super_tuple` times the number of combinations of
`super_classes_per_batch` super classes.

## Mining-weighted sampling

Use the numerical tuple-frequency utility with PyTorch's `WeightedRandomSampler`.
The [agent-facing recipe](recipes/training.md#mining-weighted-sampling-contract)
shows embedding extraction, subset-index mapping, and seeded sampling.

## FixedSetOfTriplets

When initialized, this class creates a fixed set of triplets. This is useful for determining the performance of algorithms in cases where the only ground truth data is a set of triplets.

```python
samplers.FixedSetOfTriplets(labels, num_triplets, *, generator=None)
```

The triplets are drawn once, when the sampler is created; each iterator yields
them in a new random order as consecutive `(anchor, positive, negative)` index
triples. Use a `DataLoader` batch size divisible by 3 and
`miners.EmbeddingsAlreadyPackagedAsTriplets` to recover them.

**Parameters**:

- **labels**: The list of labels for your dataset, i.e. the labels[x] should be the label of the xth element in your dataset.
- **num_triplets**: The number of triplets to create.
- **generator**: Optional `torch.Generator` used for creation and ordering.
