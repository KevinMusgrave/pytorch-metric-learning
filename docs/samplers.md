# Samplers

Samplers decide which dataset indices form each batch. `MPerClassSampler` and
`FixedSetOfTriplets` yield one dataset index at a time; pass them to
`DataLoader(..., sampler=...)`. `HierarchicalSampler` yields whole batches; pass
it as `DataLoader(..., batch_sampler=...)` and omit `batch_size` and `shuffle`.

Labels describe the whole dataset: `labels[i]` is the label of dataset item `i`.
`MPerClassSampler` and `FixedSetOfTriplets` accept a one-dimensional tensor, a
NumPy array, or any iterable of hashable labels, such as a list of strings.

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

Yields `m` indices per class in consecutive groups. For example, with a
`DataLoader` batch size of 100 and `m=5`, each batch holds 20 classes with 5
samples each. If `batch_size` is not given, most batches have `m` samples per
class, but a batch can span the boundary between two passes over the classes.

```python
samplers.MPerClassSampler(
    labels: Tensor | np.ndarray | Iterable[Hashable],
    m: int,
    batch_size: int | None = None,
    length_before_new_iter: int = 100000,
    *,
    generator: torch.Generator | None = None,
)
```

**Parameters**:

- **labels**: The dataset labels.
- **m**: The number of samples per class in each group. A class with fewer than
  `m` samples is sampled with replacement, so the batch contains duplicates.
- **batch_size**: Optional. If given, every batch of this size has `m` samples
  per class. Construction raises `ValueError` unless:
  - `batch_size` is a multiple of `m`
  - `length_before_new_iter >= batch_size`
  - `m * (number of unique labels) >= batch_size`
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

Each batch is built as follows:

- Select X super classes.
- For each super class, select Y samples from each of Z classes, such that
  Y * Z equals the batch size divided by X.

X is `super_classes_per_batch` and Y is `samples_per_class`. The sampler yields
lists of dataset indices, so pass it as the `batch_sampler` argument:

```python
sampler = samplers.HierarchicalSampler(labels, batch_size=32, samples_per_class=4)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
```

```python
samplers.HierarchicalSampler(
    labels: Tensor | np.ndarray | Sequence[Sequence[int]],
    batch_size: int,
    samples_per_class: int | Literal["all"],
    batches_per_super_tuple: int = 4,
    super_classes_per_batch: int = 2,
    inner_label: int = 0,
    outer_label: int = 1,
    *,
    generator: torch.Generator | None = None,
)
```

**Parameters**:

- **labels**: A two-dimensional array of integer labels with one row per dataset
  item and one column per hierarchy level.
- **batch_size**: The number of indices per batch. It must be positive and a
  multiple of `super_classes_per_batch`, and `batch_size // super_classes_per_batch`
  must be a multiple of `samples_per_class`; otherwise construction raises
  `ValueError`.
- **samples_per_class**: The number of samples per class in a batch, Y in the
  explanation above, or `"all"` to take every sample of each chosen class. `"all"`
  suits datasets with few samples per class.
- **batches_per_super_tuple**: The number of batches created for each
  combination of super classes.
- **super_classes_per_batch**: The number of super classes per batch, X in the
  explanation above.
- **inner_label**: The column of `labels` holding classes.
- **outer_label**: The column of `labels` holding super classes.
- **generator**: Optional `torch.Generator`; see [randomness and coverage](#randomness-and-coverage).

`len(sampler)` is `batches_per_super_tuple` times the number of combinations of
`super_classes_per_batch` super classes. Each super class fills
`batch_size // super_classes_per_batch` positions with whole classes in random
order, skipping a class that would overflow its share. A batch is shorter than
`batch_size` when a super class lacks enough classes to fill its share.

## Mining-weighted sampling

To sample examples in proportion to how often a miner selects them, pass
`utils.indexing.convert_to_weights` output to PyTorch's `WeightedRandomSampler`.
The [executable recipe](recipes/training.md#mining-weighted-sampling-contract)
shows embedding extraction, subset-index mapping, and seeded sampling.

## FixedSetOfTriplets

Creates a fixed set of triplets at construction. Use it when the only ground
truth for evaluating an algorithm is a set of triplets.

```python
samplers.FixedSetOfTriplets(
    labels: Tensor | np.ndarray | Iterable[Hashable],
    num_triplets: int,
    *,
    generator: torch.Generator | None = None,
)
```

Each triplet draws two different classes, then an anchor and a positive from the
first class and a negative from the second. A class with one sample yields an
anchor equal to its positive. Construction raises `ValueError` when
`num_triplets` is not positive or `labels` has fewer than two classes.

The triplets are drawn once, when the sampler is created; each iterator yields
them in a new random order as consecutive `(anchor, positive, negative)` index
triples. Use a `DataLoader` batch size divisible by 3 and recover the tuple
indices directly from each embedding batch:

```python
indices_tuple = torch.arange(len(embeddings), device=embeddings.device).reshape(-1, 3).unbind(1)
loss = loss_func(embeddings, labels, indices_tuple=indices_tuple)
```

**Parameters**:

- **labels**: The dataset labels.
- **num_triplets**: The number of triplets to create.
- **generator**: Optional `torch.Generator` used for creation and ordering.
