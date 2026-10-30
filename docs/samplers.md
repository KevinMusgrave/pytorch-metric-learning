# Samplers

Samplers decide which dataset indices form each batch. `MPerClassSampler` and
`HierarchicalSampler` yield whole batches; pass them as
`DataLoader(..., batch_sampler=...)` and omit `batch_size` and `shuffle`.
`FixedSetOfTriplets` yields one dataset index at a time; pass it to
`DataLoader(..., sampler=...)`.

Labels describe the whole dataset: `labels[i]` is the label of dataset item `i`.
`MPerClassSampler` and `FixedSetOfTriplets` accept a one-dimensional tensor, a
NumPy array, or any iterable of hashable labels, such as a list of strings
([`types.DatasetLabels`](types.md#datasetlabels)).
The examples on this page use this dataset of 64 items in 16 classes:

```python
import torch
from pytorch_metric_learning import samplers

labels = torch.arange(64) // 4
dataset = torch.utils.data.TensorDataset(torch.randn(64, 8), labels)
```

## Randomness and coverage

Every sampler takes a keyword-only `generator: torch.Generator | None`. With
`None`, sampling uses PyTorch's global random number generator, so
`torch.manual_seed` makes it repeatable; a dedicated generator isolates sampler
randomness from model initialization and dropout. Each new iterator draws fresh
random batches, and two samplers given generators with the same seed produce the
same sequence of epochs. Samplers run in the main process, so `DataLoader`
worker seeding does not affect them.

Samplers balance labels within batches, and balance has priority over visiting
every example equally often: when class sizes differ, equal class counts
per batch must repeat small classes. `MPerClassSampler` documents its
[coverage guarantees](#coverage-guarantees). `HierarchicalSampler` draws classes
and examples independently for each batch, with replacement across batches, so
its iterators are not epochs. A class with fewer members than requested is
sampled with replacement within its batch.

## MPerClassSampler

Yields batches of `batch_size // m` different classes with `m` indices per
class, grouped by class. For example, `batch_size=100` and `m=5` gives 20
classes with 5 samples each in every batch.

```python
sampler = samplers.MPerClassSampler(labels, m=4, batch_size=32)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
assert len(loader) == len(sampler) == 2
```

```python {.signature}
samplers.MPerClassSampler(
    labels: Tensor | np.ndarray | Iterable[Hashable],
    m: int,
    batch_size: int,
    num_batches: int | None = None,
    *,
    generator: torch.Generator | None = None,
)
```

**Parameters**:

- **labels**: The dataset labels.
- **m**: The number of samples per class in each batch.
- **batch_size**: The number of indices per batch. Construction raises
  `ValueError` unless `batch_size` is a multiple of `m` and
  `m * (number of unique labels) >= batch_size`.
- **num_batches**: The number of batches one iterator yields, which is
  `len(sampler)`. It must be positive. With `None`, the sampler uses the smallest
  length with the default coverage guarantee below:
  `ceil(ceil(largest_class_size / m) * num_classes / (batch_size // m))`.
- **generator**: Optional `torch.Generator`; see [randomness and coverage](#randomness-and-coverage).

### Coverage guarantees

Classes rotate through shuffled passes, and so do the examples of each class. In
every iterator, at every batch:

- Every batch has exactly `batch_size // m` different classes and `m` indices
  for each.
- A class appears at most once more than any other class.
- An example appears at most once more than any other example of its class.
- A batch repeats an example only when its class has fewer than `m` examples.

With the default `num_batches`, every example appears at least once per
iterator. Each example appears exactly once when all classes have the same size,
`m` divides that size, and `batch_size // m` divides the number of classes.

With unequal class sizes, exactly-once coverage is impossible: every class
appears equally often, so the default length is set by the largest class, and
smaller classes repeat. On long-tailed data that length can be large. Pass a
smaller `num_batches` to shorten an iterator; the guarantees that hold at every
batch still apply, but examples of large classes may be missed.

## HierarchicalSampler

Implementation of the sampler used in [Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf).

Each batch is built as follows:

- Select X super classes.
- For each super class, select Y samples from each of Z classes, such that
  Y * Z equals the batch size divided by X.

X is `super_classes_per_batch` and Y is `samples_per_class`. The sampler yields
lists of dataset indices, so pass it as the `batch_sampler` argument:

```python
hierarchical_labels = torch.stack([labels, labels // 4], dim=1)
sampler = samplers.HierarchicalSampler(
    hierarchical_labels, batch_size=32, samples_per_class=4
)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
assert all(len(batch) == 32 for batch in sampler)
```

```python {.signature}
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

```python {.signature}
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
from pytorch_metric_learning import losses

sampler = samplers.FixedSetOfTriplets(labels, num_triplets=30)
loader = torch.utils.data.DataLoader(dataset, batch_size=9, sampler=sampler)
model = torch.nn.Linear(8, 4)
loss_func = losses.TripletMarginLoss()

data, batch_labels = next(iter(loader))
embeddings = model(data)
indices_tuple = (
    torch.arange(len(embeddings), device=embeddings.device).reshape(-1, 3).unbind(1)
)
loss = loss_func(embeddings, batch_labels, indices_tuple=indices_tuple)
```

**Parameters**:

- **labels**: The dataset labels.
- **num_triplets**: The number of triplets to create.
- **generator**: Optional `torch.Generator` used for creation and ordering.
