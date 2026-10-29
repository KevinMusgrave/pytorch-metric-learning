# Concepts

This page explains how the library's components fit together. Each section
links to the page that documents the component in full.

```text
embeddings, labels --> miner --> indices --> loss --> loss terms --> reducer --> scalar
                                                        regularizers --^
```

## Embeddings and labels

Your model produces `embeddings`, a floating-point tensor of shape
`(batch_size, embedding_size)`. Each row is one example. `labels` is a
one-dimensional tensor with one class label per row. Two rows with the same label
are similar, and two rows with different labels are dissimilar.

```python
import torch

torch.manual_seed(0)
embeddings = torch.randn(8, 4, requires_grad=True)
labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
```

## Pairs and triplets

A positive pair is two rows with the same label, and a negative pair is two rows
with different labels. A triplet is an anchor, a positive for that anchor, and a
negative for that anchor. Pairs and triplets are stored as row indices, in the
NamedTuples from `pytorch_metric_learning.types`:

- `PairIndices(pos_anchors, positives, neg_anchors, negatives)`
- `TripletIndices(anchors, positives, negatives)`

Each field is a one-dimensional `torch.long` tensor, and position `i` of every
field describes the same pair or triplet. Losses and miners also accept plain
tuples of three or four index tensors of dtype `int64` or `int32`; other dtypes
raise `TypeError`. Indices are not bounds-checked, so out-of-range values are a
caller error.

```python
from pytorch_metric_learning.types import PairIndices
from pytorch_metric_learning.utils import indexing

pairs = indexing.get_all_pairs_indices(labels)
assert isinstance(pairs, PairIndices)
assert torch.equal(labels[pairs.pos_anchors], labels[pairs.positives])
assert not torch.any(pairs.pos_anchors == pairs.positives)
```

## Reference sets

Every loss and miner compares query rows, the anchors, with reference rows, the
positives and negatives:

- With `ref_emb=None`, the reference is `embeddings` itself. A row is never its
  own positive.
- With an explicit `ref_emb` and `ref_labels`, the reference is a separate set.
  Anchor indices refer to `embeddings`, and positive and negative indices refer
  to `ref_emb`. No pair is treated as a self-comparison, even when `ref_emb` is
  the same tensor as `embeddings`.

With `ref_emb`, pass both `labels` and `ref_labels` or neither. Passing
`ref_labels` without `ref_emb` raises `ValueError`. The
[losses overview](losses.md#supported-arguments) lists which losses accept a
reference set.

## Distances

A [distance](distances.md) computes a `(len(query), len(reference))` matrix.
Its `is_inverted` attribute gives the orientation: `False` for distances, where
smaller is closer, and `True` for similarities, where larger is closer. Losses
and miners read it, so the same loss works with either kind.

```python
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance

assert LpDistance()(embeddings).shape == (8, 8)
assert CosineSimilarity().is_inverted and not LpDistance().is_inverted
```

## Miners

A [miner](miners.md) selects informative pairs or triplets and returns their
indices. Pass the result to a loss as `indices_tuple`; without it, the loss uses
every pair or triplet that the labels define. A miner result with no indices
selects nothing, and the loss is a differentiable zero. A miner can also take
candidate indices and select only from them, so miners compose in stages.

```python
from pytorch_metric_learning import miners

pairs = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)(embeddings, labels)
```

## Losses and loss terms

A [loss](losses.md) takes `embeddings`, `labels`, an optional `indices_tuple`,
and an optional reference set, and returns a scalar tensor. Internally, it first
computes loss terms: a dictionary from term names to entries with these keys:

- `losses`: a tensor of loss values.
- `indices`: what the values belong to. `None` for an already-reduced value, a
  tensor of element indices, a pair `(anchors, others)`, or triplet indices.
- `divisor` (optional): a normalizer that only
  [DivisorReducer](reducers.md#divisorreducer) applies.

A batch with no valid pair or triplet produces no terms or terms with empty
`losses`.
`loss_func.loss_terms(...)` returns the terms without reducing them.

```python
from pytorch_metric_learning import losses

loss_func = losses.ContrastiveLoss()
terms = loss_func.loss_terms(embeddings, labels, pairs)
assert set(terms) == {"pos_loss", "neg_loss"}
loss = loss_func(embeddings, labels, pairs)
loss.backward()
```

## Reducers

A [reducer](reducers.md) turns loss terms into one scalar: it reduces each term,
for example with a mean, and sums the results. No terms, or terms
with empty `losses`, reduce to a differentiable zero. Each loss has a default reducer, and the `reducer` option
replaces it.

## Regularizers

A [regularizer](regularizers.md) penalizes the embeddings, or a loss's learnable
class weights, without labels. A loss multiplies the regularizer's value by its
weight and adds it after reducing its own terms.

```python
from pytorch_metric_learning import regularizers

loss_func = losses.ContrastiveLoss(
    embedding_regularizer=regularizers.LpRegularizer(), embedding_reg_weight=0.1
)
```

## Reference providers

Two wrappers supply a larger reference set to a loss, which may be a
`BaseMetricLossFunction` or a [MultipleLosses](losses/multiple_losses.md)
combination (`types.MetricLoss`):

- [CrossBatchMemory](losses/cross_batch_memory.md#crossbatchmemory) keeps a queue
  of detached embeddings from earlier batches and passes it to the wrapped loss
  as the reference set.
- [DistributedLossWrapper and DistributedMinerWrapper](distributed.md) gather
  embeddings from every rank, so each rank's anchors are compared with the
  global batch.

```python
memory = losses.CrossBatchMemory(
    losses.ContrastiveLoss(), embedding_size=4, memory_size=32
)
loss = memory(embeddings, labels)
assert memory.queue_idx == 8
```

## Evaluation

[AccuracyCalculator](accuracy_calculation.md) retrieves each query's nearest
references and computes retrieval metrics such as `precision_at_1`. The
[retrieval backends](retrieval.md) perform the nearest-neighbor search, and the
[evaluation recipe](recipes/evaluation.md) shows embedding extraction and
query/reference splits.

```python
from pytorch_metric_learning.evaluation import AccuracyCalculator

accuracies = AccuracyCalculator(include=("precision_at_1",)).get_accuracy(
    embeddings.detach(), labels
)
assert set(accuracies) == {"precision_at_1"}
```

## Devices and dtypes

- `labels`, `ref_emb`, `ref_labels`, and index tensors must be on the
  embeddings' device. Otherwise, components raise `ValueError` rather than copy
  them.
- Learnable parameters and buffers, such as class proxies or a memory queue,
  belong to their module. Place them with `module.to(device, dtype)`; a forward
  call never moves or casts them.
- Outside autocast, a module's parameters must have the embeddings' device and
  dtype, or the call raises `ValueError`. Under autocast, parameters may keep a
  higher precision than the embeddings.
