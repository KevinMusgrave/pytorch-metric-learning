# Types

`pytorch_metric_learning.types` declares the index tuples, loss-term
dictionaries, option dictionaries, and type aliases used in public signatures.
They describe ordinary tuples, dictionaries, tensors, and strings: there is no
runtime wrapper, and a value of the right structure is accepted wherever the
type appears. Import them for annotations:

```python
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.types import IndicesTuple, LossDict

embeddings = torch.randn(16, 8)
labels = torch.randint(0, 4, (16,))
indices: IndicesTuple = miners.BatchHardMiner()(embeddings, labels)
terms: LossDict = losses.TripletMarginLoss().loss_terms(embeddings, labels, indices)
```

The definitions on this page are copied from the source, and the test suite
checks that they match it.

## Index tuples

### PairIndices

```python {.signature}
class PairIndices(NamedTuple):
    pos_anchors: Tensor
    positives: Tensor
    neg_anchors: Tensor
    negatives: Tensor
```

Positive pairs `(pos_anchors[i], positives[i])` and negative pairs
`(neg_anchors[j], negatives[j])`. Each field is a one-dimensional `int64` or
`int32` tensor; `pos_anchors` and `positives` have equal lengths, and so do
`neg_anchors` and `negatives`. Anchors index the query embeddings; positives
and negatives index the [reference set](concepts.md#reference-sets). Pair
miners and [`utils.indexing`](indexing.md) functions return this type.

### TripletIndices

```python {.signature}
class TripletIndices(NamedTuple):
    anchors: Tensor
    positives: Tensor
    negatives: Tensor
```

Triplets `(anchors[i], positives[i], negatives[i])`, three one-dimensional
`int64` or `int32` tensors of equal length, indexed like `PairIndices`.

### IndicesTuple

```python {.signature}
type IndicesTuple = PairIndices | TripletIndices
```

The `indices_tuple` argument of losses and miners after validation. Callers
may also pass plain tuples or lists of four or three tensors, which are
validated and converted to these named tuples; see
[index tuples](losses.md#index-tuples) for the accepted forms and the errors
raised for other inputs. `None` means unrestricted, and a tuple of empty tensors
selects nothing.

### TripletsPerAnchor

```python {.signature}
type TripletsPerAnchor = int | Literal["all"]
```

The `t_per_anchor` argument of
[`convert_to_triplets`](indexing.md#convert_to_triplets): a positive number of
sampled triplets per anchor, or `"all"` to enumerate every triplet.

## Loss terms

### LossTerm

```python {.signature}
class LossTerm(TypedDict):
    losses: Tensor
    indices: Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | None
    divisor: NotRequired[Tensor | float | int]
```

One term of a loss, before reduction. Losses, regularizers, and reducers share
this contract.

`indices` determines what one value of `losses` is and how reducers group the
values:

| `indices`                                                        | `losses`                                   | Reducers group by |
| ---------------------------------------------------------------- | ------------------------------------------ | ----------------- |
| `None`                                                           | One value; the term is already reduced     | Not reduced       |
| A tensor of shape `(n,)`                                         | `(n,)`, one value per element              | That tensor       |
| `(anchors, others)`, two tensors of shape `(n,)`                 | `(n,)`, one value per (anchor, other) pair | `anchors`         |
| `TripletIndices(anchors, positives, negatives)` or three tensors | `(n,)`, one value per triplet              | `anchors`         |

- An element can be an embedding, a query, or a proxy; the indices only need
  to be meaningful to the reducer in use. Anchor indices are rows of the
  `embeddings` passed to the reducer: the query embeddings for a loss, and the
  regularized tensor for a regularizer. ProxyAnchorLoss terms index its proxies
  instead. Reducers that look up labels, such as
  [ClassWeightedReducer](reducers.md#classweightedreducer), index `labels` with
  the anchors.
- A term without indices must contain exactly one value. A term with indices
  must have one-dimensional `losses` and index tensors of equal length;
  otherwise reducers raise `ValueError`.
- A tuple of indices holds two or three tensors. Use a single tensor, not a
  one-element tuple, for per-element losses.
- Index tensors must be `int64` or `int32`. Other dtypes, including `bool` and
  `uint8`, and other structures raise `TypeError`. Indices are not
  bounds-checked, because that would synchronize the device; out-of-range or
  negative indices are a caller error with undefined results.
- An empty `losses` tensor contributes a differentiable zero; see
  [reduction](reducers.md#reduction).
- The optional `divisor`, a tensor or number, is applied only by
  [DivisorReducer](reducers.md#divisorreducer), which divides the sum of the
  term's losses by it. Every other reducer raises `ValueError` for a term with
  a divisor; route such terms with
  [MultipleReducers](reducers.md#multiplereducers).

### LossDict

```python {.signature}
type LossDict = dict[str, LossTerm]
```

What `compute_loss` returns and `loss_terms` exposes: term names mapped to
[`LossTerm`](#lossterm) values. An empty dictionary means nothing was valid to
compute and reduces to a differentiable zero.

## Options

### LossOptions

```python {.signature}
class LossOptions(TypedDict, total=False):
    distance: BaseDistance | None
    reducer: BaseReducer | None
    embedding_regularizer: BaseRegularizer | None
    embedding_reg_weight: float
```

The keyword-only `**options` of losses; see
[loss options](losses.md#loss-options) for defaults and errors. Annotate a
custom loss's `**options: Unpack[LossOptions]` and forward them to the base
class, as in [custom losses](extend/losses.md#compatibility-with-distances-and-reducers).

### ProxyLossOptions

```python {.signature}
class ProxyLossOptions(LossOptions, total=False):
    weight_regularizer: BaseRegularizer | None
    weight_reg_weight: float
    weight_init_func: Callable[[Tensor], object] | None
```

`LossOptions` plus the
[learnable weight arguments](losses.md#learnable-weight-arguments) of
[BaseProxyLoss](losses/base_proxy_loss.md) subclasses.

### RegularizerOptions

```python {.signature}
class RegularizerOptions(TypedDict, total=False):
    distance: BaseDistance | None
    reducer: BaseReducer | None
```

The keyword-only `**options` of [regularizers](regularizers.md#baseregularizer).

## Component types

### MetricLoss

```python {.signature}
type MetricLoss = BaseMetricLossFunction | MultipleLosses
```

The losses that
[CrossBatchMemory](losses/cross_batch_memory.md),
[SelfSupervisedLoss](losses/self_supervised_loss.md),
[MultipleLosses](losses/multiple_losses.md), and
[DistributedLossWrapper](distributed.md#distributedlosswrapper) accept. Both
share the forward signature
`(embeddings, labels, indices_tuple, ref_emb, ref_labels)`. Another module
raises `TypeError`, except that DistributedLossWrapper also accepts a
CrossBatchMemory.

### MiningStrategy

```python {.signature}
type MiningStrategy = Literal["hard", "semihard", "easy", "all"]
```

The `pos_strategy` and `neg_strategy` of
[BatchEasyHardMiner](miners.md#batcheasyhardminer).

### PairFunction

```python {.signature}
type PairFunction = Callable[[Tensor, Tensor], PairIndices]
```

The `pair_function` of [CrossBatchMemory](losses/cross_batch_memory.md): it
takes labels and reference labels and returns all candidate pairs.

### PNPVariant

```python {.signature}
type PNPVariant = Literal["Ds", "Dq", "Iu", "Ib", "O"]
```

The `variant` of [PNPLoss](losses/pnp_loss.md). Another value raises
`ValueError`.

### AnchorAggregation

```python {.signature}
type AnchorAggregation = Literal["mean", "sum"]
```

The `aggregation` of [PerAnchorReducer](reducers.md#peranchorreducer). Another
value raises `ValueError`.

### DistributedObjective

```python {.signature}
type DistributedObjective = Literal["global", "local_anchor"]
```

The `objective` of the [distributed wrappers](distributed.md#objectives).
Another value raises `ValueError`.

### SubCenterOutliers

```python {.signature}
class SubCenterOutliers(NamedTuple):
    outliers: Tensor
    dominant_centers: Tensor
```

The result of
[`SubCenterArcFaceLoss.get_outliers`](losses/subcenter_arcface_loss.md):
`outliers` is a one-dimensional `int64` tensor of embedding indices, and
`dominant_centers` has shape `(num_classes, embedding_size)`.

## Evaluation types

### ArrayLike

```python {.signature}
type ArrayLike = Tensor | np.ndarray
```

Embeddings and labels accepted by
[AccuracyCalculator](accuracy_calculation.md#getting-accuracy).

### NeighborLimit

```python {.signature}
type NeighborLimit = int | Literal["max_bin_count"] | None
```

The `k` of [AccuracyCalculator](accuracy_calculation.md#parameters).

### MetricNeighbors

```python {.signature}
type MetricNeighbors = int | Literal["relevant", "all"]
```

The `neighbors` of a
[RetrievalMetric](accuracy_calculation.md#adding-custom-accuracy-metrics).

### KNNFunction

```python {.signature}
type KNNFunction = Callable[[Tensor, int, Tensor, bool], tuple[Tensor, Tensor]]
```

A `knn_func(query, k, reference, ref_includes_query)` returning
`(distances, indices)`; see the
[k-NN function contract](retrieval.md#k-nn-function-contract).

### KMeansFunction

```python {.signature}
type KMeansFunction = Callable[[Tensor, int], Tensor]
```

A `kmeans_func(embeddings, num_clusters)` returning one cluster index per row,
such as [FaissKMeans](retrieval.md#faisskmeans).

### LabelComparison

```python {.signature}
type LabelComparison = Callable[[Tensor, Tensor], Tensor]
```

A `label_comparison_fn(x, y)` returning a boolean tensor of the broadcast shape;
see [custom label comparison](accuracy_calculation.md#using-a-custom-label-comparison-function).

## Sampler types

### DatasetLabels

```python {.signature}
type DatasetLabels = Tensor | np.ndarray | Iterable[Hashable]
```

The dataset labels accepted by [MPerClassSampler](samplers.md#mperclasssampler)
and [FixedSetOfTriplets](samplers.md#fixedsetoftriplets): a one-dimensional
tensor or array, or any iterable of hashable labels.
