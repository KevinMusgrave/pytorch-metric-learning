# Reducers

A reducer turns the loss terms a loss computes into one scalar. For example,
[ContrastiveLoss](losses/contrastive_loss.md#contrastiveloss) computes a loss
for every positive and every negative pair in a batch; its default reducer
averages the nonzero losses of each group and sums the results. Reducers sit
between the loss and the final value:

`Your Data --> Sampler --> Miner --> Loss --> Reducer --> Final loss value`

Pass a reducer to a loss through its `reducer` option:

```python
import torch
from pytorch_metric_learning import losses, reducers

embeddings = torch.randn(32, 16)
labels = torch.randint(0, 4, (32,))

loss_func = losses.ContrastiveLoss(reducer=reducers.ThresholdReducer(low=0.1))
loss = loss_func(embeddings, labels)
```

The loss computes a dictionary of loss terms, the reducer reduces each term to a
scalar and sums them, and the loss adds any regularization penalty afterwards.
The reducer object is used as given; the loss does not copy it. Every reducer
returns a tensor. To get the unreduced terms instead, call
[`loss_func.loss_terms(...)`](losses.md#unreduced-loss-terms).

## BaseReducer

All reducers extend this class.

```python
reducers.BaseReducer()
```

### Loss terms

A loss returns a `pytorch_metric_learning.types.LossDict`: a dictionary from term
names to `LossTerm` dictionaries with these keys:

- **losses**: A tensor of loss values.
- **indices**: Determines the kind of term:
  - `None`: the term is already reduced; `losses` holds one value.
  - A tensor: one loss per element, and the tensor holds each element's anchor
    index.
  - A tuple `(anchors, others)`: one loss per pair.
  - A tuple `(anchors, positives, negatives)`, such as `TripletIndices`: one loss
    per triplet.

  A tuple must hold two or three tensors; pass a single tensor, not a
  one-element tuple, for per-element losses. Index tensors must be `int64` or
  `int32`; other dtypes, including `bool` and `uint8`, raise `TypeError`.
  Indices must be in range for the tensors they address. Reducers do not check
  bounds, because that would synchronize the device; out-of-range or negative
  indices are a caller error with undefined results.

- **divisor** (optional): A normalizer for [DivisorReducer](#divisorreducer).

For every kind except `None`, `losses` and each index tensor are
one-dimensional with equal lengths. Anchor indices are rows of the `embeddings`
passed to the reducer: the query embeddings for a loss, and the regularized
tensor for a regularizer. ProxyAnchorLoss terms index its proxies instead.

### Reduction

`reducer(loss_dict, embeddings, labels)` reduces every term with
`reduce(term, embeddings, labels)` and returns their sum. `reduce`:

1. Validates the term. Wrong types raise `TypeError`; wrong shapes or lengths
   raise `ValueError`.
2. Raises `ValueError` if the term has a `divisor`. Only
   [DivisorReducer](#divisorreducer) applies divisors.
3. Returns an already-reduced term unchanged.
4. Returns a differentiable zero for an empty `losses` tensor.
5. Otherwise calls `element_reduction(losses, anchors, embeddings, labels)`,
   where `anchors` is the term's first index tensor.

An empty `loss_dict`, which losses return when a batch has no valid pairs or
triplets, reduces to a differentiable zero connected to `embeddings`.

Subclasses implement the one reduction hook:

```python
def element_reduction(self, losses, anchors, embeddings, labels):
    raise NotImplementedError
```

`labels` is `None` when the reducer belongs to a
[regularizer](regularizers.md).

## AvgNonZeroReducer

The average of the losses that are greater than 0. For losses `[0, 2, 0, 3]`, it
returns `2.5`. It is `ThresholdReducer(low=0)`.

```python
reducers.AvgNonZeroReducer()
```

## ClassWeightedReducer

Multiplies each loss by the weight of its anchor's class, then takes the mean:
`torch.mean(losses * weights[labels[anchors]])`.

```python
reducers.ClassWeightedReducer(weights)
```

**Parameters**:

- **weights**: A tensor where `weights[c]` is the weight of class `c`. It is
  stored as a buffer and moves with the reducer; it is cast to the losses'
  dtype when applied.

Labels must be class indices into `weights`. Without labels, as in a
regularizer, the reducer raises `ValueError`. If `weights` is on a different
device from the losses, it raises `ValueError`; move the reducer, or the loss
that owns it, with `.to(device)`.

## DivisorReducer

Divides the sum of each term by a divisor that the loss supplies, so a loss can
apply its paper's normalization while the caller can still choose another
reducer for terms without divisors.

```python
reducers.DivisorReducer()
```

A term without a divisor uses 1, so its reduction is the sum. A zero divisor
gives 0. Sums are accumulated in float32 for float16 and bfloat16 losses, and in
the losses' dtype otherwise. A Python or tensor divisor is converted directly to
that accumulation dtype, so a float64 divisor such as `1e-50` keeps its value.

These components emit divisor terms and use `DivisorReducer` by default:

| Component                                                            | Terms with a divisor                                                       |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [MarginLoss](losses/margin_loss.md#marginloss)                       | `margin_loss`; `beta_reg_loss` with `learn_beta=True` and per-class `beta` |
| [ProxyAnchorLoss](losses/proxy_anchor_loss.md#proxyanchorloss)       | `pos_loss`, `neg_loss`                                                     |
| [SparseCentersRegularizer](regularizers.md#sparsecentersregularizer) | `loss`                                                                     |

Any other reducer raises `ValueError` for these terms. To combine a divisor
term with a different reduction of other terms, route each divisor term to
`DivisorReducer` with [MultipleReducers](#multiplereducers).

## MeanReducer

The mean of the losses.

```python
reducers.MeanReducer()
```

## MultipleReducers

Applies a different reducer to each named term and sums the results.

```python
reducers.MultipleReducers(reducers, default_reducer=None)
```

**Parameters**:

- **reducers**: A mapping from term names to reducers. The names are the keys of
  the loss's terms, as returned by
  [`loss_func.loss_terms(...)`](losses.md#unreduced-loss-terms). Values that
  are not reducers raise `TypeError`.
- **default_reducer**: Used for terms not in `reducers`. If `None`,
  `MeanReducer()` is used.

`MultipleReducers` routes terms only when it is the reducer of a loss or
regularizer. Used as the inner reducer of `PerAnchorReducer` or as a value inside
another `MultipleReducers`, it raises `TypeError`.

**Example usage**:

[ContrastiveLoss](losses/contrastive_loss.md#contrastiveloss) has two terms,
`pos_loss` and `neg_loss`. Here a [ThresholdReducer](#thresholdreducer) is used
for `pos_loss` and a [MeanReducer](#meanreducer) for `neg_loss`:

```python
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import (
    MeanReducer,
    MultipleReducers,
    ThresholdReducer,
)

reducer = MultipleReducers(
    {"pos_loss": ThresholdReducer(low=0.1), "neg_loss": MeanReducer()}
)
loss_func = ContrastiveLoss(reducer=reducer)
```

## PerAnchorReducer

Aggregates losses by anchor, then passes one value per anchor to an inner
reducer. For example, [NTXentLoss](losses/ntxent_loss.md#ntxentloss) returns a
loss per positive pair; with `PerAnchorReducer`, each anchor's pair losses are
averaged first, so every anchor has equal weight regardless of its number of
positives.

```python
reducers.PerAnchorReducer(reducer=None, aggregation="mean")
```

**Parameters**:

- **reducer**: The reducer that receives the per-anchor losses. If `None`,
  [MeanReducer](#meanreducer) is used.
- **aggregation**: `"mean"` or `"sum"` of each anchor's losses. Other values
  raise `ValueError`.

Aggregation works the same for element, pair, and triplet terms, using the
term's anchor indices:

- Anchor indices are rows of the query `embeddings`. A reference set passed as
  `ref_emb` may have more rows than the query.
- Anchors without any loss are excluded before the inner reducer, so they do
  not dilute the mean.
- Duplicate entries for an anchor, such as a pair mined twice, each count.
- The inner reducer receives an element term whose indices are the anchors that
  have losses, so it can weight by the anchors' labels.

For losses `[1, 3, 5]` with anchors `[0, 0, 2]`, `aggregation="mean"` gives
per-anchor values `[2, 5]`, and the default inner reducer returns `3.5`.
Sums are accumulated in float32 for float16 and bfloat16 losses.

## SumReducer

The sum of the losses.

```python
reducers.SumReducer()
```

## ThresholdReducer

The mean of the losses that fall strictly within a range.

```python
reducers.ThresholdReducer(low=None, high=None)
```

At least one of `low` or `high` must be specified; otherwise construction raises
`ValueError`.

**Parameters**:

- **low**: Losses less than or equal to this value are ignored.
- **high**: Losses greater than or equal to this value are ignored.

**Examples**:

- `ThresholdReducer(low=6)` keeps `losses > 6`. For `[3, 7, 1, 13, 5]` it
  returns `(7+13)/2 = 10`.
- `ThresholdReducer(high=6)` keeps `losses < 6`. For `[3, 7, 1, 13, 5]` it
  returns `(1+3+5)/3 = 3`.
- `ThresholdReducer(low=6, high=12)` keeps `6 < losses < 12`. For
  `[3, 7, 1, 13, 5]` it returns `7`.

When no loss falls in the range, the result is a differentiable zero. The
reduction uses a mask rather than boolean indexing, so it performs no host
synchronization and keeps a fixed shape for `torch.compile`.
