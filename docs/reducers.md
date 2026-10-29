# Reducers

A reducer turns the many per-pair, per-triplet, or per-element values a loss
computes into one scalar. For example, [ContrastiveLoss](losses/contrastive_loss.md#contrastiveloss)
computes a loss for every positive and negative pair in a batch; its reducer
averages those values. Reducers sit between the loss and the final value:

`Your Data --> Sampler --> Miner --> Loss --> Reducer --> Final loss value`

Pass a reducer to a loss through its `reducer` option:

```python
from pytorch_metric_learning import losses, reducers
reducer = reducers.SomeReducer()
loss_func = losses.SomeLoss(reducer=reducer)
loss = loss_func(embeddings, labels)
```

The loss computes a dictionary of loss terms, the reducer reduces each term to
a scalar and sums them, and the loss adds any regularization penalty afterwards.
The reducer object is used as given; the loss does not copy it. Every reducer
returns a tensor. To get the unreduced terms instead, call
[`loss_func.loss_terms(...)`](losses.md#unreduced-loss-terms).

## AvgNonZeroReducer

The average of the losses that are greater than 0. For losses `[0, 2, 0, 3]`, it
returns `2.5`. It is `ThresholdReducer(low=0)`.

```python
reducers.AvgNonZeroReducer()
```

## BaseReducer

All reducers extend this class.

```python
reducers.BaseReducer()
```

`forward(loss_dict, embeddings, labels)` reduces every term in the dictionary
with `reduce(term, embeddings, labels)` and returns their sum. `reduce`
validates the term, returns `already_reduced` values unchanged, turns an empty
`losses` tensor into a differentiable zero, and otherwise dispatches on the
reduction type. Subclasses implement:

```python
def element_reduction(self, losses, loss_indices, embeddings, labels):
    raise NotImplementedError
```

`losses` has one value per element and `loss_indices` holds the element
indices. For `"pos_pair"`, `"neg_pair"`, and `"triplet"` terms, the default
`pos_pair_reduction`, `neg_pair_reduction`, and `triplet_reduction` call
`element_reduction` with the anchor indices, `loss_indices[0]`. Override them
when tuple terms need different treatment, as
[PerAnchorReducer](#peranchorreducer) does. Invalid terms raise `TypeError` for
wrong types and `ValueError` for wrong shapes or lengths.

## ClassWeightedReducer

Multiplies each loss by the weight of its anchor's class, then takes the mean.

```python
reducers.ClassWeightedReducer(weights)
```

**Parameters**:

- **weights**: A tensor where `weights[i]` is the weight of class `i`. It is
  stored as a buffer and moves with the reducer.

## DivisorReducer

Divides the sum of each term by a divisor the loss supplies, so a loss can hard
code its paper's normalization while still letting the caller choose another
reducer.

```python
reducers.DivisorReducer()
```

The loss includes `divisor` in its term. For example, [ProxyAnchorLoss](losses/proxy_anchor_loss.md#proxyanchorloss)
uses `DivisorReducer` by default and returns:

```python
loss_dict = {
    "pos_loss": {
        "losses": pos_term,
        "indices": loss_indices,
        "reduction_type": "element",
        "divisor": num_proxies_with_positives,
    },
    "neg_loss": {
        "losses": neg_term,
        "indices": loss_indices,
        "reduction_type": "element",
        "divisor": self.num_classes,
    },
}
```

A missing divisor is 1, and a zero divisor gives 0. Sums are accumulated in
float32 for float16 and bfloat16 losses.

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

- **reducers**: A mapping from term names to reducers. The names must match the
  loss's `sub_loss_names`.
- **default_reducer**: Used for terms not in `reducers`. If `None`, then
  `MeanReducer()` is used.

**Example usage**:

[ContrastiveLoss](losses/contrastive_loss.md#contrastiveloss) has two terms,
`pos_loss` and `neg_loss`. Here a [ThresholdReducer](#thresholdreducer) is
used for `pos_loss` and a [MeanReducer](#meanreducer) for `neg_loss`:

```python
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer
reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
reducer = MultipleReducers(reducer_dict)
loss_func = ContrastiveLoss(reducer=reducer)
```

## PerAnchorReducer

Converts unreduced pairs to unreduced elements. For example, [NTXentLoss](losses/ntxent_loss.md#ntxentloss)
returns a loss per positive pair; with `PerAnchorReducer`, the pair losses are
first aggregated per anchor, then passed to the inner reducer.

```python
def mean_per_anchor(x, num_per_row):
    return torch.where(num_per_row == 0, 0, x.sum(dim=1) / num_per_row.clamp_min(1))

reducers.PerAnchorReducer(reducer=None, aggregation_func=mean_per_anchor)
```

**Parameters**:

- **reducer**: The reducer that receives the per-element losses. The default
  is [MeanReducer](#meanreducer).
- **aggregation_func**: A function of `(x, num_per_row)` returning one value
  per row of `x`. `x` is an `(N, N)` matrix of pair losses, where `N` is the
  batch size, and `num_per_row` counts the pairs in each row. The default is the
  mean per row shown above.

Triplet terms are not supported and raise `NotImplementedError`.

## SumReducer

The sum of the losses.

```python
reducers.SumReducer()
```

## ThresholdReducer

The mean of the losses that fall within a range.

```python
reducers.ThresholdReducer(low=None, high=None)
```

At least one of `low` or `high` must be specified.

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
