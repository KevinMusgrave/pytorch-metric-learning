# Miners

Miners select which pairs or triplets in a batch contribute to a loss. They
operate on the embeddings of the current batch; offline selection of which
examples form a batch belongs in a [sampler](samplers.md). Without a miner, a
loss uses every pair or triplet that the labels define.

Every miner returns one of the NamedTuples in `pytorch_metric_learning.types`:

- Pair miners return `PairIndices(pos_anchors, positives, neg_anchors, negatives)`.
- Triplet miners return `TripletIndices(anchors, positives, negatives)`.

Each field is a one-dimensional `torch.long` tensor on the embeddings' device.
Losses accept these tuples, or plain tuples of three or four index tensors, as
their `indices_tuple` argument:

```python
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.types import PairIndices

embeddings = torch.randn(32, 16, requires_grad=True)
labels = torch.randint(0, 4, (32,))

miner = miners.MultiSimilarityMiner(epsilon=0.1)
loss_func = losses.MultiSimilarityLoss()
pairs = miner(embeddings, labels)
loss = loss_func(embeddings, labels, pairs)
loss.backward()

assert isinstance(pairs, PairIndices)
assert torch.equal(labels[pairs.pos_anchors], labels[pairs.positives])
```

## BaseMiner

All miners extend this class and accept its keyword-only `distance` and
`tuples_per_anchor` options.

```python {.signature}
miners.BaseMiner(*, distance=None, tuples_per_anchor=None)
```

**Parameters**:

- **distance**: A [distance](distances.md) object. If `None`, the miner's
  default distance is used.
- **tuples_per_anchor**: The most tuples an anchor keeps. If `None`, every mined
  tuple is returned. See [bounding the output](#bounding-the-output).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

### Calling a miner

```python {.signature}
miner(
    embeddings,
    labels,
    indices_tuple=None,
    ref_emb=None,
    ref_labels=None,
    *,
    mat=None,
)
```

- **embeddings**: `(N, D)` query embeddings. Anchor indices refer to these rows.
- **labels**: `(N,)` labels. Miners require labels.
- **indices_tuple**: Optional candidate pairs or triplets. When given, the miner
  selects only from these candidates; see [candidates](#candidates).
- **ref_emb**, **ref_labels**: Optional reference set of shape `(M, D)` and
  `(M,)`, following the [reference-set rule](concepts.md#reference-sets).
  Positive and negative indices refer to these rows. Miners require labels, so
  `ref_labels` is required with `ref_emb`.
- **mat**: Optional precomputed `(N, N)`, or `(N, M)` with `ref_emb`, distance
  matrix used in place of the miner's distance call. Every miner supports it except
  `AngularMiner`, which reads embeddings directly and raises `ValueError`. A wrong
  shape, an integer dtype, or another device raises `ValueError`. Unlike a loss, a
  miner needs no graph, so `mat` may be detached. `DistributedMinerWrapper` gathers
  its own inputs and has no `mat`. See
  [precomputed distance matrices](losses.md#precomputed-distance-matrices).

Pass `indices_tuple`, `ref_emb`, and `ref_labels` by keyword. `labels`,
`ref_emb`, `ref_labels`, and candidate tensors must be on the embeddings'
device; otherwise the miner raises `ValueError`. Mining runs under
`torch.no_grad()`, so the returned indices carry no autograd graph.

### Bounding the output

Hard-pair counts can grow quadratically with the batch, and triplet counts faster.
`tuples_per_anchor` caps what a loss receives, which bounds the tuple-indexed values
that loss computes and backpropagates through.

```python
import torch
from pytorch_metric_learning import miners

embeddings = torch.randn(64, 16)
labels = torch.arange(64) // 8
miner = miners.TripletMarginMiner(margin=2, tuples_per_anchor=4)
anchors, _, _ = miner(embeddings, labels)
assert torch.bincount(anchors, minlength=64).max() <= 4
```

Selection happens after mining, so the miner's own rule decides which tuples are
eligible. Each anchor then keeps a uniformly random subset of its mined tuples,
without replacement, and an anchor with few enough tuples keeps them all. Kept
tuples stay in mined order. A triplet miner keeps at most `tuples_per_anchor`
triplets per anchor. A pair miner keeps at most that many positive pairs and that
many negative pairs per anchor. Values other than `None` must be positive; otherwise
construction raises `ValueError`. Sampling uses the global PyTorch random number
generator, so `torch.manual_seed` makes it repeatable.

A miner's own peak memory does not change, because it still evaluates every
candidate. Unlike `triplets_per_anchor` on `TripletMarginLoss` and `MarginLoss`,
which draws random triplets with replacement before any mining, this option applies
to tuples a miner has already selected and never repeats one.

### Candidates

Every built-in miner restricts its search to `indices_tuple` when it is given:

- Pair miners use pair candidates as given. Triplet candidates
  `(a, p, n)` become the pairs `(a, p)` and `(a, n)`, one of each per triplet,
  so a pair shared by several triplets is a candidate once per triplet.
- Triplet miners use triplet candidates as given. Pair candidates become every
  combination of a positive pair and a negative pair with the same anchor.

Candidates are per call; a miner keeps no candidate state between calls. Passing
one miner's output to another narrows the selection in stages:

```python
import torch
from pytorch_metric_learning import miners

embeddings = torch.randn(32, 16)
labels = torch.randint(0, 4, (32,))

hard_pairs = miners.MultiSimilarityMiner(epsilon=0.1)(embeddings, labels)
hardest = miners.HDCMiner(filter_percentage=0.25)(
    embeddings, labels, indices_tuple=hard_pairs
)
assert len(hardest.pos_anchors) <= len(hard_pairs.pos_anchors)
```

The [cascaded mining recipe](recipes/training.md#cascaded-mining-contract)
applies this to successive embedding slices.

### Output validation

`forward` validates the indices returned by `mine`: three triplet index tensors
or four pair index tensors, each one-dimensional, on the embeddings' device, with
matching anchor and partner lengths. A non-tuple output or non-tensor elements
raise `TypeError`; wrong dimensions, lengths, devices, or tuple sizes raise
`ValueError`. The checks also run when Python optimizations are enabled.
Candidate `indices_tuple` inputs are validated the same way.

See [custom miners](extend/miners.md) to write your own miner.

## AngularMiner

Returns triplets whose angle at the negative, as defined in
[Deep Metric Learning with Angular Loss](https://arxiv.org/abs/1708.01682), is
greater than `alpha`.

```python {.signature}
miners.AngularMiner(alpha=20, *, distance=None, tuples_per_anchor=None)
```

**Parameters**

- **alpha**: The angle threshold in degrees. It has the same meaning as the
  `alpha` of [AngularLoss](losses/angular_loss.md#angularloss).

Embeddings are L2-normalized before angles are computed, as in AngularLoss.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)
  - This is the only compatible distance. Another distance class raises
    `TypeError`; other settings raise `ValueError`.

## BatchEasyHardMiner

[Improved Embeddings with Easy Positive Triplet Mining](http://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf)

Returns positive and negative pairs according to `pos_strategy` and
`neg_strategy`. To implement the loss function described in the paper, use this
miner with [`NTXentLoss(temperature=0.1)`](losses/ntxent_loss.md#ntxentloss).

```python {.signature}
miners.BatchEasyHardMiner(
    pos_strategy="easy",
    neg_strategy="semihard",
    allowed_pos_range=None,
    allowed_neg_range=None,
    *,
    distance=None,
    tuples_per_anchor=None,
)
```

**Parameters**

- **pos_strategy**: One of the `pytorch_metric_learning.types.MiningStrategy`
  values:
  - `"hard"`: the hardest positive per anchor.
  - `"semihard"`: the hardest positive per anchor that is closer than the
    selected negative.
  - `"easy"`: the easiest positive per anchor.
  - `"all"`: every positive candidate.
- **neg_strategy**: One of:
  - `"hard"`: the hardest negative per anchor.
  - `"semihard"`: the hardest negative per anchor that is farther than the
    selected positive.
  - `"easy"`: the easiest negative per anchor.
  - `"all"`: every negative candidate.
- **allowed_pos_range**: Optional inclusive `(low, high)` bounds on
  anchor-positive values, in the distance's units, for example `(0.2, 1)`.
  `None` applies no bound.
- **allowed_neg_range**: Optional inclusive `(low, high)` bounds on
  anchor-negative values. `None` applies no bound.

When neither strategy is `"all"`, the miner returns one positive pair and one
negative pair for each anchor that has both a selected positive and a selected
negative.

**Restrictions**

These raise `ValueError` at construction:

- A strategy outside `"hard"`, `"semihard"`, `"easy"`, `"all"`.
- Both strategies set to `"semihard"`.
- `"semihard"` combined with `"all"`.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## BatchHardMiner

[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf)

For each anchor, finds the hardest positive and the hardest negative and returns
them as one triplet, so a batch of `N` anchors yields at most `N` triplets.
Anchors without a positive or a negative are omitted.

This miner equals `BatchEasyHardMiner(pos_strategy="hard", neg_strategy="hard")`
with its pairs combined into triplets.

```python {.signature}
miners.BatchHardMiner(*, distance=None, tuples_per_anchor=None)
```

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## DistanceWeightedMiner

Implementation of the miner from [Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf).

```python {.signature}
miners.DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4, *, distance=None, tuples_per_anchor=None)
```

**Parameters**

- **cutoff**: Distances below this value are raised to it before weighting.
- **nonzero_loss_cutoff**: Negatives at this distance or farther are discarded.

Each anchor with a positive and an eligible negative receives as many triplets
as it has positive candidates. Positives are sampled uniformly and negatives in
proportion to the inverse of the distance density on the unit sphere, both with
replacement. Sampling weights are normalized per anchor over its eligible
negatives, so any embedding dimension produces triplets. The density still
concentrates on a narrow distance band as dimensionality grows, so the
weighting is most informative for low-dimensional embeddings, such as 64
dimensions. [UniformHistogramMiner](#uniformhistogramminer) samples across the
distance range with any dimensionality and any distance.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)
  - This is the only compatible distance. Another distance class raises
    `TypeError`; other settings raise `ValueError`.

## HDCMiner

[Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)

```python {.signature}
miners.HDCMiner(filter_percentage=0.5, *, distance=None, tuples_per_anchor=None)
```

**Parameters**:

- **filter_percentage**: The fraction of candidate pairs to keep, in `(0, 1]`.
  Values outside this range raise `ValueError`.

Each call keeps the hardest `ceil(filter_percentage * n)` of the `n` positive
candidates and, separately, of the negative candidates. Without
`indices_tuple`, the candidates are all positive and negative pairs; with it,
they are the given [candidates](#candidates). Chaining HDC miners through
`indices_tuple` reproduces the paper's cascade; see the
[cascaded mining recipe](recipes/training.md#cascaded-mining-contract).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## MultiSimilarityMiner

[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python {.signature}
miners.MultiSimilarityMiner(epsilon=0.1, *, distance=None, tuples_per_anchor=None)
```

**Parameters**

- **epsilon**:
  - Negative pairs are chosen if their similarity is greater than that of the
    hardest positive pair, minus `epsilon`.
  - Positive pairs are chosen if their similarity is less than that of the
    hardest negative pair, plus `epsilon`.

With a distance instead of a similarity, the comparisons are reversed.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

## PairMarginMiner

Returns positive and negative pairs that violate the specified margins.

```python {.signature}
miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8, *, distance=None, tuples_per_anchor=None)
```

**Parameters**

- **pos_margin**: Positive pairs with a distance above (similarity below) this
  value are chosen.
- **neg_margin**: Negative pairs with a distance below (similarity above) this
  value are chosen.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## TripletMarginMiner

Returns hard, semihard, easy, or all margin-violating triplets.

```python {.signature}
miners.TripletMarginMiner(margin=0.2, type_of_triplets="all", *, distance=None, tuples_per_anchor=None)
```

**Parameters**

- **margin**: The required difference between the anchor-negative and the
  anchor-positive distance.
- **type_of_triplets**: One of `"all"`, `"hard"`, `"semihard"`, or `"easy"`;
  other values raise `ValueError`.
  - `"all"`: all triplets that violate the margin.
  - `"hard"`: the subset of `"all"` where the negative is at least as close to
    the anchor as the positive.
  - `"semihard"`: the subset of `"all"` where the negative is farther from the
    anchor than the positive.
  - `"easy"`: all triplets that do **not** violate the margin.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## UniformHistogramMiner

Returns pairs whose distances are spread uniformly over the observed range.
Like [DistanceWeightedMiner](#distanceweightedminer), it avoids concentrating on
one distance band, but it works with high-dimensional embeddings and any
distance.

```python {.signature}
miners.UniformHistogramMiner(
    num_bins=100,
    pos_per_bin=10,
    neg_per_bin=10,
    *,
    distance=None,
    tuples_per_anchor=None,
)
```

**Parameters**

- **num_bins**: The number of equal-width bins between the smallest and largest
  candidate value. For example, if the positive distances range from 0 to 2 and
  `num_bins=100`, each bin has width 0.02.
- **pos_per_bin**: The number of positive pairs drawn from each occupied bin.
- **neg_per_bin**: The number of negative pairs drawn from each occupied bin.

Positive and negative candidates are binned separately. Draws within a bin are
uniform with replacement, so a bin with fewer pairs than requested contributes
duplicates, and empty bins contribute nothing.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Example**:

```python
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.miners import UniformHistogramMiner

miner = UniformHistogramMiner(
    num_bins=100,
    pos_per_bin=25,
    neg_per_bin=33,
    distance=SNRDistance(),
)
```

In each batch, this miner returns 25 positive pairs from each occupied
positive-distance bin and 33 negative pairs from each occupied negative-distance
bin.
