# Miners

Miners select which pairs or triplets in a batch contribute to the loss:

- Pair miners return a tuple of four index tensors: (anchors, positives, anchors, negatives).
- Triplet miners return a tuple of three index tensors: (anchors, positives, negatives).
- Without a miner, losses use all possible pairs or triplets in the batch.

These are online miners: they operate on the embeddings of the current batch.
Offline selection belongs in a [PyTorch sampler](samplers.md).

Miners are used with loss functions as follows:

```python
from pytorch_metric_learning import miners, losses
miner_func = miners.SomeMiner()
loss_func = losses.SomeLoss()
miner_output = miner_func(embeddings, labels)
losses = loss_func(embeddings, labels, miner_output)
```

## AngularMiner

```python
miners.AngularMiner(angle=20, *, distance=None)
```

**Parameters**

- **angle**: The miner will return triplets that form an angle greater than this input angle. The angle is computed as defined in the [angular loss paper](https://arxiv.org/abs/1708.01682)

**Default distance**:

- [`LpDistance(p=2, power=1, normalize_embeddings=True)`](distances.md#lpdistance)

  - This is the only compatible distance.

## BaseMiner

All miners extend this class and accept its keyword-only `distance` option.

```python
miners.BaseMiner(*, distance=None)
```

Every miner outputs a tuple of indices:

- Pair miners output a tuple of size 4: (anchors, positives, anchors, negatives)
- Triplet miners output a tuple of size 3: (anchors, positives, negatives)

See [custom miners](extend/miners.md) for details on how to write your own miner.

**Parameters**:

- **distance**: A [distance](distances.md) object. If None, then the default distance will be used.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Required Implementations**:

```python
def mine(self, embeddings, labels, ref_emb, ref_labels):
    raise NotImplementedError
```

By default, `ref_emb` is `embeddings` and `ref_labels` is `labels`. `forward`
runs `mine` under `torch.no_grad()` and validates its output.

## BatchEasyHardMiner

[Improved Embeddings with Easy Positive Triplet Mining](http://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf)

Returns positive and negative pairs according to the specified `pos_strategy` and `neg_strategy`.

To implement the loss function described in the paper, use this miner in combination with [`NTXentLoss(temperature=0.1)`](losses/ntxent_loss.md#ntxentloss).

```python
miners.BatchEasyHardMiner(
    pos_strategy="easy",
    neg_strategy="semihard",
    allowed_pos_range=None,
    allowed_neg_range=None,
    *,
    distance=None,
)
```

**Parameters**

- **pos_strategy**: one of the following:
  - `"hard"`: returns the hardest positive sample per anchor.
  - `"semihard"`: returns the hardest positive sample per anchor, such that it is closer than the selected negative.
  - `"easy"`: returns the easiest positive sample per anchor
  - `"all"`: returns all possible positive pairs
- **neg_strategy**: one of the following:
  - `"hard"`: returns the hardest negative sample per anchor.
  - `"semihard"`: returns the hardest negative sample per anchor, such that it is further than the selected positive.
  - `"easy"`: returns the easiest negative sample per anchor
  - `"all"`: returns all possible negative pairs
- **allowed_pos_range**: Optional `(low, high)` pair bounding the anchor-positive distances or similarities, for example `(0.2, 1)`. If `None`, no range is applied.
- **allowed_neg_range**: Optional `(low, high)` pair bounding the anchor-negative distances or similarities. If `None`, no range is applied.

**Restrictions**

- `pos_strategy` and `neg_strategy` cannot both be set to `"semihard"`
- If `pos_strategy` is set to `"semihard"`, then `neg_strategy` cannot be set to `"all"`, and vice versa.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## BatchHardMiner

[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf)

For each element in the batch, this miner will find the hardest positive and hardest negative, and use those to form a single triplet. So for a batch size of N, this miner will output N triplets.

This miner is equivalent to using `miners.BatchEasyHardMiner(pos_strategy="hard", neg_strategy="hard")`, and converting the output pairs to triplets.

```python
miners.BatchHardMiner(*, distance=None)
```

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## DistanceWeightedMiner

Implementation of the miner from [Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf).

```python
miners.DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4, *, distance=None)
```

**Parameters**

- **cutoff**: Pairwise distances are clipped to this value if they fall below it.
- **nonzero_loss_cutoff**: Pairs that have distance greater than this are discarded.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)
  - This is the only compatible distance.

**Important note**:

Sampling weights are normalized per anchor over its eligible negatives, so a batch of a few hundred rows produces triplets at any embedding dimension. The paper's weighting still concentrates on a narrow distance band as dimensionality grows, so this miner works best with low dimensionality embeddings (e.g 64-dim) and L2-normalized distances. Check out [UniformHistogramMiner](#uniformhistogramminer) for a miner that is roughly equivalent, but works with embeddings of any dimensionality and any distance metric.

## HDCMiner

[Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)

```python
miners.HDCMiner(filter_percentage=0.5, *, distance=None)
```

**Parameters**:

- **filter_percentage**: The percentage of pairs that will be returned. For example, if filter_percentage is 0.25, then the hardest 25% of pairs will be returned. The pool of pairs is either externally or internally set. See the important methods below for details.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Candidate pairs**:

```python
miner(embeddings, labels, ref_emb=None, ref_labels=None, candidates=None)
```

With `candidates=None`, the miner considers every positive and negative pair.
Pass the output of another miner, pairs or triplets, to restrict the selection
to those candidates; triplets are converted to pairs. Each call keeps the
hardest `filter_percentage` of the positive and of the negative candidates,
rounding up. The miner keeps no candidate state between calls.

**Example of passing another miner output to HDCMiner**:

```python
minerA = miners.MultiSimilarityMiner(epsilon=0.1)
minerB = miners.HDCMiner(filter_percentage=0.25)

hard_pairs = minerA(embeddings, labels)
very_hard_pairs = minerB(embeddings, labels, candidates=hard_pairs)
```

## MultiSimilarityMiner

[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python
miners.MultiSimilarityMiner(epsilon=0.1, *, distance=None)
```

**Parameters**

- **epsilon**:
  - Negative pairs are chosen if they have similarity greater than the hardest positive pair, minus this margin (epsilon).
  - Positive pairs are chosen if they have similarity less than the hardest negative pair, plus this margin (epsilon).

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

## PairMarginMiner

Returns positive and negative pairs that violate the specified margins.

```python
miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8, *, distance=None)
```

**Parameters**

- **pos_margin**: The distance (or similarity) over (under) which positive pairs will be chosen.
- **neg_margin**: The distance (or similarity) under (over) which negative pairs will be chosen.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## TripletMarginMiner

Returns hard, semihard, or all triplets.

```python
miners.TripletMarginMiner(margin=0.2, type_of_triplets="all", *, distance=None)
```

**Parameters**

- **margin**: The difference between the anchor-positive distance and the anchor-negative distance.
- **type_of_triplets**: one of `"all"`, `"hard"`, `"semihard"`, or `"easy"`.
  - `"all"`: all triplets that violate the margin
  - `"hard"`: the subset of `"all"` where the negative is closer to the anchor than the positive
  - `"semihard"`: the subset of `"all"` where the negative is farther from the anchor than the positive
  - `"easy"`: all triplets that do **not** violate the margin

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

## UniformHistogramMiner

Returns pairs that have uniformly distributed distances. This is like [DistanceWeightedMiner](#distanceweightedminer), except that it works well with high dimension embeddings, and works with any distance metric (not just L2 normalized distance).

```python
miners.UniformHistogramMiner(num_bins=100,
                            pos_per_bin=10,
                            neg_per_bin=10,
                            *, distance=None):
```

**Parameters**

- **num_bins**: The number of bins to divide the distances into. For example, if the distances for the current batch range from 0 to 2, and num_bins = 100, then each bin will have a width of 0.02.
- **pos_per_bin**: The number of positive pairs to mine for each bin.
- **neg_per_bin**: The number of negative pairs to mine for each bin.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Example**:

```python
from pytorch_metric_learning.miners import UniformHistogramMiner
from pytorch_metric_learning.distances import SNRDistance

miner = UniformHistogramMiner(
    num_bins=100,
    pos_per_bin=25,
    neg_per_bin=33,
    distance=SNRDistance(),
)
```

In a given batch, this will:

- Divide up the positive distances into 100 bins, and return 25 positive pairs per bin, or 0 if no pairs exist in that bin
- Divide up the negative distances into 100 bins, and return 33 negative pairs per bin, or 0 if no pairs exist in that bin

Every miner's output is validated: three triplet index tensors or four pair
index tensors, each one-dimensional, with matching anchor and partner lengths.
Invalid types raise `TypeError`; invalid shapes or lengths raise `ValueError`,
including when Python optimizations are enabled.
