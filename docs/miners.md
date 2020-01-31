# Miners
Mining functions come in two flavors:

* **[pre-gradient miners](miners.md#basepregradientminer)** output indices corresponding to a subset of the input batch. The idea is to use these miners with torch.no_grad(), and with a large input batch size.
* **[post-gradient miners](miners.md#basepostgradientminer)** output a tuple of indices: 
	* Pair miners output a tuple of size 4: (anchors, positives, anchors, negatives)
	* Triplet miners output a tuple of size 3: (anchors, positives, negatives) 

**Almost all miners are post-gradient miners.**

Post-gradient miners are used with loss functions as follows:

```python
from pytorch_metric_learning import miners, losses
miner_func = miners.SomeMiner()
loss_func = losses.SomeLoss()
miner_output = miner_func(embeddings, labels)
losses = loss_func(embeddings, labels, miner_output)
```

## AngularMiner

```python
miners.AngularMiner(angle, **kwargs)
```

**Parameters**

* **angle**: The miner will return triplets that form an angle greater than this input angle. The angle is computed as defined in the [angular loss paper](https://arxiv.org/abs/1708.01682)

## BaseMiner
All miners extend this class and therefore inherit its _\_\_init\_\__ parameters.
```python
miners.BaseMiner(normalize_embeddings=True)
```

**Parameters**

* **normalize_embeddings**: If True, embeddings will be normalized to have a Euclidean norm of 1 before any mining occurs.

**Required Implementations**:
```python
# Return indices of some form
def mine(self, embeddings, labels):
    raise NotImplementedError

# Validate the output of the miner. 
def output_assertion(self, output):
	raise NotImplementedError
```

## BasePostGradientMiner
This extends [BaseMiner](miners.md#baseminer), and most miners extend this class. 

It outputs a tuple of indices:

* Pair miners output a tuple of size 4: (anchors, positives, anchors, negatives)
* Triplet miners output a tuple of size 3: (anchors, positives, negatives) 

```python
miners.BasePostGradientMiner(**kwargs)
```

## BasePreGradientMiner
This extends [BaseMiner](miners.md#baseminer). It outputs indices corresponding to a subset of the input batch. The idea is to use these miners with torch.no_grad(), and with a large input batch size.
```python
miners.BasePreGradientMiner(output_batch_size, **kwargs)
```

**Parameters**

* **output_batch_size**: An integer that is the size of the subset that the miner will output.

## BatchHardMiner

[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf)

```python
miners.BatchHardMiner(use_similarity=False, squared_distances=False, **kwargs)
```

**Parameters**

* **use_similarity**: If True, will use dot product between vectors instead of euclidean distance.
* **squared_distances**: If True, then the euclidean distance will be squared.

## DistanceWeightedMiner
[Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf)
```python
miners.DistanceWeightedMiner(cutoff, nonzero_loss_cutoff, **kwargs)
```

**Parameters**

* **cutoff**: Pairwise distances are clipped to this value if they fall below it.
* **nonzero_loss_cutoff**: Pairs that have distance greater than this are discarded.

## EmbeddingsAlreadyPackagedAsTriplets
If your embeddings are already ordered sequentially as triplets, then use this miner to force your loss function to use the already-formed triplets.

```python
miners.EmbeddingsAlreadyPackagedAsTriplets()
``` 

## HDCMiner
[Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)
```python
miners.HDCMiner(filter_percentage, use_similarity=False, squared_distances=False, **kwargs)
```

**Parameters**:

* **filter_percentage**: The percentage of pairs that will be returned. For example, if filter_percentage is 0.25, then the hardest 25% of pairs will be returned. The pool of pairs is either externally or internally set. See the important methods below for details.
* **use_similarity**: If True, will use dot product between vectors instead of euclidean distance.
* **squared_distances**: If True, then the euclidean distance will be squared.

**Important methods**:
```python
# Pairs or triplets extracted from another miner, 
# and then passed in to HDCMiner using this function
def set_idx_externally(self, external_indices_tuple, labels):
    self.a1, self.p, self.a2, self.n = lmu.convert_to_pairs(external_indices_tuple, labels)
    self.was_set_externally = True

# Reset the internal state of the HDCMiner
def reset_idx(self):
    self.a1, self.p, self.a2, self.n = None, None, None, None
    self.was_set_externally = False
```

**Example of passing another miner output to HDCMiner**:
```python
minerA = miners.MultiSimilarityMiner(epsilon=0.1)
minerB = miners.HDCMiner(filter_percentage=0.25)

hard_pairs = minerA(embeddings, labels)
minerB.set_idx_externally(hard_pairs, labels)
very_hard_pairs = minerB(embeddings, labels)
```

## MaximumLossMiner
This is a simple pre-gradient miner. It computes the loss for random subsets of the input batch, _num\_trials_ times. Then it returns the subset with the highest loss.

```python
miners.MaximumLossMiner(loss_function, mining_function=None, num_trials=5, **kwargs)
```

**Parameters**

* **loss_function**: The loss function used to compute the loss.
* **mining_functions**: Optional post-gradient mining function which extracts pairs/triplets for the loss function.
* **num_trials**: The number of random subsets to try.

## MultiSimilarityMiner

[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python
miners.MultiSimilarityMiner(epsilon, **kwargs)
```

**Parameters**

* **epsilon**: 
	* Negative pairs are chosen if they have similarity greater than the hardest positive pair, minus this margin (epsilon). 
	* Positive pairs are chosen if they have similarity less than the hardest negative pair, plus this margin (epsilon). 


## PairMarginMiner
Returns positive and negative pairs that violate the specified margins.
```python
miners.PairMarginMiner(pos_margin, neg_margin, use_similarity, squared_distances=False, **kwargs)
```

**Parameters**

* **pos_margin**: The distance (or similarity) over (under) which positive pairs will be chosen.
* **neg_margin**: The distance (or similarity) under (over) which negative pairs will be chosen.  
* **use_similarity**: If True, will use dot product between vectors instead of euclidean distance.
* **squared_distances**: If True, then the euclidean distance will be squared.

## TripletMarginMiner
Returns hard, semihard, or all triplets.
```python
miners.TripletMarginMiner(margin, type_of_triplets="all", **kwargs)
```

**Parameters**

* **margin**: The difference between the anchor-positive distance and the anchor-negative distance.
* **type_of_triplets**: 
	* "all" means all triplets that violate the margin
	* "hard" is a subset of "all", but the negative is closer to the anchor than the positive
	* "semihard" is a subset of "all", but the negative is further from the anchor than the positive