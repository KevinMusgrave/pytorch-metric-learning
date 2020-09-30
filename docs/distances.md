# Distances

Distance classes compute pairwise distances/similarities between input embeddings.

Consider the TripletMarginLoss in its default form:
```python
from pytorch_metric_learning.losses import TripletMarginLoss
loss_func = TripletMarginLoss(margin=0.2)
```
This loss function attempts to minimize [d<sub>ap</sub> - d<sub>an</sub> + margin]<sub>+</sub>.

Typically, d<sub>ap</sub> and d<sub>an</sub> represent Euclidean or L2 distances. But what if we want to use a squared L2 distance, or an unnormalized L1 distance, or a completely different distance measure like signal-to-noise ratio? With the distances module, you can try out these ideas easily:
```python
### TripletMarginLoss with squared L2 distance ###
from pytorch_metric_learning.distances import LpDistance
loss_func = TripletMarginLoss(margin=0.2, distance=LpDistance(power=2))

### TripletMarginLoss with unnormalized L1 distance ###
loss_func = TripletMarginLoss(margin=0.2, distance=LpDistance(normalize_embeddings=False, p=1))

### TripletMarginLoss with signal-to-noise ratio###
from pytorch_metric_learning.distances import SNRDistance
loss_func = TripletMarginLoss(margin=0.2, distance=SNRDistance())
```

You can also use similarity measures rather than distances, and the loss function will make the necessary adjustments:
```python
### TripletMarginLoss with cosine similarity##
from pytorch_metric_learning.distances import CosineSimilarity
loss_func = TripletMarginLoss(margin=0.2, distance=CosineSimilarity())
```
With a similarity measure, the TripletMarginLoss internally swaps the anchor-positive and anchor-negative terms: [s<sub>an</sub> - s<sub>ap</sub> + margin]<sub>+</sub>. In other words, it will try to make the anchor-negative similarities smaller than the anchor-positive similarities.

All **losses, miners, and regularizers** accept a ```distance``` argument. So you can try out the ```MultiSimilarityMiner``` using ```SNRDistance```, or the ```NTXentLoss``` using ```LpDistance(p=1)``` and so on. Note that some losses/miners/regularizers have restrictions on the type of distances they can accept. For example, some classification losses only allow ```CosineSimilarity``` or ```DotProductSimilarity``` as their distance measure between embeddings and weights. To view restrictions for specific loss functions, see the [losses page](losses.md)

## BaseDistance

All distances extend this class and therefore inherit its ```__init__``` parameters.

```python
distances.BaseDistance(collect_stats = True,
	                         normalize_embeddings=True, 
	                         p=2, 
	                         power=1, 
	                         is_inverted=False)
```

**Parameters**:

* **collect_stats**: If True, will collect various statistics that may be useful to analyze during experiments. If False, these computations will be skipped.
* **normalize_embeddings**: If True, embeddings will be normalized to have an Lp norm of 1, before the distance/similarity matrix is computed.
* **p**: The distance norm.
* **power**: If not 1, each element of the distance/similarity matrix will be raised to this power.
* **is_inverted**: Should be set by child classes. If False, then small values represent embeddings that are close together. If True, then large values represent embeddings that are similar to each other.

**Required Implementations**:
```python
# Must return a matrix where mat[j,k] represents 
# the distance/similarity between query_emb[j] and ref_emb[k]
def compute_mat(self, query_emb, ref_emb):
    raise NotImplementedError

# Must return a tensor where output[j] represents 
# the distance/similarity between query_emb[j] and ref_emb[j]
def pairwise_distance(self, query_emb, ref_emb):
    raise NotImplementedError
```


## CosineSimilarity
```python
distances.CosineSimilarity(**kwargs)
```

The returned ```mat[i,j]``` is the cosine similarity between ```query_emb[i]``` and ```ref_emb[j]```. This class is equivalent to [```DotProductSimilarity(normalize_embeddings=True)```](distances.md#dotproductsimilarity).

## DotProductSimilarity
```python
distances.DotProductSimilarity(**kwargs)
```
The returned ```mat[i,j]``` is equal to ```torch.sum(query_emb[i] * ref_emb[j])```


## LpDistance
```python
distances.LpDistance(**kwargs)
```
The returned ```mat[i,j]``` is the Lp distance between ```query_emb[i]``` and ```ref_emb[j]```. With default parameters, this is the Euclidean distance.

## SNRDistance
[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}
```python
distances.SNRDistance(**kwargs)
```
The returned ```mat[i,j]``` is equal to:

```python
torch.var(query_emb[i] - ref_emb[j]) / torch.var(query_emb[i])
```