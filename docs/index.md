# PyTorch Metric Learning

## Installation:
Conda:
```
conda install pytorch-metric-learning -c metric-learning
```

Pip:
```
pip install pytorch-metric-learning
```

## Overview
Let’s try the vanilla [triplet margin loss](losses/#tripletmarginloss). In all examples, _embeddings_ is assumed to be of size (N, embedding_size), and _labels_ is of size (N).
```python
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(margin=0.1)
loss = loss_func(embeddings, labels)
```
Loss functions typically come with a variety of parameters. For example, with the TripletMarginLoss, you can control how many triplets per sample to use in each batch. You can also use all possible triplets within each batch:
```python
loss_func = losses.TripletMarginLoss(triplets_per_anchor="all")
```
Sometimes it can help to add a mining function:
```python
from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner(epsilon=0.1)
loss_func = losses.TripletMarginLoss(margin=0.1)
hard_pairs = miner(embeddings, labels)
loss = loss_func(embeddings, labels, hard_pairs)
```
In the above code, the miner finds positive and negative pairs that it thinks are particularly difficult. Note that even though the TripletMarginLoss operates on triplets, it’s still possible to pass in pairs. This is because the library automatically converts pairs to triplets and triplets to pairs, when necessary.

In general, all [loss functions](losses) take in embeddings and labels, with an optional indices_tuple argument (i.e. the output of a miner):
```python
# From BaseMetricLossFunction
def forward(self, embeddings, labels, indices_tuple=None)
```
And all [mining functions](miners) take in embeddings and labels:
```python
# From BaseMiner
def forward(self, embeddings, labels)
```

