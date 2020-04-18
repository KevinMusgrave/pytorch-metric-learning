# PyTorch Metric Learning

## Installation
**Pip**:
```
pip install pytorch-metric-learning
```

**To get the latest dev version**:
```
pip install pytorch-metric-learning==0.9.84
```

**To install on Windows**:
```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-metric-learning
```

**Conda**:
```
conda install pytorch-metric-learning -c metric-learning
```
We have recently noticed some sporadic issues with the conda installation, so we recommend installing with pip. You can use pip inside of conda:
```
conda install pip
pip install pytorch-metric-learning
```
If you run into problems during installation, please post in [this issue](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/55#issue-600601602).



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
And (almost) all [mining functions](miners) take in embeddings and labels:
```python
# From BaseMiner
def forward(self, embeddings, labels)
```

For more complex approaches, like deep adversarial metric learning, use one of the [trainers](trainers).

To check the accuracy of your model, use one of the [testers](testers). Which tester should you use? Almost definitely [GlobalEmbeddingSpaceTester](testers/#globalembeddingspacetester), because it does what most metric-learning papers do. 

Also check out the [example scripts](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples).
