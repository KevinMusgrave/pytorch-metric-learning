# PyTorch Metric Learning

## Google Colab Examples
See the [examples folder](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/README.md) for notebooks that show entire train/test workflows with logging and model saving.

## Installation
### Pip
```
pip install pytorch-metric-learning
```

**To get the latest dev version**:
```
pip install pytorch-metric-learning==0.9.88
```

**To install on Windows**:
```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-metric-learning
```

**To install with evaluation and logging capabilities (This will install the unofficial pypi version of faiss-gpu)**:
```
pip install pytorch-metric-learning[with-hooks]
```

**To install with evaluation and logging capabilities (CPU) (This will install the unofficial pypi version of faiss-cpu)**:
```
pip install pytorch-metric-learning[with-hooks-cpu]
```

### Conda
```
conda install pytorch-metric-learning -c metric-learning -c pytorch
```

**To use the testing module, you'll need faiss, which can be installed via conda as well. See the [installation instructions for faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).**


## Overview
Let’s try the vanilla [triplet margin loss](losses/#tripletmarginloss). In all examples, _embeddings_ is assumed to be of size (N, embedding_size), and _labels_ is of size (N).
```python
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(margin=0.1)
loss = loss_func(embeddings, labels) # in your training loop
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
hard_pairs = miner(embeddings, labels) # in your training loop
loss = loss_func(embeddings, labels, hard_pairs)
```
In the above code, the miner finds positive and negative pairs that it thinks are particularly difficult. Note that even though the TripletMarginLoss operates on triplets, it’s still possible to pass in pairs. This is because the library automatically converts pairs to triplets and triplets to pairs, when necessary.

Here's what the above examples look like in a typical training loop:
```python
from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner(epsilon=0.1)
loss_func = losses.TripletMarginLoss(margin=0.1)

# borrowed from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    embeddings = net(inputs)
    hard_pairs = miner(embeddings, labels)
    loss = loss_func(embeddings, labels, hard_pairs)
    loss.backward()
    optimizer.step()
```

For more complex approaches, like deep adversarial metric learning, use one of the [trainers](trainers).

To check the accuracy of your model, use one of the [testers](testers). Which tester should you use? Almost definitely [GlobalEmbeddingSpaceTester](testers/#globalembeddingspacetester), because it does what most metric-learning papers do. 

Also check out the [example Google Colab notebooks](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples).
