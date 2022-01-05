# Wrappers
Loss wrappers can be used as follows:

```python
from pytorch_metric_learning import losses
from pytorch_metric_learning,wrappers import SelfSupervisedLossWrapper

loss_fn = losses.SomeLoss()
loss_fn = SelfSupervisedLossWrapper(loss_fn)

loss = loss_fn(embeddings, labels)
```

## SelfSupervisedLossWrapper

A common use case is to have embeddings and ref_emb be augmented versions of each other. For most losses right now you have to create labels to indicate which embeddings correspond with which ref_emb. `SelfSupervisedLossWrapper` automates this.

```python
loss_fn = losses.TripletMarginLoss()
loss_fn = SelfSupervisedLossWrapper(loss_fn)
loss = loss_fn(embeddings, labels)
```

**Supported Loss Functions**:
 - [AngularLoss](losses.md#AngularLoss)
 - [CircleLoss](losses.md#CircleLoss)
 - [ContrastiveLoss](losses.md#ContrastiveLoss)
 - [IntraPairVarianceLoss](losses.md#IntraPairVarianceLoss)
 - [MultiSimilarityLoss](losses.md#MultiSimilarityLoss)
 - [NTXentLoss](losses.md#NTXentLoss)
 - [SignalToNoiseRatioContrastiveLoss](losses.md#SignalToNoiseRatioContrastiveLoss)
 - [SupConLoss](losses.md#SupConLoss)
 - [TripletMarginLoss](losses.md#TripletMarginLoss)
 - [TupletMarginLoss](losses.md#TupletMarginLoss)

## CrossBatchMemory 
This wraps a loss function, and implements [Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf){target=_blank}. It stores embeddings from previous iterations in a queue, and uses them to form more pairs/triplets with the current iteration's embeddings.

```python
wrappers.CrossBatchMemory(loss, embedding_size, memory_size=1024, miner=None)
```

**Parameters**:

* **loss**: The loss function to be wrapped. For example, you could pass in ```ContrastiveLoss()```.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **memory_size**: The size of the memory queue.
* **miner**: An optional [tuple miner](miners.md), which will be used to mine pairs/triplets from the memory queue.

**Forward function**
```python
loss_fn(embeddings, labels, indices_tuple=None, enqueue_idx=None)
```

As shown above, CrossBatchMemory comes with a 4th argument in its ```forward``` function:

* **enqueue_idx**: The indices of ```embeddings``` that will be added to the memory queue. In other words, only ```embeddings[enqueue_idx]``` will be added to memory. This enables CrossBatchMemory to be used in self-supervision frameworks like [MoCo](https://arxiv.org/pdf/1911.05722.pdf). Check out the [MoCo on CIFAR100](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples#simple-examples) notebook to see how this works.


**Supported Loss Functions**:
 - [AngularLoss](losses.md#AngularLoss)
 - [CircleLoss](losses.md#CircleLoss)
 - [ContrastiveLoss](losses.md#ContrastiveLoss)
 - [GeneralizedLiftedStructureLoss](losses.md#GeneralizedLiftedStructureLoss)
 - [IntraPairVarianceLoss](losses.md#IntraPairVarianceLoss)
 - [LiftedStructureLoss](losses.md#LiftedStructureLoss)
 - [MultiSimilarityLoss](losses.md#MultiSimilarityLoss)
 - [NTXentLoss](losses.md#NTXentLoss)
 - [SignalToNoiseRatioContrastiveLoss](losses.md#SignalToNoiseRatioContrastiveLoss)
 - [SupConLoss](losses.md#SupConLoss)
 - [TripletMarginLoss](losses.md#TripletMarginLoss)
 - [TupletMarginLoss](losses.md#TupletMarginLoss)



## MultipleLosses
This is a simple wrapper for multiple losses. Pass in a list of already-initialized loss functions. Then, when you call forward on this object, it will return the sum of all wrapped losses.
```python
losses.MultipleLosses(losses, miners=None, weights=None)
```
**Parameters**:

* **losses**: A list or dictionary of initialized loss functions. On the forward call of MultipleLosses, each wrapped loss will be computed, and then the average will be returned.
* **miners**: Optional. A list or dictionary of mining functions. This allows you to pair mining functions with loss functions. For example, if ```losses = [loss_A, loss_B]```, and ```miners = [None, miner_B]``` then no mining will be done for ```loss_A```, but the output of ```miner_B``` will be passed to ```loss_B```. The same logic applies if ```losses = {"loss_A": loss_A, "loss_B": loss_B}``` and ```miners = {"loss_B": miner_B}```.
* **weights**: Optional. A list or dictionary of loss weights, which will be multiplied by the corresponding losses obtained by the loss functions. The default is to multiply each loss by 1. If ```losses``` is a list, then ```weights``` must be a list. If ```losses``` is a dictionary, ```weights``` must contain the same keys as ```losses```. 
