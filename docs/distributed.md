# Distributed

Wrap a loss or miner with these when using PyTorch's [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (i.e. multiprocessing).


## DistributedLossWrapper 
```python
utils.distributed.DistributedLossWrapper(loss)
```

Example usage:
```python
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist

loss_func = losses.ContrastiveLoss()
loss_func = pml_dist.DistributedLossWrapper(loss_func)

# in each process during training
loss = loss_func(embeddings, labels)
```

## DistributedMinerWrapper
```python
utils.distributed.DistributedMinerWrapper(miner)
```

Example usage:
```python
from pytorch_metric_learning import miners
from pytorch_metric_learning.utils import distributed as pml_dist
miner = pml_dist.DistributedMinerWrapper(miner = miners.MultiSimilarityMiner())

# in each process
tuples = miner(embeddings, labels)
```
