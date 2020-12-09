# Distributed

Wrap a loss or miner with these when using PyTorch's [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (i.e. multiprocessing).


## DistributedLossWrapper 
```python
utils.distributed.DistributedLossWrapper(loss, **kwargs)
```
The ```**kwargs``` allow you to pass in arguments which will be fed to ```DistributedDataParallel```. This is used only if the loss function has trainable parameters (e.g. ArcFaceLoss).

Example usage for a loss **without** parameters:
```python
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist

loss_func = losses.ContrastiveLoss()
loss_func = pml_dist.DistributedLossWrapper(loss = loss_func)

# in each process during training
loss = loss_func(embeddings, labels)
```


Example usage for a loss **with** parameters:
```python
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import distributed as pml_dist

loss_func = losses.ArcFaceLoss(num_classes = 100, embedding_size = 512)

# Pass in the rank of the process
# The loss function will be wrapped with DistributedDataParallel
# And device_ids = [rank] will be passed into the init
loss_func = pml_dist.DistributedLossWrapper(loss = loss_func, device_ids = [rank])

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
