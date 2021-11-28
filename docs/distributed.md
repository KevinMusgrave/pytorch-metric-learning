# Distributed

Wrap a tuple loss or miner with these when using PyTorch's [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (i.e. multiprocessing).


## DistributedLossWrapper 
```python
utils.distributed.DistributedLossWrapper(loss, efficient=False)
```

**Parameters**:

* **loss**: The loss function to wrap
* **efficient**: If False, memory usage is not optimal, but the resulting gradients will be identical to the non-distributed code. If True, memory usage is decreased, but gradients will differ from non-distributed code. 

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
utils.distributed.DistributedMinerWrapper(miner, efficient=False)
```

**Parameters**:

* **miner**: The miner to wrap
* **efficient**: If False, memory usage is not optimal, but the resulting gradients will be identical to the non-distributed code. If True, memory usage is decreased, but gradients will differ from non-distributed code. 

Example usage:
```python
from pytorch_metric_learning import miners
from pytorch_metric_learning.utils import distributed as pml_dist

miner = miners.MultiSimilarityMiner()
miner = pml_dist.DistributedMinerWrapper(miner)

# in each process
tuples = miner(embeddings, labels)
# pass into a DistributedLossWrapper
loss = loss_func(embeddings, labels, indices_tuple)
```
