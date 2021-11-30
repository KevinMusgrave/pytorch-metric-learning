# Distributed

Wrap a tuple loss or miner with these when using PyTorch's [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (i.e. multiprocessing).


## DistributedLossWrapper 
```python
utils.distributed.DistributedLossWrapper(loss, efficient=False)
```

**Parameters**:

* **loss**: The loss function to wrap
* **efficient**:
    * ```True```: each process uses its own embeddings for anchors, and the gathered embeddings for positives/negatives. Gradients will **not** be equal to those in non-distributed code, but the benefit is reduced memory and faster training.
    * ```False```: each process uses gathered embeddings for both anchors and positives/negatives. Gradients will be equal to those in non-distributed code, but at the cost of doing unnecessary operations (i.e. doing computations where both anchors and positives/negatives have no gradient).

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
* **efficient**: If your distributed loss function has ```efficient=True``` then you must also set the distributed miner's ```efficient``` to True.

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
