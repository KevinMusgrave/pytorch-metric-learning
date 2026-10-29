# Use PyTorch Metric Learning

Choose a loss for embeddings produced by your PyTorch model, then add mining,
custom distances, reduction, or regularization as needed. Applications own
training loops, data loading, optimizers, and logging.

## Install

```sh
pip install pytorch-metric-learning
pip install "pytorch-metric-learning[faiss]"       # Faiss retrieval backends
pip install "pytorch-metric-learning[clustering]"  # NMI and AMI metrics
```

Version 3 requires Python 3.12 or newer, PyTorch 2.14.0 or newer, and NumPy
2.5.3 or newer. See the [release notes](release-notes/3.0.0.md), the
[migration guide](migration-2-to-3.md), and the
[version 2 documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/v2/).
New to the library? Read the [concepts overview](concepts.md).

[![Your PyTorch code computes embeddings and calls the library's miner and loss to get a scalar loss, then calls backward and steps its optimizer. For evaluation, your code extracts embeddings and AccuracyCalculator retrieves nearest neighbors to compute metrics.](imgs/architecture.svg)](imgs/architecture.svg)

## Choose by task

| Task                                        | Public entry point                                          | Contract and runnable composition                                                                                                                    |
| ------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Learn from labels and pairs                 | `losses.ContrastiveLoss`                                    | [Contrastive loss](losses/contrastive_loss.md#contrastiveloss), [training recipe](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs) |
| Learn relative distances                    | `losses.TripletMarginLoss`                                  | [Triplet margin loss](losses/triplet_margin_loss.md#tripletmarginloss)                                                                               |
| Train learnable class proxies               | `losses.NormalizedSoftmaxLoss`, `losses.ProxyAnchorLoss`    | [Proxy parameters and optimizer ownership](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs)                                        |
| Select difficult pairs or triplets          | `miners`                                                    | [Miners](miners.md), [cascaded mining](recipes/training.md#cascaded-mining-contract)                                                                 |
| Compare separate embedding streams          | Loss `ref_emb` / `ref_labels` arguments                     | [Two-stream gradients](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs)                                                            |
| Use augmented views                         | `losses.SelfSupervisedLoss`, `losses.VICRegLoss`            | [Self-supervision](losses/self_supervised_loss.md#selfsupervisedloss)                                                                                |
| Reuse detached embeddings                   | `losses.CrossBatchMemory`                                   | [Memory contract](losses/cross_batch_memory.md#crossbatchmemory), [warmup](recipes/training.md#memory-warmup-contract)                               |
| Balance batches by label                    | `samplers.MPerClassSampler`, `samplers.HierarchicalSampler` | [Samplers](samplers.md)                                                                                                                              |
| Sample from mined tuple frequency           | `utils.indexing.convert_to_weights`                         | [Weighted sampling](recipes/training.md#mining-weighted-sampling-contract)                                                                           |
| Evaluate retrieval                          | `evaluation.AccuracyCalculator`                             | [Metrics and query/reference rules](accuracy_calculation.md), [evaluation recipe](recipes/evaluation.md)                                             |
| Evaluate hierarchical or two-stream labels  | `AccuracyCalculator` with explicit subsets                  | [Parent-conditioned and two-stream evaluation](recipes/evaluation.md#parent-conditioned-hierarchical-retrieval)                                      |
| Find nearest neighbors or threshold matches | `evaluation.CustomKNN`, `evaluation.MatchFinder`            | [Retrieval backends](retrieval.md)                                                                                                                   |
| Train across devices                        | `distributed.DistributedLossWrapper`                        | [Distributed values and gradients](distributed.md)                                                                                                   |
| Generate hard negatives                     | Ordinary PyTorch modules and triplet loss                   | [Adversarial objectives](recipes/training.md#adversarial-objectives-and-gradient-ownership)                                                          |

## Component references

- [Concepts](concepts.md): how embeddings, tuples, distances, loss terms, and reducers fit together.
- [Losses](losses.md), [miners](miners.md), [distances](distances.md),
  [reducers](reducers.md), [regularizers](regularizers.md), and [samplers](samplers.md).
- [Retrieval metrics](accuracy_calculation.md) and [retrieval backends](retrieval.md).
- [Distributed execution](distributed.md) and [measured performance](performance.md).
- [Custom losses](extend/losses.md) and [custom miners](extend/miners.md).
- [Versioning and support](versioning.md) and [how to cite](citing.md).

## Execute a recipe

The [training](recipes/training.md) and [evaluation](recipes/evaluation.md)
recipes list their inputs, expected results, supported composition, and exact
verification command. They use synthetic tensors and download nothing.

## Imports

```python
from pytorch_metric_learning import distances, losses, miners, reducers, regularizers, samplers
from pytorch_metric_learning.evaluation import AccuracyCalculator, CustomKNN, MatchFinder
from pytorch_metric_learning.distributed import DistributedLossWrapper
from pytorch_metric_learning.utils import indexing
from pytorch_metric_learning.types import PairIndices, TripletIndices
```

`utils.indexing` holds the public pair, triplet, and weight utilities used by
custom losses and miners; other modules under `utils` are private. Shared type
annotations live in `pytorch_metric_learning.types`. Names starting with an
underscore are private.
