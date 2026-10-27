# Use PyTorch Metric Learning

Choose a loss for embeddings produced by your PyTorch model, then add mining,
custom distances, reduction, or regularization as needed. Applications own
training loops, data loading, optimizers, and logging.

For a complete first training step and PyPI installation with uv or pip, start
at the [repository README](https://github.com/KevinMusgrave/pytorch-metric-learning#readme).
For this development branch, read [the next major release notes](release-notes-3.md).

## Choose by task

| Task                                        | Public entry point                                          | Contract and runnable composition                                                                                                   |
| ------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Learn from labels and pairs                 | `losses.ContrastiveLoss`                                    | [Contrastive loss](losses.md#contrastiveloss), [training recipe](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs) |
| Learn relative distances                    | `losses.TripletMarginLoss`                                  | [Triplet margin loss](losses.md#tripletmarginloss)                                                                                  |
| Train learnable class proxies               | `losses.NormalizedSoftmaxLoss`, `losses.ProxyAnchorLoss`    | [Proxy parameters and optimizer ownership](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs)                       |
| Select difficult pairs or triplets          | `miners`                                                    | [Miners](miners.md), [cascaded mining](recipes/training.md#cascaded-mining-contract)                                                |
| Compare separate embedding streams          | Loss `ref_emb` / `ref_labels` arguments                     | [Two-stream gradients](recipes/training.md#metric-classifier-proxy-and-two-stream-inputs)                                           |
| Use augmented views                         | `losses.SelfSupervisedLoss`, `losses.VICRegLoss`            | [Self-supervision](losses.md#selfsupervisedloss)                                                                                    |
| Reuse detached embeddings                   | `losses.CrossBatchMemory`                                   | [Memory contract](losses.md#crossbatchmemory), [warmup](recipes/training.md#memory-warmup-contract)                                 |
| Balance batches by label                    | `samplers.MPerClassSampler`, `samplers.HierarchicalSampler` | [Samplers](samplers.md)                                                                                                             |
| Sample from mined tuple frequency           | `utils.loss_and_miner_utils.convert_to_weights`             | [Weighted sampling](recipes/training.md#mining-weighted-sampling-contract)                                                          |
| Evaluate retrieval                          | `utils.accuracy_calculator.AccuracyCalculator`              | [Metrics and query/reference rules](accuracy_calculation.md)                                                                        |
| Find nearest neighbors or threshold matches | `utils.inference.CustomKNN`, `utils.inference.MatchFinder`  | [Retrieval primitives](inference_models.md)                                                                                         |
| Train across devices                        | `utils.distributed.DistributedLossWrapper`                  | [Distributed values and gradients](distributed.md)                                                                                  |
| Generate hard negatives                     | Ordinary PyTorch modules and triplet loss                   | [Adversarial objectives](recipes/training.md#adversarial-objectives-and-gradient-ownership)                                         |

## Component references

- [Losses](losses.md): objectives, defaults, supported arguments, and papers.
- [Miners](miners.md): candidate selection and index conventions.
- [Distances](distances.md): normalization, distance, and similarity orientation.
- [Reducers](reducers.md): filtering and aggregation of per-tuple penalties.
- [Regularizers](regularizers.md): embedding and learnable-weight penalties.
- [Samplers](samplers.md): label structure and PyTorch data-loader integration.
- [Retrieval metrics](accuracy_calculation.md) and [retrieval primitives](inference_models.md).
- [Distributed execution](distributed.md) and [measured performance](performance.md).
- [Custom losses](extend/losses.md) and [custom miners](extend/miners.md).

## Execute a recipe

The small [training recipes](recipes/training.md) sit beside their contracts.
Each lists its inputs, expected results, supported composition, and exact
verification command. They use synthetic tensors and no downloads. There is no
separate tutorial gallery or notebook environment to set up.

The repository's frozen environment and test commands are documented in the
[development guide](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/DEVELOPMENT.md).
