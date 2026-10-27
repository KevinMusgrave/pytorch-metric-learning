![PyTorch Metric Learning](docs/imgs/Logo2.png)

Train embedding models with composable PyTorch losses, hard-example miners,
distances, reducers, and regularizers. Use them in your own training loop for
retrieval, clustering, recognition, or learning from pairs and triplets.

## Install

Install the published package from PyPI:

```sh
uv pip install pytorch-metric-learning
```

The core package needs only PyTorch and NumPy. Optional extras add the Faiss
retrieval backend, `pytorch-metric-learning[faiss]`, and scikit-learn for the
NMI/AMI clustering metrics, `pytorch-metric-learning[clustering]`.

This branch is preparing the next major release. Its development baseline is
Python 3.12.13 and PyTorch 2.14.0; published versions have their own requirements.
See the [release notes](docs/release-notes-3.md) for breaking changes and
[development setup](DEVELOPMENT.md) to run this checkout.

## A minimal training step

Your model produces one embedding per input. Equal labels identify positive
pairs; different labels identify negatives. Here a contrastive loss pulls
positive pairs together and pushes negatives apart:

```python
import torch
from pytorch_metric_learning import losses

torch.manual_seed(17)
model = torch.nn.Linear(8, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
data = torch.randn(12, 8)
labels = torch.arange(4).repeat_interleave(3)
loss_func = losses.ContrastiveLoss(neg_margin=1.4)

optimizer.zero_grad(set_to_none=True)
loss = loss_func(model(data), labels)
loss.backward()
optimizer.step()
```

The model returns embeddings shaped `(12, 4)` and labels have shape `(12,)`.
This loss L2-normalizes embeddings by default. It averages active positive and
negative pair penalties separately, then adds them. See the
[contrastive-loss contract](docs/losses.md#contrastiveloss) for distances,
margins, and reduction options.

## How it fits together

[![Your PyTorch code computes embeddings and calls the library's miner and loss to get a scalar loss, then calls backward and steps its optimizer. For evaluation, your code extracts embeddings and AccuracyCalculator retrieves nearest neighbors to compute metrics.](docs/imgs/architecture.svg)](docs/imgs/architecture.svg)

Your code keeps the training loop. The library turns embeddings and labels
into a differentiable loss, and turns query and reference embeddings into
retrieval metrics.

## Find the right component

Start with the [task and component index](docs/index.md). It links to public
signatures, defaults, tensor contracts, paper references, and small runnable
recipes using ordinary PyTorch code.

- [Losses](docs/losses.md) define the training objective.
- [Miners](docs/miners.md) select pairs or triplets within a batch.
- [Distances](docs/distances.md), [reducers](docs/reducers.md), and
  [regularizers](docs/regularizers.md) customize the computation.
- [Samplers](docs/samplers.md) construct batches with useful label structure.
- [Retrieval metrics](docs/accuracy_calculation.md) measure embedding quality.
- [Distributed support](docs/distributed.md) documents multi-device gradients
  and supported combinations.

The [documentation website](https://kevinmusgrave.github.io/pytorch-metric-learning/)
resides alongside the Markdown sources in this repository. For timing and memory
measurements, see [performance](docs/performance.md).

## Contributing and credit

See [DEVELOPMENT.md](DEVELOPMENT.md) for environment setup and verification,
and [AGENTS.md](AGENTS.md) for repository maintenance rules. Thanks to all
[contributors, upstream projects, and supporters](CREDITS.md).

## Citing this library

If you'd like to cite pytorch-metric-learning in your paper, you can use this bibtex:

```latex
@article{Musgrave2020PyTorchML,
  title={PyTorch Metric Learning},
  author={Kevin Musgrave and Serge J. Belongie and Ser-Nam Lim},
  journal={ArXiv},
  year={2020},
  volume={abs/2008.09164}
}
```
