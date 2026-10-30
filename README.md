![PyTorch Metric Learning](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/master/docs/imgs/Logo2.png)

Train embedding models with composable PyTorch losses, hard-example miners,
distances, reducers, and regularizers. Use them in your own training loop for
retrieval, clustering, recognition, or learning from pairs and triplets.

[Docs for humans](https://kevinmusgrave.github.io/pytorch-metric-learning/) ·
[Documentation source](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/docs/index.md)

[![How embeddings flow through miners, losses, reducers, and retrieval metrics.](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/master/docs/imgs/architecture.svg)](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/master/docs/imgs/architecture.svg)

## Install

Install the version 3 release candidate from PyPI with uv or pip:

```sh
uv pip install "pytorch-metric-learning==3.0.0rc1"
pip install "pytorch-metric-learning==3.0.0rc1"
```

The core package needs only PyTorch and NumPy. Optional extras add the Faiss
retrieval backend, `pytorch-metric-learning[faiss]`, and scikit-learn for the
NMI/AMI clustering metrics, `pytorch-metric-learning[clustering]`.

Version 3 requires Python 3.12 or newer, PyTorch 2.14.0 or newer, and NumPy
2.5.3 or newer. Upgrading from version 2? Read the
[release notes](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/docs/release-notes/3.0.0rc1.md) and look up old names in the
[migration guide](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/docs/migration-2-to-3.md). The
[version 2 documentation source](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v2.9.0/docs/index.md) remains available.

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
[contrastive-loss contract](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/docs/losses/contrastive_loss.md#contrastiveloss) for distances,
margins, and reduction options.

## Learn more

The [Markdown documentation](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/docs/index.md) starts with a task index and a
[concepts overview](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/docs/concepts.md) of embeddings, tuples, distances, loss
terms, and reducers. Every component page states its signature, defaults,
tensor contract, and paper, and the [recipes](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/docs/recipes/training.md) show
complete training and evaluation code.

## Contributing

See [CONTRIBUTING.md](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/CONTRIBUTING.md).
Thanks to all [contributors, upstream projects, and supporters](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v3.0.0rc1/CREDITS.md).

## Citing this library

If you use this library in your research, please cite the
[PyTorch Metric Learning paper](https://arxiv.org/abs/2008.09164):

```bibtex
@article{Musgrave2020PyTorchML,
  title={PyTorch Metric Learning},
  author={Kevin Musgrave and Serge J. Belongie and Ser-Nam Lim},
  journal={ArXiv},
  year={2020},
  volume={abs/2008.09164}
}
```
