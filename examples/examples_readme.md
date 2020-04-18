# Examples on Google Colab
The following notebooks are meant to show entire workflows. If you just want to use a loss or miner in your own code, you can do that too.

## Example using trainers.MetricLossOnly
View [this Google Colab notebook](https://colab.research.google.com/drive/1fwTC-GRW3X6QiJq6_abJ47On2f3s9e5e) to see an example that does the following:
- initializes models, optimizers, and image transforms
- create train/validation splits that are class-disjoint, using the CIFAR100 dataset
- initializes a loss, miner, sampler, trainer, and tester
- trains the model, records accuracy, and plots the embedding space

