# Examples on Google Colab
The following notebooks are meant to show entire workflows. If you want to use just a loss or miner in your own code, you can do that too. 

Before running the notebooks, make sure to change the runtime type to "GPU", by going to the Runtime menu, and clicking on "Change runtime type"

## [Example using trainers.MetricLossOnly](https://colab.research.google.com/drive/1fwTC-GRW3X6QiJq6_abJ47On2f3s9e5e)
View [this Google Colab notebook](https://colab.research.google.com/drive/1fwTC-GRW3X6QiJq6_abJ47On2f3s9e5e) to see an example that:
- initializes models, optimizers, and image transforms
- creates train/validation splits that are class-disjoint, using the CIFAR100 dataset
- initializes a loss, miner, sampler, trainer, and tester
- trains the model, records accuracy, and plots the embedding space

## [Example using trainers.TrainWithClassifier](https://colab.research.google.com/drive/1o3VeS7lnpZudoxc6HU566LUvfdrbo5nC)
This is the same as the ```trainers.MetricLossOnly``` notebook, but it appends a classification network, and adds a classification loss.
