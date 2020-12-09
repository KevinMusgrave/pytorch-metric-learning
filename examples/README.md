# Examples on Google Colab

Before running the notebooks, make sure that the runtime type is set to "GPU", by going to the Runtime menu, and clicking on "Change runtime type".

Click "Open in playground" on the Colab header to interact with the notebook.


## Simple examples

|Notebook|Description|Colab Link|
|:---|:---:|:---|
[MNIST using TripletMarginLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb) | Train with TripletMarginLoss, evaluate with AccuracyCalculator. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb)
[MoCo on CIFAR10](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MoCoCIFAR10.ipynb) | Self-supervision using MoCo with CrossBatchMemory |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MoCoCIFAR10.ipynb)
[Multiprocessing with DistributedDataParallel](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DistributedTripletMarginLossMNIST.ipynb) | An example using pytorch_metric_learning.utils.distributed |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DistributedTripletMarginLossMNIST.ipynb)

## Training/testing workflows with logging and model saving

The following notebooks are meant to show entire training/testing workflows. (If you want to use just a loss or miner in your own code, see the notebooks above.) They generally go through the following steps:
- initialize models, optimizers, and transforms
- creates train/validation splits
- initialize a loss, miner, sampler, trainer, and tester
- train the model, record accuracy, and plot the embedding space


|Notebook|Description|Colab Link|
|:---|:---:|:---|
[MetricLossOnly](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb) | Use just a metric loss. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb)
[A scRNAseq Metric Embedding](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/scRNAseq_MetricEmbedding.ipynb) | An example using canonical single-cell RNAseq cell types. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/scRNAseq_MetricEmbedding.ipynb)
[TwoStreamMetricLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TwoStreamMetricLoss.ipynb) | For use with two-stream datasets, where anchors and positives/negatives come from different sources. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TwoStreamMetricLoss.ipynb)
[TrainWithClassifier](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TrainWithClassifier.ipynb) | Use a metric loss + classification loss and network. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TrainWithClassifier.ipynb)
[CascadedEmbeddings](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/CascadedEmbeddings.ipynb) | Use multiple sub-networks and mine their outputs. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/CascadedEmbeddings.ipynb)
[DeepAdversarialMetricLearning](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DeepAdversarialMetricLearning.ipynb) | Use a generator to create hard negatives during training. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DeepAdversarialMetricLearning.ipynb)
[Inference](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/Inference.ipynb) | Use the inference module after you're done training. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/Inference.ipynb)
