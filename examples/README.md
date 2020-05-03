# Examples on Google Colab
The following notebooks are meant to show entire training/testing workflows. (If you want to use just a loss or miner in your own code, you can do that too.) They generally go through the following steps:
- initialize models, optimizers, and transforms
- creates train/validation splits
- initialize a loss, miner, sampler, trainer, and tester
- train the model, record accuracy, and plot the embedding space

Before running the notebooks, make sure that the runtime type is set to "GPU", by going to the Runtime menu, and clicking on "Change runtime type".

Click "Open in playground" on the Colab header to interact with the notebook.

|Notebook|Description|Colab Link|
|:---|:---:|:---|
[MetricLossOnly](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb) | Use just a metric loss. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb)
[A scRNAseq Metric Embedding](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/scRNAseq_MetricEmbedding.ipynb) | An example using canonical single-cell RNAseq cell types. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/scRNAseq_MetricEmbedding.ipynb)
TwoStreamMetricLoss | For use with two-stream datasets, where anchors and positives/negatives come from different sources. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1moDUSeKY6teOrqSZPWUPJqjJcEGqqgKm)
[TrainWithClassifier](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TrainWithClassifier.ipynb) | Use a metric loss + classification loss and network. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TrainWithClassifier.ipynb)
[CascadedEmbeddings](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/CascadedEmbeddings.ipynb) | Use multiple sub-networks and mine their outputs. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/CascadedEmbeddings.ipynb)
[DeepAdversarialMetricLearning](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DeepAdversarialMetricLearning.ipynb) | Use a generator to create hard negatives during training. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DeepAdversarialMetricLearning.ipynb)
