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
MetricLossOnly | Use just a metric loss. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fwTC-GRW3X6QiJq6_abJ47On2f3s9e5e)
A scRNAseq Metric Embedding | An example using canonical single-cell RNAseq cell types. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DhSLDv6qXiLFKkSXKUFRjCEK1V4kPy0a)
TrainWithClassifier | Use a metric loss + classification loss and network. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o3VeS7lnpZudoxc6HU566LUvfdrbo5nC)
CascadedEmbeddings | Use multiple sub-networks and mine their outputs. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1P2Zq-sE07xvVAHihwVWQKIZ25NQoeRts)
DeepAdversarialMetricLearning | Use a generator to create hard negatives during training. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qENr4zEoF_VfHw_2gv902ZuHZ657NGS8)
TwoStreamMetricLoss | For use with two-stream datasets, where anchors and positives/negatives come from different sources. |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1moDUSeKY6teOrqSZPWUPJqjJcEGqqgKm)
