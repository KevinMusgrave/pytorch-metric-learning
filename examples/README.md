# Examples on Google Colab
The following notebooks are meant to show entire workflows. If you want to use just a loss or miner in your own code, you can do that too. 

Before running the notebooks, make sure that the runtime type is set to "GPU", by going to the Runtime menu, and clicking on "Change runtime type"

## [Example using trainers.MetricLossOnly](https://colab.research.google.com/drive/1fwTC-GRW3X6QiJq6_abJ47On2f3s9e5e)
[This notebook](https://colab.research.google.com/drive/1fwTC-GRW3X6QiJq6_abJ47On2f3s9e5e) uses the [MetricLossOnly](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#metriclossonly) trainer, and does the following:
- initializes models, optimizers, and image transforms
- creates train/validation splits that are class-disjoint, using the CIFAR100 dataset
- initializes a loss, miner, sampler, trainer, and tester
- trains the model, records accuracy, and plots the embedding space

## [Example using trainers.TrainWithClassifier](https://colab.research.google.com/drive/1o3VeS7lnpZudoxc6HU566LUvfdrbo5nC)
[This notebook](https://colab.research.google.com/drive/1o3VeS7lnpZudoxc6HU566LUvfdrbo5nC) uses the [TrainWithClassifier](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#trainwithclassifier) trainer. It does the same thing as the MetricLossOnly notebook, but adds a classification network and a classification loss.

## [Example using trainers.CascadedEmbeddings](https://colab.research.google.com/drive/1P2Zq-sE07xvVAHihwVWQKIZ25NQoeRts)
[This notebook](https://colab.research.google.com/drive/1P2Zq-sE07xvVAHihwVWQKIZ25NQoeRts) uses the [CascadedEmbeddings](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#cascadedembeddings) trainer. The setup is more complicated in this one, because the trunk and embedder models each consist of 3 sub-networks, and the outputs are then concatenated to get the final embedding.

## [Example using trainers.DeepAdversarialMetricLearning](https://colab.research.google.com/drive/1qENr4zEoF_VfHw_2gv902ZuHZ657NGS8)
[This notebook](https://colab.research.google.com/drive/1qENr4zEoF_VfHw_2gv902ZuHZ657NGS8) uses the [DeepAdversarialMetricLearning](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#deepadversarialmetriclearning) trainer. It uses a generator to create hard negatives during training.



