<h1>
<a href="https://github.com/KevinMusgrave/pytorch-metric-learning">
<img alt="PyTorch Metric Learning" src="https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/docs/imgs/Logo2.png">
</a>
</h1>

<p align="center">
 <a href="https://pypi.org/project/pytorch-metric-learning">
     <img alt="PyPi version" src="https://img.shields.io/pypi/v/pytorch-metric-learning?color=bright-green">
 </a>
	
	
 
 <a href="https://anaconda.org/conda-forge/pytorch-metric-learning">
     <img alt="Anaconda version" src="https://img.shields.io/conda/v/conda-forge/pytorch-metric-learning?color=bright-green">
 </a>
</p>

## News

**July 24**: v2.6.0
- Changed the `emb` argument of `DistributedLossWrapper.forward` to `embeddings` to be consistent with the rest of the library.
- Added a warning and early-return when `DistributedLossWrapper` is being used in a non-distributed setting.
- Thank you [elisim](https://github.com/elisim).

**April 1**: v2.5.0
- Improved `get_all_triplets_indices` so that large batch sizes don't trigger the `INT_MAX` error.
- See the [release notes](https://github.com/KevinMusgrave/pytorch-metric-learning/releases/tag/v2.5.0).
- Thank you [mkmenta](https://github.com/mkmenta).

## Documentation
- [**View the documentation here**](https://kevinmusgrave.github.io/pytorch-metric-learning/)
- [**View the installation instructions here**](https://github.com/KevinMusgrave/pytorch-metric-learning#installation)
- [**View the available losses, miners etc. here**](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/CONTENTS.md) 


## Google Colab Examples
See the [examples folder](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/README.md) for notebooks you can download or run on Google Colab.


## PyTorch Metric Learning Overview
This library contains 9 modules, each of which can be used independently within your existing codebase, or combined together for a complete train/test workflow.

![high_level_module_overview](docs/imgs/high_level_module_overview.png)



## How loss functions work

### Using losses and miners in your training loop
Let’s initialize a plain [TripletMarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss):
```python
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss()
```

To compute the loss in your training loop, pass in the embeddings computed by your model, and the corresponding labels. The embeddings should have size (N, embedding_size), and the labels should have size (N), where N is the batch size.

```python
# your training loop
for i, (data, labels) in enumerate(dataloader):
	optimizer.zero_grad()
	embeddings = model(data)
	loss = loss_func(embeddings, labels)
	loss.backward()
	optimizer.step()
```

The TripletMarginLoss computes all possible triplets within the batch, based on the labels you pass into it. Anchor-positive pairs are formed by embeddings that share the same label, and anchor-negative pairs are formed by embeddings that have different labels. 

Sometimes it can help to add a mining function:
```python
from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner()
loss_func = losses.TripletMarginLoss()

# your training loop
for i, (data, labels) in enumerate(dataloader):
	optimizer.zero_grad()
	embeddings = model(data)
	hard_pairs = miner(embeddings, labels)
	loss = loss_func(embeddings, labels, hard_pairs)
	loss.backward()
	optimizer.step()
```
In the above code, the miner finds positive and negative pairs that it thinks are particularly difficult. Note that even though the TripletMarginLoss operates on triplets, it’s still possible to pass in pairs. This is because the library automatically converts pairs to triplets and triplets to pairs, when necessary.

### Customizing loss functions
Loss functions can be customized using [distances](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/), [reducers](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/), and [regularizers](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/). In the diagram below, a miner finds the indices of hard pairs within a batch. These are used to index into the distance matrix, computed by the distance object. For this diagram, the loss function is pair-based, so it computes a loss per pair. In addition, a regularizer has been supplied, so a regularization loss is computed for each embedding in the batch. The per-pair and per-element losses are passed to the reducer, which (in this diagram) only keeps losses with a high value. The averages are computed for the high-valued pair and element losses, and are then added together to obtain the final loss.

![high_level_loss_function_overview](docs/imgs/high_level_loss_function_overview.png)

Now here's an example of a customized TripletMarginLoss:
```python
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(distance = CosineSimilarity(), 
				     reducer = ThresholdReducer(high=0.3), 
			 	     embedding_regularizer = LpRegularizer())
```
This customized triplet loss has the following properties:

 - The loss will be computed using cosine similarity instead of Euclidean distance.
 - All triplet losses that are higher than 0.3 will be discarded.
 - The embeddings will be L2 regularized.  

### Using loss functions for unsupervised / self-supervised learning

A `SelfSupervisedLoss` wrapper is provided for self-supervised learning:

```python
from pytorch_metric_learning.losses import SelfSupervisedLoss
loss_func = SelfSupervisedLoss(TripletMarginLoss())

# your training for-loop
for i, data in enumerate(dataloader):
	optimizer.zero_grad()
	embeddings = your_model(data)
	augmented = your_model(your_augmentation(data))
	loss = loss_func(embeddings, augmented)
	loss.backward()
	optimizer.step()
```

If you're interested in [MoCo](https://arxiv.org/pdf/1911.05722.pdf)-style self-supervision, take a look at the [MoCo on CIFAR10](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples#simple-examples) notebook. It uses CrossBatchMemory to implement the momentum encoder queue, which means you can use any tuple loss, and any tuple miner to extract hard samples from the queue.


## Highlights of the rest of the library

- For a convenient way to train your model, take a look at the [trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/).
- Want to test your model's accuracy on a dataset? Try the [testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/).
- To compute the accuracy of an embedding space directly, use [AccuracyCalculator](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/).

If you're short of time and want a complete train/test workflow, check out the [example Google Colab notebooks](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples).

To learn more about all of the above, [see the documentation](https://kevinmusgrave.github.io/pytorch-metric-learning). 


## Installation

### Required PyTorch version
 - ```pytorch-metric-learning >= v0.9.90``` requires ```torch >= 1.6```
 - ```pytorch-metric-learning < v0.9.90``` doesn't have a version requirement, but was tested with ```torch >= 1.2```

Other dependencies: ```numpy, scikit-learn, tqdm, torchvision```

### Pip
```
pip install pytorch-metric-learning
```

**To get the latest dev version**:
```
pip install pytorch-metric-learning --pre
```

**To install on Windows**:
```
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-metric-learning
```

**To install with evaluation and logging capabilities**

(This will install the unofficial pypi version of faiss-gpu, plus record-keeper and tensorboard):
```
pip install pytorch-metric-learning[with-hooks]
```

**To install with evaluation and logging capabilities (CPU)**

(This will install the unofficial pypi version of faiss-cpu, plus record-keeper and tensorboard):
```
pip install pytorch-metric-learning[with-hooks-cpu]
```
	
### Conda
```
conda install -c conda-forge pytorch-metric-learning
```

**To use the testing module, you'll need faiss, which can be installed via conda as well. See the [installation instructions for faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).**

</details>
	


## Benchmark results
See [powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker/) to view benchmark results and to use the benchmarking tool.


## Development
Development is done on the ```dev``` branch:
```
git checkout dev
```

Unit tests can be run with the default ```unittest``` library:
```bash
python -m unittest discover
```

You can specify the test datatypes and test device as environment variables. For example, to test using float32 and float64 on the CPU:
```bash
TEST_DTYPES=float32,float64 TEST_DEVICE=cpu python -m unittest discover
```

To run a single test file instead of the entire test suite, specify the file name:
```bash
python -m unittest tests/losses/test_angular_loss.py
```

Code is formatted using ```black``` and ```isort```:
```bash
pip install black isort
./format_code.sh
```


## Acknowledgements

### Contributors
Thanks to the contributors who made pull requests!

| Contributor | Highlights |
| -- | -- |
|[domenicoMuscill0](https://github.com/domenicoMuscill0)| - [ManifoldLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#manifoldloss) <br/> - [P2SGradLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#p2sgradloss) <br/> - [HistogramLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#histogramloss) <br/> - [DynamicSoftMarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#dynamicsoftmarginloss) <br/> - [RankedListLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#rankedlistloss) |
|[mlopezantequera](https://github.com/mlopezantequera) | - Made the [testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers) work on any combination of query and reference sets <br/> - Made [AccuracyCalculator](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/) work with arbitrary label comparisons |
|[cwkeam](https://github.com/cwkeam) | - [SelfSupervisedLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#selfsupervisedloss) <br/> - [VICRegLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#vicregloss) <br/> - Added mean reciprocal rank accuracy to [AccuracyCalculator](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/) <br/> - BaseLossWrapper|
|[marijnl](https://github.com/marijnl)| - [BatchEasyHardMiner](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batcheasyhardminer) <br/> - [TwoStreamMetricLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#twostreammetricloss) <br/> - [GlobalTwoStreamEmbeddingSpaceTester](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globaltwostreamembeddingspacetester) <br/> - [Example using trainers.TwoStreamMetricLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TwoStreamMetricLoss.ipynb) |
| [chingisooinar](https://github.com/chingisooinar) | [SubCenterArcFaceLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#subcenterarcfaceloss) |
| [elias-ramzi](https://github.com/elias-ramzi) | [HierarchicalSampler](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#hierarchicalsampler) |
| [fjsj](https://github.com/fjsj) | [SupConLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss) |
| [AlenUbuntu](https://github.com/AlenUbuntu) | [CircleLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss) |
| [interestingzhuo](https://github.com/interestingzhuo) | [PNPLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#pnploss) |
| [wconnell](https://github.com/wconnell) | [Learning a scRNAseq Metric Embedding](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/scRNAseq_MetricEmbedding.ipynb) |
| [mkmenta](https://github.com/mkmenta) | Improved `get_all_triplets_indices` (fixed the `INT_MAX` error) |
| [AlexSchuy](https://github.com/AlexSchuy) | optimized ```utils.loss_and_miner_utils.get_random_triplet_indices``` |
| [JohnGiorgi](https://github.com/JohnGiorgi) | ```all_gather``` in [utils.distributed](https://kevinmusgrave.github.io/pytorch-metric-learning/distributed) |
| [Hummer12007](https://github.com/Hummer12007) | ```utils.key_checker``` |
| [vltanh](https://github.com/vltanh) | Made ```InferenceModel.train_indexer``` accept datasets |
| [btseytlin](https://github.com/btseytlin) | ```get_nearest_neighbors``` in [InferenceModel](https://kevinmusgrave.github.io/pytorch-metric-learning/inference_models) |
| [mlw214](https://github.com/mlw214) | Added ```return_per_class``` to [AccuracyCalculator](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/) |
| [layumi](https://github.com/layumi) | [InstanceLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#instanceloss) |
| [NoTody](https://github.com/NoTody) | Helped add `ref_emb` and `ref_labels` to the distributed wrappers. |
| [ElisonSherton](https://github.com/ElisonSherton) | Fixed an edge case in ArcFaceLoss. |
| [stompsjo](https://github.com/stompsjo) | Improved documentation for NTXentLoss. |
| [Puzer](https://github.com/Puzer) | Bug fix for PNPLoss. |
| [elisim](https://github.com/elisim) | Developer improvements to DistributedLossWrapper. |
| [GaetanLepage](https://github.com/GaetanLepage) | |
| [z1w](https://github.com/z1w) | |
| [thinline72](https://github.com/thinline72) | |
| [tpanum](https://github.com/tpanum) | |
| [fralik](https://github.com/fralik) | |
| [joaqo](https://github.com/joaqo) | |
| [JoOkuma](https://github.com/JoOkuma) | |
| [gkouros](https://github.com/gkouros) | |
| [yutanakamura-tky](https://github.com/yutanakamura-tky) | |
| [KinglittleQ](https://github.com/KinglittleQ) | |
| [martin0258](https://github.com/martin0258) | |
| [michaeldeyzel](https://github.com/michaeldeyzel) | |
| [HSinger04](https://github.com/HSinger04) | |
| [rheum](https://github.com/rheum) | |
| [bot66](https://github.com/bot66) | |



### Facebook AI
Thank you to [Ser-Nam Lim](https://sites.google.com/site/sernam) at [Facebook AI](https://ai.facebook.com/), and my research advisor, [Professor Serge Belongie](https://www.belongielab.org/). This project began during my internship at Facebook AI where I received valuable feedback from Ser-Nam, and his team of computer vision and machine learning engineers and research scientists. In particular, thanks to [Ashish Shah](https://www.linkedin.com/in/ashish217/) and [Austin Reiter](https://www.linkedin.com/in/austin-reiter-3962aa7/) for reviewing my code during its early stages of development.

### Open-source repos
This library contains code that has been adapted and modified from the following great open-source repos:
- https://github.com/bnu-wangxun/Deep_Metric
- https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning
- https://github.com/facebookresearch/deepcluster
- https://github.com/geonm/proxy-anchor-loss
- https://github.com/idstcv/SoftTriple
- https://github.com/kunhe/FastAP-metric-learning
- https://github.com/ronekko/deep_metric_learning
- https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
- http://kaizhao.net/regularface
- https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts

### Logo
Thanks to [Jeff Musgrave](https://www.designgenius.ca/) for designing the logo.

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
