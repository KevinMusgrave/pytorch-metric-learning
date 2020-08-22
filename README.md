<h1 align="center">
PyTorch Metric Learning
</h2>
<p align="center">
 <a href="https://badge.fury.io/py/pytorch-metric-learning">
     <img alt="PyPi version" src="https://badge.fury.io/py/pytorch-metric-learning.svg">
 </a>
 
<a href="https://pypistats.org/packages/pytorch-metric-learning">
     <img alt="PyPi stats" src="https://img.shields.io/pypi/dm/pytorch-metric-learning">
 </a>
 
</p>

</p>
 <p align="center">
<a href="https://anaconda.org/metric-learning/pytorch-metric-learning">
     <img alt="Anaconda version" src="https://img.shields.io/conda/v/metric-learning/pytorch-metric-learning?color=bright-green">
 </a>

<a href="https://anaconda.org/metric-learning/pytorch-metric-learning">
     <img alt="Anaconda downloads" src="https://img.shields.io/conda/dn/metric-learning/pytorch-metric-learning?color=bright-green">
 </a>
</p>


 <p align="center">
<a href="https://github.com/KevinMusgrave/pytorch-metric-learning/commits/master">
     <img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/KevinMusgrave/pytorch-metric-learning">
 </a>
 
<a href="https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/LICENSE">
     <img alt="License" src="https://img.shields.io/github/license/KevinMusgrave/pytorch-metric-learning?color=bright-green">
 </a>
</p>

## News

**August 7**: v0.9.90 MEGA UPDATE
* New distances module makes loss functions even more modular.
* Now compatible with half-precision
* Unfortunately also comes with numerous **breaking changes**. See the [release notes](https://github.com/KevinMusgrave/pytorch-metric-learning/releases/tag/v0.9.90) for details. 
* Hopefully this is the last major structural change to the library.

**July 25**: v0.9.89 comes with some bug fixes for CrossBatchMemory, AccuracyCalculator, BaseTester, and a new feature for InferenceModel. See the [release notes](https://github.com/KevinMusgrave/pytorch-metric-learning/releases/tag/v0.9.89) for details

**June 20**: v0.9.87 comes with some major changes that may cause your existing code to break. See the [release notes](https://github.com/KevinMusgrave/pytorch-metric-learning/releases/tag/v0.9.87) for details.

## Documentation
- [**View the documentation here**](https://kevinmusgrave.github.io/pytorch-metric-learning/)
- [**View the installation instructions here**](https://github.com/KevinMusgrave/pytorch-metric-learning#installation)

## Google Colab Examples
See the [examples folder](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/README.md) for notebooks you can download or run on Google Colab.


## Overview
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
Loss functions can be customized using [distances](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/), [reducers](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/), and [regularizers](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/).
![high_level_loss_function_overview](docs/imgs/high_level_loss_function_overview.png)

Here's an example of a customized TripletMarginLoss:
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

The TripletMarginLoss is an embedding-based or tuple-based loss. This means that internally, there is no real notion of "classes". Tuples (pairs or triplets) are formed at each iteration, based on the labels it receives. The labels don't have to represent classes. They simply need to indicate the positive and negative relationships between the embeddings. Thus, it is easy to use these loss functions for unsupervised or self-supervised learning. 

For example, the code below is a simplified version of the augmentation strategy commonly used in self-supervision. The dataset does not come with any labels. Instead, the labels are created in the training loop, solely to indicate which embeddings are positive pairs.

```python
# your training for-loop
for i, data in enumerate(dataloader):
	optimizer.zero_grad()
	embeddings = your_model(data)
	augmented = your_model(your_augmentation(data))
	labels = torch.arange(embeddings.size(0))

	embeddings = torch.cat([embeddings, augmented], dim=0)
	labels = torch.cat([labels, labels], dim=0)

	loss = loss_func(embeddings, labels)
	loss.backward()
	optimizer.step()
```


## Highlights of the rest of the library

- For a convenient way to train your model, take a look at the [trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/).
- Want to test your model's accuracy on a dataset? Try the [testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/).
- To compute the accuracy of an embedding space directly, use [AccuracyCalculator](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/).

If you're short of time and want a complete train/test workflow, check out the [example Google Colab notebooks](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples).

To learn more about all of the above, [see the documentation](https://kevinmusgrave.github.io/pytorch-metric-learning). 


## Installation
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
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-metric-learning
```

**To install with evaluation and logging capabilities (This will install the unofficial pypi version of faiss-gpu)**:
```
pip install pytorch-metric-learning[with-hooks]
```

**To install with evaluation and logging capabilities (CPU) (This will install the unofficial pypi version of faiss-cpu)**:
```
pip install pytorch-metric-learning[with-hooks-cpu]
```

### Conda
```
conda install pytorch-metric-learning -c metric-learning -c pytorch
```

**To use the testing module, you'll need faiss, which can be installed via conda as well. See the [installation instructions for faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).**



## Library contents
### [Distances](https://kevinmusgrave.github.io/pytorch-metric-learning/distances)
| Name | Reference Papers |
|---|---|
| [**CosineSimilarity**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#cosinesimilarity) |
| [**DotProductSimilarity**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#dotproductsimilarity) |
| [**LpDistance**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#lpdistance) |
| [**SNRDistance**](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/#snrdistance) | [Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

### [Losses](https://kevinmusgrave.github.io/pytorch-metric-learning/losses)
| Name | Reference Papers |
|---|---|
| [**AngularLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#angularloss) | [Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf)
| [**ArcFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#arcfaceloss) | [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf)
| [**CircleLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss) | [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/pdf/2002.10857.pdf)
| [**ContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss) | [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
| [**CosFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#cosfaceloss) | - [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf) <br/> - [Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf)
| [**FastAPLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#fastaploss) | [Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf)
| [**GeneralizedLiftedStructureLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#generalizedliftedstructureloss) | [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf)
| [**IntraPairVarianceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#intrapairvarianceloss) | [Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)
| [**LargeMarginSoftmaxLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#largemarginsoftmaxloss) | [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf)
| [**LiftedStructreLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#liftedstructureloss) | [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf)
| [**MarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#marginloss) | [Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf)
| [**MultiSimilarityLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss) | [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
| [**NCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ncaloss) | [Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf)
| [**NormalizedSoftmaxLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss) | - [NormFace: L2 Hypersphere Embedding for Face Verification](https://arxiv.org/pdf/1704.06369.pdf) <br/> - [Classification is a Strong Baseline for DeepMetric Learning](https://arxiv.org/pdf/1811.12649.pdf)
| [**NPairsLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#npairsloss) | [Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)
| [**NTXentLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss) | - [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf) <br/> - [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf) <br/> - [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
| [**ProxyAnchorLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#proxyanchorloss) | [Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/pdf/2003.13911.pdf)
| [**ProxyNCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#proxyncaloss) | [No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf)
| [**SignalToNoiseRatioContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#signaltonoiseratiocontrastiveloss) | [Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
| [**SoftTripleLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#softtripleloss) | [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)
| [**SphereFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#spherefaceloss) | [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf)
| [**TripletMarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss) | [Distance Metric Learning for Large Margin Nearest Neighbor Classification](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf)
| [**TupletMarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tupletmarginloss) | [Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf)

### [Miners](https://kevinmusgrave.github.io/pytorch-metric-learning/miners)
| Name | Reference Papers |
|---|---|
| [**AngularMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#angularminer) | 
| [**BatchHardMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batchhardminer) | [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf)
| [**DistanceWeightedMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer) | [Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf)
| [**EmbeddingsAlreadyPackagedAsTriplets**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#embeddingsalreadypackagedastriplets) | 
| [**HDCMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#hdcminer) | [Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)
| [**MaximumLossMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#maximumlossminer) | 
| [**MultiSimilarityMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#multisimilarityminer) | [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
| [**PairMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#pairmarginminer) | 
| [**TripletMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#tripletmarginminer) | [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

### [Reducers](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers)
| Name | Reference Papers |
|---|---|
| [**AvgNonZeroReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#avgnonzeroreducer)
| [**ClassWeightedReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#classweightedreducer)
| [**DivisorReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#divisorreducer)
| [**DoNothingReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#donothingreducer)
| [**MeanReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#meanreducer)
| [**ThresholdReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#thresholdreducer)

### [Regularizers](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers)
| Name | Reference Papers |
|---|---|
| [**CenterInvariantRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#centerinvariantregularizer) | [Deep Face Recognition with Center Invariant Loss](http://www1.ece.neu.edu/~yuewu/files/2017/twu024.pdf)
| [**LpRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#lpregularizer) | 
| [**RegularFaceRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#regularfaceregularizer) | [RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf)
| [**SparseCentersRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#sparsecentersregularizer) | [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf)
| [**ZeroMeanRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#zeromeanregularizer) | [Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

### [Samplers](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers)
| Name | Reference Papers |
|---|---|
| [**MPerClassSampler**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#mperclasssampler) |
| [**FixedSetOfTriplets**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#fixedsetoftriplets) |

### [Trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers)
| Name | Reference Papers |
|---|---|
| [**MetricLossOnly**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#metriclossonly)
| [**TrainWithClassifier**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#trainwithclassifier)
| [**CascadedEmbeddings**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#cascadedembeddings) | [Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf)
| [**DeepAdversarialMetricLearning**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#deepadversarialmetriclearning) | [Deep Adversarial Metric Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)
| [**UnsupervisedEmbeddingsUsingAugmentations**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#unsupervisedembeddingsusingaugmentations) |
| [**TwoStreamMetricLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#twostreammetricloss) |

### [Testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers)
| Name | Reference Papers |
|---|---|
| [**GlobalEmbeddingSpaceTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globalembeddingspacetester) |
| [**WithSameParentLabelTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#withsameparentlabeltester) |
| [**GlobalTwoStreamEmbeddingSpaceTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globaltwostreamembeddingspacetester) |

### Utils
| Name | Reference Papers |
|---|---|
| [**AccuracyCalculator**](https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation) | 
| [**HookContainer**](https://kevinmusgrave.github.io/pytorch-metric-learning/logging_presets) | 
| [**InferenceModel**](https://kevinmusgrave.github.io/pytorch-metric-learning/inference_models) |
| [**TorchInitWrapper**](https://kevinmusgrave.github.io/pytorch-metric-learning/common_functions/#torchinitwrapper) |

### Base Classes, Mixins, and Wrappers
| Name | Reference Papers |
|---|---|
| [**CrossBatchMemory**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#crossbatchmemory) | [Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf)
| [**GenericPairLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#genericpairloss) |
| [**MultipleLosses**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multiplelosses) |
| [**MultipleReducers**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#multiplereducers) |
| **EmbeddingRegularizerMixin** |
| **WeightMixin** |
| [**WeightRegularizerMixin**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#weightregularizermixin) |
| [**BaseDistance**](https://kevinmusgrave.github.io/pytorch-metric-learning/distance/#basedistance) | 
| [**BaseMetricLossFunction**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#basemetriclossfunction) | 
| [**BaseMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#baseminer) |
| [**BaseTupleMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#basetupleminer) |
| [**BaseSubsetBatchMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#basesubsetbatchminer) |
| [**BaseReducer**](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/#basereducer) |
| [**BaseRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#baseregularizer) |
| [**BaseTrainer**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#basetrainer) |
| [**BaseTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#basetester) |


## Benchmark results
See [powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker/) to view benchmark results and to use the benchmarking tool.


## Development
In order to run unit tests do:
```bash
pip install -e .[dev]
pytest tests
```
The first command may fail initially on Windows. In such a case, install `torch` by following the official
guide. Proceed to `pip install -e .[dev]` afterwards.


## Acknowledgements

### Contributors
Thanks to the contributors who made pull requests!

#### Algorithm implementations
- [AlenUbuntu](https://github.com/AlenUbuntu):
	- [CircleLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss)
- [marijnl](https://github.com/marijnl)
    - [TwoStreamMetricLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#twostreammetricloss)
    - [GlobalTwoStreamEmbeddingSpaceTester](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globaltwostreamembeddingspacetester)

#### Example notebooks
- [wconnell](https://github.com/wconnell)
	- [Learning a scRNAseq Metric Embedding](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/scRNAseq_MetricEmbedding.ipynb)
- [marijnl](https://github.com/marijnl)
    - [Example using trainers.TwoStreamMetricLoss](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TwoStreamMetricLoss.ipynb)

#### New features
- [btseytlin](https://github.com/btseytlin)
    - ```get_nearest_neighbors``` in [InferenceModel](https://kevinmusgrave.github.io/pytorch-metric-learning/inference_models)

#### General improvements and bug fixes
- [wconnell](https://github.com/wconnell)
- [marijnl](https://github.com/marijnl)
- [fralik](https://github.com/fralik)
- [JoOkuma](https://github.com/JoOkuma)

### Facebook AI
Thank you to [Ser-Nam Lim](https://research.fb.com/people/lim-ser-nam/) at [Facebook AI](https://ai.facebook.com/), and my research advisor, [Professor Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/). This project began during my internship at Facebook AI where I received valuable feedback from Ser-Nam, and his team of computer vision and machine learning engineers and research scientists. In particular, thanks to [Ashish Shah](https://www.linkedin.com/in/ashish217/) and [Austin Reiter](https://www.linkedin.com/in/austin-reiter-3962aa7/) for reviewing my code during its early stages of development.

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



## Citing this library
If you'd like to cite pytorch-metric-learning in your paper, you can use this bibtex:
```latex
@misc{Musgrave2019,
  author = {Musgrave, Kevin and Lim, Ser-Nam and Belongie, Serge},
  title = {PyTorch Metric Learning},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KevinMusgrave/pytorch-metric-learning}},
}
```
