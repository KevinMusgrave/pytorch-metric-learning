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
 <p align="center">
<a href="https://github.com/KevinMusgrave/pytorch-metric-learning/commits/master">
     <img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/KevinMusgrave/pytorch-metric-learning">
 </a>
 
<a href="https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/LICENSE">
     <img alt="License" src="https://img.shields.io/github/license/KevinMusgrave/pytorch-metric-learning?color=bright-green">
 </a>
</p>


## Documentation
[**View the documentation here**](https://kevinmusgrave.github.io/pytorch-metric-learning/)

## Google Colab Examples
See the [examples folder](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/README.md) for notebooks that show entire train/test workflows with logging and model saving.

## Benefits of this library
1. Ease of use
   - Add metric learning to your application with just 2 lines of code in your training loop.  
   - Mine pairs and triplets with a single function call. 
2. Flexibility
   - Mix and match losses, miners, and trainers in ways that other libraries don't allow.

## Installation
### Pip
```
pip install pytorch-metric-learning
```

**To get the latest dev version**:
```
pip install pytorch-metric-learning==0.9.87.dev4
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

## Benchmark results
See [powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker/) to view benchmark results and to use the benchmarking tool.

## Library contents
### [Losses](https://kevinmusgrave.github.io/pytorch-metric-learning/losses):
- [**AngularLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#angularloss) ([Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf))
- [**ArcFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#arcfaceloss) ([ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf))
- [**CircleLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss) ([Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/pdf/2002.10857.pdf))
- [**ContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss) ([Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf))
- [**CosFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#cosfaceloss) ([CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf))
- [**FastAPLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#fastaploss) ([Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf))
- [**GeneralizedLiftedStructureLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#generalizedliftedstructureloss) ([Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf))
- [**IntraPairVarianceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#intrapairvarianceloss) ([Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf))
- [**LargeMarginSoftmaxLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#largemarginsoftmaxloss) ([Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf))
- [**MarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#marginloss) ([Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf))
- [**MultiSimilarityLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss) ([Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- [**NCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ncaloss) ([Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf))
- [**NormalizedSoftmaxLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss) ([Classification is a Strong Baseline for DeepMetric Learning](https://arxiv.org/pdf/1811.12649.pdf))
- [**NPairsLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#npairsloss) ([Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf))
- [**NTXentLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss) ([A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709))
- [**ProxyAnchorLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#proxyanchorloss) ([Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/pdf/2003.13911.pdf))
- [**ProxyNCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#proxyncaloss) ([No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf))
- [**SignalToNoiseRatioContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#signaltonoiseratiocontrastiveloss) ([Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- [**SoftTripleLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#softtripleloss) ([SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf))
- [**SphereFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#spherefaceloss) ([SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf))
- [**TripletMarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss) ([Distance Metric Learning for Large Margin Nearest Neighbor Classification](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf))
- [**TupletMarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tupletmarginloss) ([Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf))

### [Miners](https://kevinmusgrave.github.io/pytorch-metric-learning/miners):
- [**AngularMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#angularminer)
- [**BatchHardMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batchhardminer) ([In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf))
- [**DistanceWeightedMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer) ([Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf))
- [**EmbeddingsAlreadyPackagedAsTriplets**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#embeddingsalreadypackagedastriplets)
- [**HDCMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#hdcminer) ([Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf))
- [**MaximumLossMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#maximumlossminer)
- [**MultiSimilarityMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#multisimilarityminer) ([Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- [**PairMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#pairmarginminer)
- [**TripletMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#tripletmarginminer) ([FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf))
- [**BatchEasyHardMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batcheasyhardminer) ([Improved Embeddings with Easy Positive Triplet Mining](http://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf)

### [Regularizers](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers):
- [**CenterInvariantRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#centerinvariantregularizer) ([Deep Face Recognition with Center Invariant Loss](http://www1.ece.neu.edu/~yuewu/files/2017/twu024.pdf))
- [**RegularFaceRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#regularfaceregularizer) ([RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf))

### [Samplers](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers):
- [**MPerClassSampler**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#mperclasssampler)
- [**FixedSetOfTriplets**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#fixedsetoftriplets)

### [Trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers):
- [**MetricLossOnly**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#metriclossonly)
- [**TrainWithClassifier**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#trainwithclassifier)
- [**CascadedEmbeddings**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#cascadedembeddings) ([Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf))
- [**DeepAdversarialMetricLearning**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#deepadversarialmetriclearning) ([Deep Adversarial Metric Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf))
- [**UnsupervisedEmbeddingsUsingAugmentations**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#unsupervisedembeddingsusingaugmentations)
- [**TwoStreamMetricLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#twostreammetricloss)

### [Testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers):
- [**GlobalEmbeddingSpaceTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globalembeddingspacetester)
- [**WithSameParentLabelTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#withsameparentlabeltester)
- [**GlobalTwoStreamEmbeddingSpaceTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globaltwostreamembeddingspacetester)

### [Utils](https://kevinmusgrave.github.io/pytorch-metric-learning/utils):
- [**AccuracyCalculator**](https://kevinmusgrave.github.io/pytorch-metric-learning/utils/#accuracycalculator)
- [**HookContainer**](https://kevinmusgrave.github.io/pytorch-metric-learning/utils/#hookcontainer)

### Base Classes, Mixins, and Wrappers:
- [**BaseMetricLossFunction**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#basemetriclossfunction)
- [**BaseMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#baseminer)
- [**BaseTupleMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#basetupleminer)
- [**BaseSubsetBatchMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#basesubsetbatchminer)
- [**BaseWeightRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#baseweightregularizer)
- [**BaseTrainer**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#basetrainer)
- [**BaseTester**](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#basetester)
- [**CrossBatchMemory**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#crossbatchmemory) ([Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf))
- [**GenericPairLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#genericpairloss)
- [**MultipleLosses**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multiplelosses)
- [**WeightRegularizerMixin**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#weightregularizermixin)


## Overview
Let’s try the vanilla triplet margin loss. In all examples, embeddings is assumed to be of size (N, embedding_size), and labels is of size (N).
```python
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(margin=0.1)
loss = loss_func(embeddings, labels) # in your training loop
```
Loss functions typically come with a variety of parameters. For example, with the TripletMarginLoss, you can control how many triplets per sample to use in each batch. You can also use all possible triplets within each batch:
```python
loss_func = losses.TripletMarginLoss(triplets_per_anchor="all")
```
Sometimes it can help to add a mining function:
```python
from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner(epsilon=0.1)
loss_func = losses.TripletMarginLoss(margin=0.1)
hard_pairs = miner(embeddings, labels) # in your training loop
loss = loss_func(embeddings, labels, hard_pairs)
```
In the above code, the miner finds positive and negative pairs that it thinks are particularly difficult. Note that even though the TripletMarginLoss operates on triplets, it’s still possible to pass in pairs. This is because the library automatically converts pairs to triplets and triplets to pairs, when necessary.

In general, all loss functions take in embeddings and labels, with an optional indices_tuple argument (i.e. the output of a miner):
```python
# From BaseMetricLossFunction
def forward(self, embeddings, labels, indices_tuple=None)
```
And (almost) all mining functions take in embeddings and labels:
```python
# From BaseMiner
def forward(self, embeddings, labels)
```

Here's what the above examples look like in a typical training loop:
```python
from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner(epsilon=0.1)
loss_func = losses.TripletMarginLoss(margin=0.1)

# borrowed from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    embeddings = net(inputs)
    hard_pairs = miner(embeddings, labels)
    loss = loss_func(embeddings, labels, hard_pairs)
    loss.backward()
    optimizer.step()
```
For more complex approaches, like deep adversarial metric learning, use one of the [trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers).

To check the accuracy of your model, use one of the [testers](https://kevinmusgrave.github.io/pytorch-metric-learning/testers). Which tester should you use? Almost definitely [GlobalEmbeddingSpaceTester](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/#globalembeddingspacetester), because it does what most metric-learning papers do. 

Also check out the [example Google Colab notebooks](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples/README.md).

To learn more about all of the above, [see the documentation](https://kevinmusgrave.github.io/pytorch-metric-learning). 

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

#### General improvements and bug fixes
- [wconnell](https://github.com/wconnell)
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
