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
<a href="https://anaconda.org/metric-learning/pytorch-metric-learning">
     <img alt="Anaconda version" src="https://anaconda.org/metric-learning/pytorch-metric-learning/badges/version.svg">
 </a>
 
<a href="https://anaconda.org/metric-learning/pytorch-metric-learning">
     <img alt="Anaconda last updated" src="https://anaconda.org/metric-learning/pytorch-metric-learning/badges/latest_release_date.svg">
 </a>

<a href="https://anaconda.org/metric-learning/pytorch-metric-learning">
     <img alt="Anaconda downloads" src="https://anaconda.org/metric-learning/pytorch-metric-learning/badges/downloads.svg">
 </a>
</p>

</p>
 <p align="center">
<a href="https://github.com/KevinMusgrave/pytorch-metric-learning/commits/master">
     <img alt="Commit activity" src="https://img.shields.io/github/commit-activity/m/KevinMusgrave/pytorch-metric-learning">
 </a>
 
<a href="https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/LICENSE">
     <img alt="License" src="https://img.shields.io/github/license/KevinMusgrave/pytorch-metric-learning">
 </a>
</p>


## Documentation
[**View the documentation here**](https://kevinmusgrave.github.io/pytorch-metric-learning/)

## Benefits of this library
1. Ease of use
   - Add metric learning to your application with just 2 lines of code in your training loop.  
   - Mine pairs and triplets with a single function call. 
2. Flexibility
   - Mix and match losses, miners, and trainers in ways that other libraries don't allow.

## Installation:
Conda:
```
conda install pytorch-metric-learning -c metric-learning
```

Pip:
```
pip install pytorch-metric-learning
```

## Benchmark results
See [powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker/) to view benchmark results and to use the benchmarking tool.

## Currently implemented classes:
### [Loss functions](https://kevinmusgrave.github.io/pytorch-metric-learning/losses):
- [**AngularLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#angularloss) ([Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf))
- [**ArcFaceLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#arcfaceloss) ([ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf))
- [**ContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss)
- [**FastAPLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#fastaploss) ([Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf))
- [**GeneralizedLiftedStructureLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#generalizedliftedstructureloss)
- [**MarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#marginloss) ([Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf))
- [**MultiSimilarityLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss) ([Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- [**NCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ncaloss) ([Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf))
- [**NormalizedSoftmaxLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#normalizedsoftmaxloss) ([Classification is a Strong Baseline for DeepMetric Learning](https://arxiv.org/pdf/1811.12649.pdf))
- [**NPairsLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#npairsloss) ([Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf))
- [**ProxyNCALoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#proxyncaloss) ([No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf))
- [**SignalToNoiseRatioContrastiveLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#signaltonoiseratiocontrastiveloss) ([Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- [**SoftTripleLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#softtripleloss) ([SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf))
- [**TripletMarginLoss**](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss)
- **more to be added**

### [Mining functions](https://kevinmusgrave.github.io/pytorch-metric-learning/miners):
- [**AngularMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#angularminer)
- [**BatchHardMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#batchhardminer) ([In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf))
- [**DistanceWeightedMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer) ([Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf))
- [**HDCMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#hdcminer) ([Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf))
- [**MaximumLossMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#maximumlossminer)
- [**MultiSimilarityMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#multisimilarityminer) ([Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- [**PairMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#pairmarginminer)
- [**TripletMarginMiner**](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#tripletmarginminer)
- **more to be added**

### [Regularizers](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers):
- [**RegularFaceRegularizer**](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/#regularfaceregularizer) ([RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf))

### [Samplers](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers):
- [**MPerClassSampler**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#mperclasssampler)
- [**FixedSetOfTriplets**](https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#fixedsetoftriplets)
- **more to be added**

### [Training methods](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers):
- [**MetricLossOnly**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#metriclossonly)
- [**TrainWithClassifier**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#trainwithclassifier)
- [**CascadedEmbeddings**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#cascadedembeddings) ([Hard-Aware Deeply Cascaded Embedding](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf))
- [**DeepAdversarialMetricLearning**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#deepadversarialmetriclearning) ([Deep Adversarial Metric Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf))
- [**UnsupervisedEmbeddingsUsingAugmentations**](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/#unsupervisedembeddingsusingaugmentations)
- **more to be added**

### [Testing methods](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/pytorch-metric-learning/testers):
- [**GlobalEmbeddingSpaceTester**](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/pytorch-metric-learning/testers/global_embedding_space.py)
- [**WithSameParentLabelTester**](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/pytorch-metric-learning/testers/with_same_parent_label.py)
- **more to be added**

## Overview
Let’s try the vanilla triplet margin loss. In all examples, embeddings is assumed to be of size (N, embedding_size), and labels is of size (N).
```python
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(margin=0.1)
loss = loss_func(embeddings, labels)
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
hard_pairs = miner(embeddings, labels)
loss = loss_func(embeddings, labels, hard_pairs)
```
In the above code, the miner finds positive and negative pairs that it thinks are particularly difficult. Note that even though the TripletMarginLoss operates on triplets, it’s still possible to pass in pairs. This is because the library automatically converts pairs to triplets and triplets to pairs, when necessary.

In general, all loss functions take in embeddings and labels, with an optional indices_tuple argument (i.e. the output of a miner):
```python
# From BaseMetricLossFunction
def forward(self, embeddings, labels, indices_tuple=None)
```
And all mining functions take in embeddings and labels:
```python
# From BaseMiner
def forward(self, embeddings, labels)
```
For more complex approaches, like deep adversarial metric learning, use one of the [trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers).

To check the accuracy of your model, use one of the [testers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers). Which tester should you use? Almost definitely GlobalEmbeddingSpaceTester, because it does what most metric-learning papers do. 

Also check out the [example scripts](https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/examples).

To learn more about all of the above, [see the documentation](https://kevinmusgrave.github.io/pytorch-metric-learning). 

## Acknowledgements
Thank you to Ser-Nam Lim at Facebook AI, and my research advisor, Professor Serge Belongie. This project began during my internship at Facebook AI where I received valuable feedback from Ser-Nam, and his team of computer vision and machine learning engineers and research scientists.

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
