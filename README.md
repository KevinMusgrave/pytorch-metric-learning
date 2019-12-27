# pytorch_metric_learning

## [Benchmark results (in progress)](https://drive.google.com/open?id=1Y_stkiqlHA7HTMNrhyPCnYhR0oevphRR): 
- [Train/val 50/50](https://docs.google.com/spreadsheets/d/1kiJ5rKmneQvnYKpVO9vBFdMDNx-yLcXV2wbDXlb-SB8/edit?usp=sharing)

## See [powerful_benchmarker](https://github.com/KevinMusgrave/powerful_benchmarker/) to use the benchmarking tool.

## Why use this library?
1. Flexibility
   - In this library, the various aspects of metric-learning are "dis-entangled" such that they can be mixed and matched in ways that other libraries don't allow. For example, in this library, you can use a cascaded-embeddings training method, with the multi-similarity loss, distance-weighted miner, and a classifier layer.  
2. Modularity
   - The function types are completely independent of each other, so you can import just the functions that suit your application. For instance, you might want to use a mining function by itself, to get access to the information it provides about your embedding space. 

## Installation:
```
pip install pytorch_metric_learning
```

## Currently implemented classes:
### [Loss functions](https://github.com/KevinMusgrave/pytorch_metric_learning/tree/master/pytorch_metric_learning/losses):
- AngularLoss ([paper](https://arxiv.org/pdf/1708.01682.pdf))
- ArcFaceLoss ([paper](https://arxiv.org/pdf/1801.07698.pdf))
- ContrastiveLoss
- GeneralizedLiftedStructureLoss
- MarginLoss ([paper](https://arxiv.org/pdf/1706.07567.pdf))
- MultiSimilarityLoss ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- NCALoss ([paper](https://www.cs.toronto.edu/~hinton/absps/nca.pdf))
- NPairsLoss ([paper](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf))
- ProxyNCALoss ([paper](https://arxiv.org/pdf/1703.07464.pdf))
- SignalToNoiseRatioContrastiveLoss ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- SoftTripleLoss ([paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf))
- TripletMarginLoss
- **more to be added**

### [Mining functions](https://github.com/KevinMusgrave/pytorch_metric_learning/tree/master/pytorch_metric_learning/miners):
- AngularMiner
- BatchHardMiner ([paper](https://arxiv.org/pdf/1703.07737.pdf))
- DistanceWeightedMiner ([paper](https://arxiv.org/pdf/1706.07567.pdf))
- HDCMiner ([paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf))
- MaximumLossMiner
- MultiSimilarityMiner ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf))
- PairMarginMiner
- TripletMarginMiner
- **more to be added**

### [Samplers](https://github.com/KevinMusgrave/pytorch_metric_learning/tree/master/pytorch_metric_learning/samplers):
- MPerClassSampler
- FixedSetOfTriplets
- **more to be added**

### [Training methods](https://github.com/KevinMusgrave/pytorch_metric_learning/tree/master/pytorch_metric_learning/trainers):
- MetricLossOnly
- TrainWithClassifier
- CascadedEmbeddings ([paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.pdf))
- DeepAdversarialMetricLearning ([paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf))
- UnsupervisedEmbeddingsUsingAugmentations
- **more to be added**

### [Testing methods](https://github.com/KevinMusgrave/pytorch_metric_learning/tree/master/pytorch_metric_learning/testers):
- GlobalEmbeddingSpaceTester
- WithSameParentLabelTester
- **more to be added**

## Want to add your own loss, miner, sampler, trainer, or tester? 
Please create a pull request or issue, and I will be happy to add your class to the library!

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

Train using advanced approaches, like deep adversarial metric learning:
```python
from pytorch_metric_learning import trainers

# Set up your models, optimizers, loss functions etc.
models = {"trunk": your_trunk_model, 
          "embedder": your_embedder_model,
          "generator": your_negative_generator}

optimizers = {"trunk_optimizer": your_trunk_optimizer, 
              "embedder_optimizer": your_embedder_optimizer,
              "generator_optimizer": your_negative_generator_optimizer}
              
loss_funcs = {"metric_loss": losses.AngularNPairs(alpha=35),
              "synth_loss": losses.Angular(alpha=35), 
              "g_adv_loss": losses.Angular(alpha=35)}

mining_funcs = {}

loss_weights = {"metric_loss": 1, 
                "classifier_loss": 0,
                "synth_loss": 0.1,
                "g_adv_loss": 0.1,
                "g_hard_loss": 0.1,
                "g_reg_loss": 0.1}

# Create trainer object
trainer = trainers.DeepAdversarialMetricLearning(
  models=models,
  optimizers=optimizers,
  batch_size=120,
  loss_funcs=loss_funcs,
  mining_funcs=mining_funcs,
  num_epochs=50,
  iterations_per_epoch=100,
  dataset=your_dataset,
  loss_weights=loss_weights
)
  
trainer.train()
```

## Details about losses
Every loss function extends [BaseMetricLossFunction](https://github.com/KevinMusgrave/pytorch_metric_learning/blob/master/pytorch_metric_learning/losses/base_metric_loss_function.py). The three default input arguments are:
- ```normalize_embeddings```: If True, normalizes embeddings to have Euclidean norm of 1, before computing the loss
- ```learnable_param_names```: An optional list of strings that specifies which loss parameters you want to convert to nn.Parameter (and therefore make them learnable by using a PyTorch optimizer). If not specified, then no loss parameters will be converted. 
- ```num_class_per_param```: An optional integer which specifies the size of the learnable parameters listed in learnable_param_names. If not specified, then each nn.Parameter will be of size 1.

## Details about miners
Every mining function extends either [BasePreGradientMiner](https://github.com/KevinMusgrave/pytorch_metric_learning/blob/master/pytorch_metric_learning/miners/base_miner.py#L84) or [BasePostGradientMiner](https://github.com/KevinMusgrave/pytorch_metric_learning/blob/master/pytorch_metric_learning/miners/base_miner.py#L39).

Pre-gradient miners take in a batch of embeddings, and output indices corresponding to a subset of the batch. The idea is to use these miners with torch.no_grad(), and with a large input batch size.

Post-gradient miners take in a batch of embeddings, and output a tuple of indices. If the miner outputs pairs, then the tuple is of size 4 (anchors, positives, anchors, negatives). If the miner outputs triplets, then the tuple is of size 3 (anchors, positives, negatives). These miners are typically used just before the loss is computed.

Note that in the provided training methods, you can use zero, one, or both types of miners at the same time.

What about miners that keep track of a global set of hard pairs or triplets? These should be implemented as Samplers.

## Details about samplers
Every sampler extends the standard PyTorch [Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) class that is passed into the Dataloader. Currently, the only implemented sampler is MPerClassSampler, which returns m samples per class, at every iteration.

## Details about trainers
Every trainer extends [BaseTrainer](https://github.com/KevinMusgrave/pytorch_metric_learning/blob/master/pytorch_metric_learning/trainers/base_trainer.py). The base class takes in a number of arguments:
- ```models```: a dictionary of the form: {"trunk": trunk_model, "embedder": embedder_model}
- ```optimizers```: a dictionary mapping strings to optimizers. The base class does not require any specific keys. For example, you could provide an empty dictionary, in which case no optimization will happen. Or you could provide just an optimizer for your trunk_model. But most likely, you'll want to pass in {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}.
- ```batch_size```
- ```loss_funcs```: a dictionary mapping strings to loss functions. The required keys depend on the training method, but all methods are likely to require a bare minimum of {"metric_loss": some_loss_func}
- ```mining_funcs```: a dictionary mapping strings to mining functions. Pass in an empty dictionary, or one or more of the following keys: {"pre_gradient_miner": some_mining_func_1, "post_gradient_miner": some_mining_func_2}
- ```iterations_per_epoch```: this is what actually defines what an "epoch" is. (In this library, epochs are just a measure of the number of iterations that have passed. Epochs in the traditional sense do not necessarily make sense in the context of metric learning, because it is common to sample data in a way that is not completely random.
- ```dataset```: The dataset you want to train on. Note that training methods do not perform validation, so do not pass in your validation or test set. Your dataset's ```__getitem__``` should return a dictionary. See [this class](https://github.com/KevinMusgrave/powerful_benchmarker/blob/master/datasets/cub200.py) for an example.
- ```data_device```: *Optional*. The device that you want to put batches of data on. If not specified, it will put the data on any available GPUs.
- ```loss_weights```: *Optional*. A dictionary mapping loss names to numbers. Each loss will be multiplied by the corresponding value in the dictionary. If not specified, then no loss weighting will occur.
- ```label_mapper```: *Optional*. A function that takes in a label and returns another label. For example, it might be useful to move a set of labels ranging from 100-200 to a range of 0-100, in which case you could pass in ```lambda x: x-100```. If not specified, then the original labels are used.
- ```sampler```: *Optional*. The sampler used by the dataloader. If not specified, then random sampling will be used.
- ```collate_fn```: *Optional*. The collate function used by the dataloader.
- ```record_keeper```: *Optional*. See the [record_keeper](https://github.com/KevinMusgrave/record_keeper) package.
- ```lr_schedulers```: *Optional*. A dictionary of PyTorch learning rate schedulers. Each scheduler will be stepped at the end of every epoch.
- ```gradient_clippers```: *Optional*. A dictionary of gradient clipping functions. Each function will be called before the optimizers.
- ```freeze_trunk_batchnorm```: *Optional*. If True, then the BatchNorm parameters of the trunk model will be frozen during training.
- ```label_hierarchy_level```: *Optional*. If each sample in your dataset has multiple hierarchical labels, then this can be used to select which hierarchy to use. This assumes that your labels are "2-dimensional" with shape (num_samples, num_hierarchy_levels).
- ```dataloader_num_workers```: *Optional*. For the dataloader.

## Details about testers
**The testers module requires faiss and scikit-learn. Please install these via pip or conda** 

Every tester extends [BaseTester](https://github.com/KevinMusgrave/pytorch_metric_learning/blob/master/pytorch_metric_learning/testers/base_tester.py). The arguments are:
- ```reference_set```: This specifies from which set the nearest neighbors will be retrieved. 
   - If "compared_to_self", each dataset split will refer to itself to find nearest neighbors. 
   - If "compared_to_sets_combined", each dataset split will refer to all provided splits to find nearest neighbors. 
   - If "compared_to_training_set", each dataset will refer to the training set to find nearest neighbors.
- ```normalize_embeddings```: If True, embeddings will be normalized to Euclidean norm of 1 before nearest neighbors are computed.
- ```use_trunk_output```: If True, the output of the embedder model will be ignored.
- ```batch_size```
- ```dataloader_num_workers```
- ```pca```: The number of dimensions that your embeddings will be reduced to, using PCA. The default is None, meaning PCA will not be applied.
- ```metric_for_best_epoch```: The performance metric that will be used to determine which model is best. Requires record_keeper.
- ```record_keeper```: See the [record_keeper](https://github.com/KevinMusgrave/record_keeper) package.

Which tester should you use? Almost definitely GlobalEmbeddingSpaceTester, because it does what most metric-learning papers do.

After you've initialized the tester, run ```tester.test(dataset_dict, epoch, trunk_model, embedder_model)```. 
```dataset_dict``` is a dictionary mapping from strings to datasets. If your ```reference_set = "compared_to_training_set"``` then your ```dataset_dict``` must include a key called "train".

## Acknowledgements
Thank you to Ser-Nam Lim at Facebook AI, and my research advisor, Professor Serge Belongie. This project began during my internship at Facebook AI where I received valuable feedback from Ser-Nam, and his team of computer vision and machine learning engineers and research scientists.
