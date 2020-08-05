# Losses
All loss functions are used as follows:

```python
from pytorch_metric_learning import losses
loss_func = losses.SomeLoss()
loss = loss_func(embeddings, labels) # in your training for-loop
```

Or if you are using a loss in conjunction with a [miner](miners.md):

```python
from pytorch_metric_learning import miners, losses
miner_func = miners.SomeMiner()
loss_func = losses.SomeLoss()
miner_output = miner_func(embeddings, labels) # in your training for-loop
loss = loss_func(embeddings, labels, miner_output)
```

You can also specify how losses get reduced to a single value by using a [reducer](reducers.md):
```python
from pytorch_metric_learning import losses, reducers
reducer = reducers.SomeReducer()
loss_func = losses.SomeLoss(reducer=reducer)
loss = loss_func(embeddings, labels) # in your training for-loop
```


## AngularLoss 
[Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf){target=_blank}
```python
losses.AngularLoss(alpha, **kwargs)
```
**Equation**:

![angular_loss_equation](imgs/angular_loss_equation.png){: style="height:200px"}


**Parameters**:

* **alpha**: The angle specified in degrees. The paper uses values between 36 and 55.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss for every ```a1```, where ```(a1,p)``` represents every positive pair in the batch. Reduction type is ```"element"```.


## ArcFaceLoss 
[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf){target=_blank}

```python
losses.ArcFaceLoss(margin, num_classes, embedding_size, scale=64, **kwargs)
```

**Equation**:

![arcface_loss_equation](imgs/arcface_loss_equation.png){: style="height:80px"}


**Parameters**:

* **margin**: The angular margin penalty in degrees. In the above equation, ```m = radians(margin)```. The paper uses 0.5 radians, which is 28.6 degrees.
* **num_classes**: The number of classes in your training dataset.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **scale**: This is ```s``` in the above equation. The paper uses 64.

**Other info**: 

* This also extends [WeightRegularizerMixin](losses.md#weightregularizermixin), so it accepts a ```regularizer``` and ```reg_weight``` as optional init arguments.
* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.ArcFaceLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.



## BaseMetricLossFunction
All loss functions extend this class and therefore inherit its ```__init__``` parameters.

```python
losses.BaseMetricLossFunction(normalize_embeddings=True, reducer=None)
```

**Parameters**:

* **normalize_embeddings**: If True, embeddings will be normalized to have a Euclidean norm of 1 before the loss is computed.
* **reducer**: A [reducer](reducers.md) object. If None, then the default reducer will be used.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Required Implementations**:
```python
def compute_loss(self, embeddings, labels, indices_tuple=None):
    raise NotImplementedError
```


## CircleLoss 
[Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/pdf/2002.10857.pdf){target=_blank}

```python
losses.CircleLoss(m=0.4, gamma=80, **kwargs)
```

**Equations**:

![circle_loss_equation1](imgs/circle_loss_equation1.png){: style="height:150px"}

where

![circle_loss_equation2](imgs/circle_loss_equation2.png){: style="height:70px"}

![circle_loss_equation7](imgs/circle_loss_equation7.png){: style="height:25px"}

![circle_loss_equation8](imgs/circle_loss_equation8.png){: style="height:25px"}

![circle_loss_equation3](imgs/circle_loss_equation3.png){: style="height:25px"}

![circle_loss_equation4](imgs/circle_loss_equation4.png){: style="height:25px"}

![circle_loss_equation5](imgs/circle_loss_equation5.png){: style="height:25px"}

![circle_loss_equation6](imgs/circle_loss_equation6.png){: style="height:25px"}



**Parameters**:

* **m**: The relaxation factor that controls the radius of the decision boundary. The paper uses 0.25 for face recognition, and 0.4 for fine-grained image retrieval (images of birds, cars, and online products).
* **gamma**: The scale factor that determines the largest scale of each similarity score. The paper uses 256 for face recognition, and 80 for fine-grained image retrieval.

**Default reducer**: [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.

## ContrastiveLoss
```python
losses.ContrastiveLoss(pos_margin=0, 
					neg_margin=1, 
					use_similarity=False, 
					power=1,
					**kwargs):
```

**Equation**:

![contrastive_loss_equation](imgs/contrastive_loss_equation.png){: style="height:30px"}

**Parameters**:

* **pos_margin**: The distance (or similarity) over (under) which positive pairs will contribute to the loss.
* **neg_margin**: The distance (or similarity) under (over) which negative pairs will contribute to the loss.  
* **use_similarity**: If True, will use dot product between vectors instead of euclidean distance.
* **power**: Each pair's loss will be raised to this power.

Note that the default values for ```pos_margin``` and ```neg_margin``` are suitable if ```use_similarity = False```. If you set ```use_similarity = True```, then more appropriate values would be ```pos_margin = 1``` and ```neg_margin = 0```.

**Default reducer**: [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

* **pos_loss**: The loss per positive pair in the batch. Reduction type is ```"pos_pair"```.
* **neg_loss**: The loss per negative pair in the batch. Reduction type is ```"neg_pair"```.


## CosFaceLoss 
[CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf){target=_blank}

```python
losses.CosFaceLoss(margin, num_classes, embedding_size, scale=64, **kwargs)
```

**Equation**:

![cosface_loss_equation](imgs/cosface_loss_equation.png){: style="height:80px"}

**Parameters**:

* **margin**: The cosine margin penalty (m in the above equation). The paper used values between 0.25 and 0.45.
* **num_classes**: The number of classes in your training dataset.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **scale**: This is ```s``` in the above equation. The paper uses 64.

**Other info**: 

* This also extends [WeightRegularizerMixin](losses.md#weightregularizermixin), so it accepts a ```regularizer``` and ```reg_weight``` as optional init arguments.
* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.CosFaceLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.


## CrossBatchMemory 
This wraps a loss function, and implements [Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf){target=_blank}. It stores embeddings from previous iterations in a queue, and uses them to form more pairs/triplets with the current iteration's embeddings.

```python
losses.CrossBatchMemory(loss, embedding_size, memory_size=1024, miner=None)
```

**Parameters**:

* **loss**: The loss function to be wrapped. For example, you could pass in ```ContrastiveLoss()```.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **memory_size**: The size of the memory queue.
* **miner**: An optional [tuple miner](miners.md), which will be used to mine pairs/triplets from the memory queue.


## FastAPLoss
[Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf){target=_blank}

```python
losses.FastAPLoss(num_bins, **kwargs)
```

**Parameters**:

* **num_bins**: The number of soft histogram bins for calculating average precision. The paper suggests using 10.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element that has at least 1 positive in the batch. Reduction type is ```"element"```.


## GenericPairLoss
```python
losses.GenericPairLoss(use_similarity, mat_based_loss, squared_distances=False, **kwargs)
```

**Parameters**:

* **use_similarity**: Set to True if the loss function uses pairwise similarity (dot product of each embedding pair). Otherwise, euclidean distance will be used.
* **mat_based_loss**: See required implementations.
* **squared_distances**: If True, then the euclidean distance will be squared.

**Required Implementations**:
```python
# If mat_based_loss is True, then this takes in mat, pos_mask, neg_mask
# If False, this takes in pos_pair, neg_pair, indices_tuple
def _compute_loss(self):
    raise NotImplementedError
```

## GeneralizedLiftedStructureLoss
This was presented in [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf){target=_blank}. It is a modification of the original [LiftedStructureLoss](losses.md#liftedstructureloss)

```python
losses.GeneralizedLiftedStructureLoss(neg_margin, pos_margin=0, **kwargs)
```
**Equation**:

![generalized_lifted_structure_loss_equation](imgs/generalized_lifted_structure_loss_equation.png){: style="height:250px"}

**Parameters**:

* **pos_margin**: The margin in the expression ```e^(D - margin)```. The paper uses ```pos_margin = 0 ```, which is why this margin does not appear in the above equation.
* **neg_margin**: This is ```m``` in the above equation. The paper used values between 0.1 and 1.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.

## IntraPairVarianceLoss
[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf){target=_blank}
```python
losses.IntraPairVarianceLoss(pos_eps=0.01, neg_eps=0.01, **kwargs)
```

**Equations**:

![intra_pair_variance_loss_equation1](imgs/intra_pair_variance_loss_equation1.png){: style="height:39px"}

![intra_pair_variance_loss_equation2](imgs/intra_pair_variance_loss_equation2.png){: style="height:34px"}

**Parameters**:

* **pos_eps**: The epsilon in the L<sub>pos</sub> equation. The paper uses 0.01.
* **neg_eps**: The epsilon in the L<sub>neg</sub> equation. The paper uses 0.01.

You should probably use this in conjunction with another loss, as described in the paper. You can accomplish this by using [MultipleLosses](losses.md#multiplelosses):
```python
main_loss = losses.TupletMarginLoss(margin=5)
var_loss = losses.IntraPairVarianceLoss()
complete_loss = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **pos_loss**: The loss per positive pair in the batch. Reduction type is ```"pos_pair"```.
* **neg_loss**: The loss per negative pair in the batch. Reduction type is ```"neg_pair"```.



## LargeMarginSoftmaxLoss
[Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf){target=_blank}

```python
losses.LargeMarginSoftmaxLoss(margin, num_classes, embedding_size, scale=1, normalize_weights=False, **kwargs)
```

**Equations**:

![large_margin_softmax_loss_equation1](imgs/large_margin_softmax_loss_equation1.png){: style="height:80px"}

where

![large_margin_softmax_loss_equation2](imgs/large_margin_softmax_loss_equation2.png){: style="height:90px"}

**Parameters**:

* **margin**: An integer which dictates the size of the angular margin. This is ```m``` in the above equation. The paper finds ```m=4``` works best.
* **num_classes**: The number of classes in your training dataset.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **scale**: The exponent multiplier in the loss's softmax expression. The paper uses ```scale = 1 ```, which is why it does not appear in the above equation.
* **normalize_weights**: If True, the learned weights will be normalized to have Euclidean norm of 1, before the loss is computed. Note that when this parameter is True, it becomes equivalent to [SphereFaceLoss](losses.md#spherefaceloss).

**Other info**: 

* This also extends [WeightRegularizerMixin](losses.md#weightregularizermixin), so it accepts a ```regularizer``` and ```reg_weight``` as optional init arguments.
* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.LargeMarginSoftmaxLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.


## LiftedStructureLoss
The original lifted structure loss as presented in [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf){target=_blank}

```python
losses.LiftedStructureLoss(neg_margin, pos_margin=0, **kwargs):
```

**Equation**:

![lifted_structure_loss_equation](imgs/lifted_structure_loss_equation.png){: style="height:150px"}

**Parameters**:

* **pos_margin**: The margin in the expression ```D_(i,j) - margin```. The paper uses ```pos_margin = 0 ```, which is why it does not appear in the above equation.
* **neg_margin**: This is ```alpha``` in the above equation. The paper uses 1.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per positive pair in the batch. Reduction type is ```"pos_pair"```.


## MarginLoss
[Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf){target=_blank}
```python
losses.MarginLoss(margin, nu, beta, triplets_per_anchor="all", learn_beta=False, num_classes=None, **kwargs)
```

**Equations**:

![margin_loss_equation2](imgs/margin_loss_equation2.png){: style="height:60px"}

where

![margin_loss_equation1](imgs/margin_loss_equation1.png){: style="height:40px"}


**Parameters**:

* **margin**: This is alpha in the above equation. The paper uses 0.2.
* **nu**: The regularization weight for the magnitude of beta.
* **beta**: This is beta in the above equation. The paper uses 1.2 as the initial value.
* **triplets_per_anchor**: The number of triplets per element to sample within a batch. Can be an integer or the string "all". For example, if your batch size is 128, and triplets_per_anchor is 100, then 12800 triplets will be sampled. If triplets_per_anchor is "all", then all possible triplets in the batch will be used.
* **learn_beta**: If True, beta will be a torch.nn.Parameter, which can be optimized using any PyTorch optimizer.
* **num_classes**: If not None, then beta will be of size ```num_classes```, so that a separate beta is used for each class during training.

**Default reducer**: [DivisorReducer](reducers.md#divisorreducer)

**Reducer input**:

* **margin_loss**: The loss per triplet in the batch. Reduction type is ```"triplet"```.
* **beta_reg_loss**: The regularization loss per element in ```self.beta```. Reduction type is ```"already_reduced"``` if ```self.num_classes = None```. Otherwise it is ```"element"```.


## MultipleLosses
This is a simple wrapper for multiple losses. Pass in a list of already-initialized loss functions. Then, when you call forward on this object, it will return the sum of all wrapped losses.
```python
losses.MultipleLosses(losses, weights=None)
```
**Parameters**:

* **losses**: A list or dictionary of initialized loss functions. On the forward call of MultipleLosses, each wrapped loss will be computed, and then the average will be returned.
* **weights**: Optional. A list or dictionary of loss weights, which will be multiplied by the corresponding losses obtained by the loss functions. The default is to multiply each loss by 1. If ```losses``` is a list, then ```weights``` must be a list. If ```losses``` is a dictionary, ```weights``` must contain the same keys as ```losses```. 

## MultiSimilarityLoss
[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}
```python
losses.MultiSimilarityLoss(alpha, beta, base=0.5, **kwargs)
```

**Equation**:

![multi_similarity_loss_equation](imgs/multi_similarity_loss_equation.png){: style="height:150px"}


**Parameters**:

* **alpha**: The weight applied to positive pairs. The paper uses 2.
* **beta**: The weight applied to negative pairs. The paper uses 50.
* **base**: The offset applied to the exponent in the loss. This is lambda in the above equation. The paper uses 1. 

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.

## NCALoss
[Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf){target=_blank}
```python
losses.NCALoss(softmax_scale=1, **kwargs)
```

**Equations**:

![nca_loss_equation1](imgs/nca_loss_equation1.png){: style="height:50px"}

where

![nca_loss_equation2](imgs/nca_loss_equation2.png){: style="height:60px"}

![nca_loss_equation3](imgs/nca_loss_equation3.png){: style="height:60px"}

In this implementation, we use ```-g(A)``` as the loss.

**Parameters**:

* **softmax_scale**: The exponent multiplier in the loss's softmax expression. The paper uses ```softmax_scale = 1 ```, which is why it does not appear in the above equations.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch, that results in a non zero exponent in the cross entropy expression. Reduction type is ```"element"```.



## NormalizedSoftmaxLoss
[Classification is a Strong Baseline for Deep Metric Learning](https://arxiv.org/pdf/1811.12649.pdf){target=_blank}
```python
losses.NormalizedSoftmaxLoss(temperature, embedding_size, num_classes, **kwargs)
```

**Equation**:

![normalized_softmax_loss_equation](imgs/normalized_softmax_loss_equation.png){: style="height:80px"}


**Parameters**:

* **temperature**: This is sigma in the above equation. The paper uses 0.05.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **num_classes**: The number of classes in your training dataset.

**Other info**

* This also extends [WeightRegularizerMixin](losses.md#weightregularizermixin), so it accepts a ```regularizer``` and ```reg_weight``` as optional init arguments.
* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.NormalizedSoftmaxLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.


## NPairsLoss
[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf){target=_blank}
```python
losses.NPairsLoss(l2_reg_weight=0, **kwargs)
```

**Parameters**:

* **l2_reg_weight**: The regularization weight for the L2 norm of the embeddings.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.
* **l2_reg**: The L2 regularization loss per element in the batch. This key exists only if ```self.l2_reg_weight > 0```. Reduction type is ```"element"```.


## NTXentLoss
This is also known as InfoNCE, and is a generalization of the [NPairsLoss](losses.md#npairsloss). It has been used in self-supervision papers such as: 

 - [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf){target=_blank}
 - [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf){target=_blank}
 - [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf){target=_blank}
```python
losses.NTXentLoss(temperature, **kwargs)
```

**Equation**:

![ntxent_loss_equation](imgs/ntxent_loss_equation.png){: style="height:70px"}

**Parameters**:

* **temperature**: This is tau in the above equation. The MoCo paper uses 0.07, while SimCLR uses 0.5.

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per positive pair in the batch. Reduction type is ```"pos_pair"```.

## ProxyAnchorLoss
[Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/pdf/2003.13911.pdf){target=_blank}
```python
losses.ProxyAnchorLoss(num_classes, embedding_size, margin = 0.1, alpha = 32, **kwargs)
```

**Equation**:

![proxy_anchor_loss_equation](imgs/proxy_anchor_loss_equation.png){: style="height:150px"}


**Parameters**:

* **num_classes**: The number of classes in your training dataset.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **margin**: This is delta in the above equation. The paper uses 0.1.
* **alpha**: This is alpha in the above equation. The paper uses 32.

**Other info**

* This also extends [WeightRegularizerMixin](losses.md#weightregularizermixin), so it accepts a ```regularizer``` and ```reg_weight``` as optional init arguments.
* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.ProxyAnchorLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [DivisorReducer](reducers.md#divisorreducer)

**Reducer input**:

* **pos_loss**: The positive pair loss per proxy. Reduction type is ```"element"```.
* **neg_loss**: The negative pair loss per proxy. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.


## ProxyNCALoss
[No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf){target=_blank}
```python
losses.ProxyNCALoss(num_classes, embedding_size, **kwargs)
```

**Parameters**:

* **num_classes**: The number of classes in your training dataset.
* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **softmax_scale**: See [NCALoss](losses.md#ncaloss)

**Other info**

* This also extends [WeightRegularizerMixin](losses.md#weightregularizermixin), so it accepts a ```regularizer``` and ```reg_weight``` as optional init arguments.
* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.ProxyNCALoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch, that results in a non zero exponent in the cross entropy expression. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.


## SignalToNoiseRatioContrastiveLoss
[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}
```python
losses.SignalToNoiseRatioContrastiveLoss(pos_margin, 
										neg_margin, 
										regularizer_weight, 
										**kwargs)
```

**Parameters**:

* **pos_margin**: The noise-to-signal ratio over which positive pairs will contribute to the loss.
* **neg_margin**: The noise-to-signal ratio under which negative pairs will contribute to the loss.
* **regularizer_weight**: The regularizer encourages the embeddings to have zero-mean distributions. 

**Default reducer**: 

* [MeanReducer](reducers.md#meanreducer) for ```reg_loss```
* [AvgNonZeroReducer](reducers.md#avgnonzeroreducer) for ```pos_loss``` and ```neg_loss```

**Reducer input**:

* **pos_loss**: The loss per positive pair in the batch. Reduction type is ```"pos_pair"```.
* **neg_loss**: The loss per negative pair in the batch. Reduction type is ```"neg_pair"```.
* **reg_loss**: The regularization loss per element in the batch. Reduction type is ```"element"```.

## SoftTripleLoss   
[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf){target=_blank}
```python
losses.SoftTripleLoss(embedding_size, 
					num_classes, 
					centers_per_class, 
					la=20, 
					gamma=0.1, 
					reg_weight=0.2, 
					margin=0.01, 
					**kwargs)
```

**Equations**:

![soft_triple_loss_equation1](imgs/soft_triple_loss_equation1.png){: style="height:100px"}

where

![soft_triple_loss_equation2](imgs/soft_triple_loss_equation2.png){: style="height:80px"}


**Parameters**:

* **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set ```embedding_size``` to 512.
* **num_classes**: The number of classes in your training dataset.
* **centers_per_class**: The number of weight vectors per class. (The regular cross entropy loss has 1 center per class.) The paper uses 10.
* **la**: This is lambda in the above equation.
* **gamma**: This is gamma in the above equation. The paper uses 0.1.
* **reg_weight**: The regularization weight which encourages class centers to be close to each other. The paper uses 0.2.
* **margin**: The is delta in the above equations. The paper uses 0.01.

**Other info**

* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.SoftTripleLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.



## SphereFaceLoss 
[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf){target=_blank}

```python
losses.SphereFaceLoss(margin, num_classes, embedding_size, scale=1, **kwargs)
```

**Parameters**:

See [LargeMarginSoftmaxLoss](losses.md#largemarginsoftmaxloss)

**Other info**

* This also extends [WeightRegularizerMixin](losses.md#weightregularizermixin), so it accepts a ```regularizer``` and ```reg_weight``` as optional init arguments.
* This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:
```python
loss_func = losses.SphereFaceLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per element in the batch. Reduction type is ```"element"```.
* **reg_loss**: The weight regularization loss, if any. Reduction type is ```"already_reduced"```.


## TripletMarginLoss

```python
losses.TripletMarginLoss(margin=0.05, 
						distance_norm=2, 
						power=1, 
						swap=False, 
						smooth_loss=False, 
						triplets_per_anchor="all", 
						**kwargs)
```

**Equation**:

![triplet_margin_loss_equation](imgs/triplet_margin_loss_equation.png){: style="height:35px"}

**Parameters**:

* **margin**: The desired difference between the anchor-positive distance and the anchor-negative distance. This is ```m``` in the above equation.
* **distance_norm**: The norm used when calculating distance between embeddings
* **power**: Each pair's loss will be raised to this power.
* **swap**: Use the positive-negative distance instead of anchor-negative distance, if it violates the margin more.
* **smooth_loss**: Use the log-exp version of the triplet loss
* **triplets_per_anchor**: The number of triplets per element to sample within a batch. Can be an integer or the string "all". For example, if your batch size is 128, and triplets_per_anchor is 100, then 12800 triplets will be sampled. If triplets_per_anchor is "all", then all possible triplets in the batch will be used.

**Default reducer**: [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

* **loss**: The loss per triplet in the batch. Reduction type is ```"triplet"```.

## TupletMarginLoss
[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf){target=_blank}
```python
losses.TupletMarginLoss(margin, scale=64, **kwargs)
```

**Equation**:

![tuplet_margin_loss_equation](imgs/tuplet_margin_loss_equation.png){: style="height:80px"}

**Parameters**:

* **margin**: The angular margin (in degrees) applied to positive pairs. This is beta in the above equation. The paper uses a value of 5.73 degrees (0.1 radians).
* **scale**: This is ```s``` in the above equation.

The paper combines this loss with [IntraPairVarianceLoss](losses.md#intrapairvarianceloss). You can accomplish this by using [MultipleLosses](losses.md#multiplelosses):
```python
main_loss = losses.TupletMarginLoss(margin=5)
var_loss = losses.IntraPairVarianceLoss()
complete_loss = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
```

**Default reducer**: [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

* **loss**: The loss per positive pair in the batch. Reduction type is ```"pos_pair"```.

## WeightRegularizerMixin
Losses can extend this class in addition to BaseMetricLossFunction. You should extend this class if your loss function can make use of a [weight regularizer](regularizers.md).
```python
losses.WeightRegularizerMixin(regularizer, reg_weight, **kwargs)
```

**Parameters**:

* **regularizer**: The [regularizer](regularizers.md) to apply to the loss's learned weights.
* **reg_weight**: The amount the regularization loss will be multiplied by.

Extended by:

* [ArcFaceLoss](losses.md#arcfaceloss)
* [CosFaceLoss](losses.md#cosfaceloss)
* [LargeMarginSoftmaxLoss](losses.md#largemarginsoftmaxloss)
* [NormalizedSoftmaxLoss](losses.md#normalizedsoftmaxloss)
* [ProxyAnchorLoss](losses.md#proxyanchorloss)
* [ProxyNCALoss](losses.md#proxyncaloss)
* [SphereFaceLoss](losses.md#spherefaceloss)