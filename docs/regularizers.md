# Regularizers

Regularizers are applied to weights and embeddings without the need for labels or tuples.

Here is an example of a weight regularizer being passed to a loss function.
```python
from pytorch_metric_learning import losses, regularizers
R = regularizers.RegularFaceRegularizer()
loss = losses.ArcFaceLoss(margin=30, num_classes=100, embedding_size=128, weight_regularizer=R)
```

## BaseRegularizer
```python
regularizers.BaseWeightRegularizer(collect_stats = False, 
								reducer = None, 
								distance = None)
```

An object that extends this class can be passed as the ```embedding_regularizer``` into any loss function. It can also be passed as the ```weight_regularizer``` into any class that extends [WeightRegularizerMixin](losses.md#weightregularizermixin).

**Parameters**

* **collect_stats**: If True, will collect various statistics that may be useful to analyze during experiments. If False, these computations will be skipped. Want to make ```True``` the default? Set the global [COLLECT_STATS](common_functions.md#collect_stats) flag.
* **reducer**: A [reducer](reducers.md) object. If None, then the default reducer will be used.
* **distance**: A [distance](distances.md) object. If None, then the default distance will be used.

**Default distance**: 

 - [```LpDistance(normalize_embeddings=True, p=2, power=1)```](distances.md#lpdistance)


**Default reducer**: 

 - [MeanReducer](reducers.md#meanreducer)


## CenterInvariantRegularizer
[Deep Face Recognition with Center Invariant Loss](http://www1.ece.neu.edu/~yuewu/files/2017/twu024.pdf){target=_blank}

This encourages unnormalized embeddings or weights to all have the same Lp norm.
```python
regularizers.CenterInvariantRegularizer(**kwargs)
```

**Default distance**: 

 - [```LpDistance(normalize_embeddings=False, p=2, power=1)```](distances.md#lpdistance)
     - The distance must be ```LpDistance(normalize_embeddings=False, power=1)```. However, ```p``` can be changed.


**Default reducer**: 

 - [MeanReducer](reducers.md#meanreducer)


## LpRegularizer
This encourages embeddings/weights to have a small Lp norm.
```python
regularizers.LpRegularizer(p=2, power=1, **kwargs)
```

**Parameters**

* **p**: The type of norm. For example, ```p=1``` is the Manhattan distance, and ```p=2``` is Euclidean distance.


**Default distance**: 

 - This regularizer does not use a distance object, so setting this parameter will have no effect.


**Default reducer**: 

 - [MeanReducer](reducers.md#meanreducer)


## RegularFaceRegularizer
[RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf){target=_blank}

This should be applied as a weight regularizer. It penalizes class vectors that are very close together.

```python
regularizers.RegularFaceRegularizer(**kwargs)
```

**Default distance**: 

 - [```CosineSimilarity()```](distances.md#cosinesimilarity)
     - Only inverted distances are compatible. For example, [```DotProductSimilarity()```](distances.md#dotproductsimilarity) also works.

**Default reducer**: 

 - [MeanReducer](reducers.md#meanreducer)


## SparseCentersRegularizer
[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf){target=_blank}

This should be applied as a weight regularizer. It encourages multiple class centers to "merge", i.e. group together.

```python
regularizers.SparseCentersRegularizer(num_classes, centers_per_class, **kwargs)
```

**Parameters**

* **num_classes**: The number of classes in your training dataset.
* **centers_per_class**: The number of rows in the weight matrix that correspond to 1 class.

**Default distance**: 

 - [```CosineSimilarity()```](distances.md#cosinesimilarity)
     - This is the only compatible distance.

**Default reducer**: 

 - [DivisorReducer](reducers.md#divisorreducer)


## ZeroMeanRegularizer
[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}


```python
regularizers.ZeroMeanRegularizer(**kwargs)
```

**Equation**

In this equation, ```N``` is the batch size, ```M``` is the size of each embedding.

![zero_mean_regularizer_equation](imgs/zero_mean_regularizer_equation.png){: style="height:70px"}


**Default distance**: 

 - This regularizer does not use a distance object, so setting this parameter will have no effect.


**Default reducer**: 

 - [MeanReducer](reducers.md#meanreducer)