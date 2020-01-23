# Regularizers

Regularizers are like helper-loss functions. They might not require any labels or embeddings as input, and might instead operate on weights that are learned by a network or loss function.

Here is an example when used in conjunction with a compatible loss function:
```python
from pytorch_metric_learning import losses, regularizers
R = regularizers.RegularFaceRegularizer()
loss = losses.ArcFaceLoss(margin=30, num_classes=100, embedding_size=128, regularizer=R)
```

## BaseWeightRegularizer
Weight regularizers take in a 2-D tensor of weights of size (num_classes, embedding_size).
```python
regularizers.BaseWeightRegularizer(normalize_weights=True)
```

**Parameters**

* **normalize_weights**: If True, weights will be normalized to have a Euclidean norm of 1 before any regularization occurs.

An object of this class can be passed as the _regularizer_ argument into any class that extends [WeightRegularizerMixin](losses.md#weightregularizermixin).

## RegularFaceRegularizer
[RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf)
```python
regularizers.RegularFaceRegularizer(**kwargs)
```
Extends [BaseWeightRegularizer](regularizers.md#baseweightregularizer).



