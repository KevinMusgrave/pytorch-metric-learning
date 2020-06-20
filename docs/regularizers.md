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
regularizers.BaseWeightRegularizer(normalize_weights=True, reducer=None)
```

**Parameters**

* **normalize_weights**: If True, weights will be normalized to have a Euclidean norm of 1 before any regularization occurs.
* **reducer**: A [reducer](reducers.md) object. If None, then the default reducer will be used.

An object of this class can be passed as the ```regularizer``` argument into any class that extends [WeightRegularizerMixin](losses.md#weightregularizermixin).


## CenterInvariantRegularizer
[Deep Face Recognition with Center Invariant Loss](http://www1.ece.neu.edu/~yuewu/files/2017/twu024.pdf)
```python
regularizers.CenterInvariantRegularizer(normalize_weights=False)
```
Extends [BaseWeightRegularizer](regularizers.md#baseweightregularizer).
```normalize_weights``` must be False.


## RegularFaceRegularizer
[RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf)
```python
regularizers.RegularFaceRegularizer(normalize_weights=True)
```
Extends [BaseWeightRegularizer](regularizers.md#baseweightregularizer).



