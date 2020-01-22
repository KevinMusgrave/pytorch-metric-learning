# Regularizers

Regularizers are like helper-loss functions. They might not require any labels or embeddings as input, and might instead operate on weights that are learned by a network or loss function.

Here is an example when used in conjunction with a compatible loss function:

## BaseWeightRegularizer
Weight regularizers take in a 2-D tensor of weights of size (num_classes, embedding_size).
```python
regularizers.BaseWeightRegularizer(normalize_weights=True)
```

**Parameters**

* **normalize_weights**: If True, weights will be normalized to have a Euclidean norm of 1 before any regularization occurs.


## RegularFaceRegularizer
[RegularFace: Deep Face Recognition via Exclusive Regularization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_RegularFace_Deep_Face_Recognition_via_Exclusive_Regularization_CVPR_2019_paper.pdf)
```python
regularizers.RegularFaceRegularizer(**kwargs)
```
Extends [BaseWeightRegularizer](regularizers.md#baseweightregularizer)

