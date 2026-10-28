# NPairsLoss

[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf){target=_blank}

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

If your batch has more than 2 samples per label, then you should use [NTXentLoss](ntxent_loss.md#ntxentloss).

```python
losses.NPairsLoss(**kwargs)
```

**Default distance**:

- [`DotProductSimilarity()`](../distances.md#dotproductsimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
