# NPairsLoss

[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf){target=_blank}

If your batch has more than 2 samples per label, then you should use [NTXentLoss](ntxent_loss.md#ntxentloss).

```python
losses.NPairsLoss(**options)
```

**Equation**:

$$
L = \frac{1}{N} \sum_{i=1}^{N} \log\left(1 + \sum_{j \neq i} \exp\left(\mathbf{x}_i^\top \mathbf{x}_j^+ - \mathbf{x}_i^\top \mathbf{x}_i^+\right)\right)
$$

Here the batch provides $N$ pairs $(\mathbf{x}_i, \mathbf{x}_i^+)$ with distinct labels, and the loss is the cross entropy of classifying anchor $i$'s positive among all $N$ positives using dot-product similarity.

**Implementation notes**: For each label, the first positive pair in index order forms the $(\mathbf{x}_i, \mathbf{x}_i^+)$ pair; other samples of that label are ignored. Embeddings are not normalized by default, as in the paper, which regularizes their norms separately.

**Default distance**:

- [`DotProductSimilarity()`](../distances.md#dotproductsimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
