# FastAPLoss

[Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf){target=_blank}

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.FastAPLoss(num_bins=10, **kwargs)
```

**Parameters**:

- **num_bins**: The number of soft histogram bins for calculating average precision. The paper suggests using 10.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](../distances.md#lpdistance)
  - The only compatible distance is `LpDistance(normalize_embeddings=True, p=2)`. However, the `power` value can be changed.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element that has at least 1 positive in the batch. Reduction type is `"element"`.
