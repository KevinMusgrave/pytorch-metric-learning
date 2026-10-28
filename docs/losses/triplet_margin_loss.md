# TripletMarginLoss

```python
losses.TripletMarginLoss(margin=0.05,
                        swap=False,
                        smooth_loss=False,
                        triplets_per_anchor="all",
                        **kwargs)
```

**Equation**:

$$
L_{triplet} = \left[d_{ap} - d_{an} + m\right]_+
$$

Here $d_{ap}$ and $d_{an}$ are the anchor-positive and anchor-negative distances, and $[x]_+ = \max(0, x)$. With a similarity such as `CosineSimilarity`, the loss is $\left[s_{an} - s_{ap} + m\right]_+$.

**Parameters**:

- **margin**: The desired difference between the anchor-positive distance and the anchor-negative distance. This is `m` in the above equation.
- **swap**: Use the positive-negative distance instead of anchor-negative distance, if it violates the margin more.
- **smooth_loss**: Use the log-exp version of the triplet loss
- **triplets_per_anchor**: The number of triplets per element to sample within a batch. Can be an integer or the string "all". For example, if your batch size is 128, and triplets_per_anchor is 100, then 12800 triplets will be sampled. If triplets_per_anchor is "all", then all possible triplets in the batch will be used.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: The loss per triplet in the batch. Reduction type is `"triplet"`.
