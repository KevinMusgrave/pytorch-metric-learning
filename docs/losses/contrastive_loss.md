# ContrastiveLoss

```python
losses.ContrastiveLoss(pos_margin=0, neg_margin=1, **kwargs):
```

**Equation**:

If using a distance metric like [LpDistance](../distances.md#lpdistance), the loss is:

$$
L_{contrastive} = \left[d_p - m_{pos}\right]_+ + \left[m_{neg} - d_n\right]_+
$$

Here $d_p$ and $d_n$ are positive-pair and negative-pair distances, $m_{pos}$ and $m_{neg}$ are `pos_margin` and `neg_margin`, and $[x]_+ = \max(0, x)$.

If using a similarity metric like [CosineSimilarity](../distances.md#cosinesimilarity), the loss is:

$$
L_{contrastive} = \left[m_{pos} - s_p\right]_+ + \left[s_n - m_{neg}\right]_+
$$

Here $s_p$ and $s_n$ are positive-pair and negative-pair similarities.

**Implementation notes**: The hinges are not squared. The default reducer averages the positive-pair losses greater than zero and, separately, the negative-pair losses greater than zero, then adds the two averages.

**Parameters**:

- **pos_margin**: The distance (or similarity) over (under) which positive pairs will contribute to the loss.
- **neg_margin**: The distance (or similarity) under (over) which negative pairs will contribute to the loss.

Note that the default values for `pos_margin` and `neg_margin` are suitable if you are using a non-inverted distance measure, like [LpDistance](../distances.md#lpdistance). If you use an inverted distance measure like [CosineSimilarity](../distances.md#cosinesimilarity), then more appropriate values would be `pos_margin = 1` and `neg_margin = 0`.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **pos_loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
- **neg_loss**: The loss per negative pair in the batch. Reduction type is `"neg_pair"`.
