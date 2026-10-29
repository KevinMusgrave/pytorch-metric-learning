# FastAPLoss

[Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf){target=_blank}

```python
losses.FastAPLoss(num_bins=10, **options)
```

**Equation**:

$$
\operatorname{FastAP}_q = \sum_{z=1}^{L} \frac{h_q^+(z)\, H_q^+(z)}{H_q(z)},
\qquad
h_q^+(z) = \sum_{j \in \mathcal{S}_q^+} \delta\left(d_{qj}, z\right),
\qquad
H_q^+(z) = \sum_{l \le z} h_q^+(l)
$$

$$
L = \frac{1}{|Q|} \sum_{q \in Q} \left(1 - \frac{\operatorname{FastAP}_q}{|\mathcal{S}_q^+|}\right)
$$

Here $d_{qj}$ is the squared Euclidean distance between normalized embeddings, binned into $L$ bins over $[0, 2^{\text{power}}]$ with the triangular kernel $\delta(d, z) = \left[1 - |d - z| / \Delta\right]_+$; $h_q$ and $H_q$ are the soft histogram and cumulative histogram over all retrieved items, and $h_q^+$, $H_q^+$ over the positives of query $q$. $Q$ is the set of batch elements with at least one positive.

**Parameters**:

- **num_bins**: $L$, the number of soft histogram bins for calculating average precision. The paper suggests 10.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](../distances.md#lpdistance)
  - The only compatible distance is `LpDistance(normalize_embeddings=True, p=2)`. However, the `power` value can be changed.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element that has at least 1 positive in the batch. Reduction type is `"element"`.
