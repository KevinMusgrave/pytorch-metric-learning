# HistogramLoss

[Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)

```python
losses.HistogramLoss(n_bins=None, delta=None, **options)
```

**Equation**:

$$
L = \sum_{r=1}^{R} p^-_r \, \Phi^+_r, \qquad \Phi^+_r = \sum_{q=1}^{r} p^+_q
$$

$$
p^\pm_r = \frac{1}{|\mathcal{S}^\pm|} \sum_{(i,j) \in \mathcal{S}^\pm} \delta_{ij,r},
\qquad
\delta_{ij,r} = \begin{cases} (s_{ij} - t_{r-1}) / \Delta & s_{ij} \in [t_{r-1}, t_r] \\ (t_{r+1} - s_{ij}) / \Delta & s_{ij} \in [t_r, t_{r+1}] \\ 0 & \text{otherwise} \end{cases}
$$

Here $s_{ij}$ is the cosine similarity of pair $(i, j)$, $\mathcal{S}^+$ and $\mathcal{S}^-$ are the sets of positive and negative pairs, $t_r$ are the $R$ uniformly spaced nodes of $[-1, 1]$ with spacing $\Delta$ (`delta`), and $p^\pm$ are the linearly interpolated histograms of positive and negative similarities. The loss is the probability that a random negative pair is more similar than a random positive pair.

**Implementation notes**: Each unordered pair is counted once, so duplicate pairs from all-triplet enumeration do not change the histograms.

**Parameters**:

- **n_bins**: The number of bins used to construct the histogram. Default is 100 when both `n_bins` and `delta` are `None`.
- **delta**: The mesh of the uniform partition of the interval [-1, 1] used to construct the histogram. If not set the value of n_bins will be used.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- This loss returns an **already reduced** loss.
