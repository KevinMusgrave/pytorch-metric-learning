# HistogramLoss

[Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.HistogramLoss(n_bins=None, delta=None)
```

**Parameters**:

- **n_bins**: The number of bins used to construct the histogram. Default is 100 when both `n_bins` and `delta` are `None`.
- **delta**: The mesh of the uniform partition of the interval [-1, 1] used to construct the histogram. If not set the value of n_bins will be used.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- This loss returns an **already reduced** loss.
