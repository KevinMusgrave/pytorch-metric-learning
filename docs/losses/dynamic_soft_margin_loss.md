# DynamicSoftMarginLoss

[Learning Local Descriptors With a CDF-Based Dynamic Soft Margin](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Local_Descriptors_With_a_CDF-Based_Dynamic_Soft_Margin_ICCV_2019_paper.pdf)

```python
losses.DynamicSoftMarginLoss(min_val=-2.0, num_bins=10, momentum=0.01, **options)
```

**Equation**:

$$
L = \frac{1}{T} \sum_{t=1}^{T} \hat F(\Delta_t)\, \Delta_t, \qquad \Delta_t = d(\mathbf{x}_a, \mathbf{x}_p) - d(\mathbf{x}_a, \mathbf{x}_n)
$$

Here $\Delta_t$ is the distance difference of triplet $t$, and $\hat F$ is the running empirical cumulative distribution of $\Delta$, estimated with a histogram updated by momentum:

$$
\hat h \leftarrow (1 - \eta)\, \hat h + \eta\, h_{\text{batch}}, \qquad \hat F(\Delta) = \sum_{b \le \operatorname{bin}(\Delta)} \hat h_b
$$

**Paper settings**: The authors use 512 bins; the library default is 10. Pass `num_bins=512` to match.

The running histogram is a buffer of the loss module. Move the loss with
`loss_func.to(device, dtype)`; outside autocast, a placement mismatch raises
`ValueError`.
Under autocast, bin coordinates and histogram contributions use the buffer's
dtype even when distance differences have lower precision. Updating the
histogram does not change its dtype.

**Parameters**:

- **min_val**: finite negative lower bound of the histogram. Its upper bound is
  `-min_val`. The default range `[-2, 2]` covers differences of Euclidean
  distances between normalized embeddings.
- **num_bins**: number of equally spaced bin centers including both endpoints;
  at least 2. The library default is 10; the authors' implementation uses 512.
- **momentum**: weight of the current batch's normalized histogram, between 0
  and 1. The first nonempty batch initializes the histogram regardless of this
  setting. Empty valid sets do not update it.

With labels or supplied triplets, this loss weights `d_positive - d_negative`
for each selected triplet. With neither, row `i` in `embeddings` is matched to
row `i` in `ref_emb`; all off-diagonal candidates are negatives. Each matched
pair uses the closest negative from either stream. If stream sizes differ,
extra rows provide negatives but do not create additional positive pairs.
This path does not apply the Brown-dataset outlier threshold used by the
authors' descriptor-training code.

Each difference contributes linearly to its two adjacent histogram centers.
The cumulative probability at the lower center weights the original difference.
Values outside the histogram range use the nearest endpoint. Histogram updates
and weights are detached; gradients flow through the distances. The compatible
distance is normalized, unsquared Euclidean distance.

The running histogram is saved in `state_dict`. It supports ordinary successive
training steps and multiple forwards before backward. Starting a new module
starts a new histogram. The corrected endpoint spacing and per-batch momentum
change numerical results from version 2.9.0.

The original PML implementation was contributed by **Domenico Muscillo**.
See the [authors' reference code](https://github.com/lg-zhang/dynamic-soft-margin-pytorch)
for the descriptor-training setup.
