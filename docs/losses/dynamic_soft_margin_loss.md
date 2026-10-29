# DynamicSoftMarginLoss

[Learning Local Descriptors With a CDF-Based Dynamic Soft Margin](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Local_Descriptors_With_a_CDF-Based_Dynamic_Soft_Margin_ICCV_2019_paper.pdf)

```python
losses.DynamicSoftMarginLoss(
    min_val=-2.0,
    num_bins=10,
    momentum=0.01,
    *,
    distance=None,
    embedding_regularizer=None,
    embedding_reg_weight=1.0,
)
```

**Equation**:

$$
L = \frac{1}{T} \sum_{t=1}^{T} \hat F(\Delta_t)\, \Delta_t, \qquad \Delta_t = d(\mathbf{x}_a, \mathbf{x}_p) - d(\mathbf{x}_a, \mathbf{x}_n)
$$

Here $\Delta_t$ is the distance difference of triplet $t$, and $\hat F$ is the running empirical cumulative distribution of $\Delta$, estimated with a histogram updated by momentum:

$$
\hat h \leftarrow (1 - \eta)\, \hat h + \eta\, h_{\text{batch}}, \qquad \hat F(\Delta) = \sum_{b \le \operatorname{bin}(\Delta)} \hat h_b
$$

**Implementation notes**: With labels or an `indices_tuple`, the differences
come from every triplet. With neither, row `i` of `embeddings` and row `i` of
`ref_emb` form a positive pair, every off-diagonal entry is a negative, and each
pair uses the closest negative from either its row or its column. If the two
sets have different sizes, the extra rows provide negatives but no positive
pairs. This path requires `ref_emb`; without it, a call with neither labels nor
an `indices_tuple` raises `ValueError`. This path does not apply the Brown-dataset outlier
threshold used by the authors' descriptor-training code; see the
[authors' reference code](https://github.com/lg-zhang/dynamic-soft-margin-pytorch)
for that setup.

Each forward pass updates the histogram with the current batch before weighting
it. Each difference contributes linearly to its two adjacent histogram centers,
and the cumulative probability at the lower center weights the difference.
Values outside the histogram range use the nearest endpoint. The first nonempty
batch initializes the histogram; batches without differences leave it
unchanged. Histogram updates and weights are detached, so gradients flow only
through the distances.

The running histogram `hist` is a buffer of the loss module and is saved in
`state_dict`. Move the loss with `loss_func.to(device, dtype)`; outside
autocast, a placement mismatch raises `ValueError`. Under autocast, bin
coordinates and histogram contributions use the buffer's dtype, and updating
the histogram does not change its dtype. Multiple forward passes before one
backward pass are supported. A new module starts a new histogram.

**Paper settings**: The authors use 512 bins; pass `num_bins=512` to match.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.DynamicSoftMarginLoss(num_bins=512)
anchors = torch.randn(32, 128, requires_grad=True)
positives = torch.randn(32, 128, requires_grad=True)
loss = loss_func(anchors, ref_emb=positives)
loss.backward()
```

**Parameters**:

- **min_val**: The finite negative lower bound of the histogram. Its upper
  bound is `-min_val`. The default range `[-2, 2]` covers differences of
  Euclidean distances between normalized embeddings. A non-finite or
  non-negative value raises `ValueError`.
- **num_bins**: The number of equally spaced bin centers, including both
  endpoints. Values below 2 raise `ValueError`.
- **momentum**: The weight $\eta$ of the current batch's normalized histogram,
  between 0 and 1; other values raise `ValueError`.
- **distance**, **embedding_regularizer**, **embedding_reg_weight**: See
  [loss options](../losses.md#loss-options).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance).
  This is the only compatible distance; another distance class raises
  `TypeError` and other settings raise `ValueError`.

**Reducer**:

- None. The loss reduces its own value, so it has no `reducer` option; passing
  `reducer` raises `TypeError`.

**Loss terms**:

- **loss**: The mean of $\hat F(\Delta_t)\,\Delta_t$ over all differences,
  with `indices=None`.
