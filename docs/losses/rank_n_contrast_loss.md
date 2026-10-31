# RankNContrastLoss

[Rank-N-Contrast: Learning Continuous Representations for Regression](https://arxiv.org/abs/2210.01189)
(Zha et al.), first circulated as Supervised Contrastive Regression, orders embeddings by
how close their continuous targets are. For each anchor, a sample should be more similar
than every sample whose target is at least as far away. Train the encoder with this loss,
then fit a regressor on the frozen embeddings.

```python {.signature}
losses.RankNContrastLoss(temperature=2.0, label_ndim=1, label_distance=None, **options)
```

**Equation**:

For anchor $i$, with $M$ other samples, target distance $d$, and similarity $s$:

$$
\ell_i = \frac{1}{M} \sum_{j \neq i}
-\log \frac{\exp(s_{i,j} / \tau)}{\sum_{k \in S_{i,j}} \exp(s_{i,k} / \tau)},
\qquad
S_{i,j} = \{ k \neq i : d(y_i, y_k) \geq d(y_i, y_j) \}
$$

The loss is the mean of $\ell_i$ over anchors. The default similarity is the negative
Euclidean distance between unnormalized embeddings, and the default target distance is L1.

**Target contract**: `labels` hold continuous targets, not classes:

- A `(N,)` floating-point tensor by default, or `(N, T)` with `label_ndim=2`, where the
  default target distance sums over `T`. Integer targets raise `TypeError`; convert them
  with `.float()`, choosing a dtype that represents them exactly.
- NaN or infinite targets raise `ValueError`.
- Targets keep their own dtype. They are never cast to the embeddings' dtype, because
  rounding would turn nearby targets into ties.
- `label_distance(labels, references)` replaces the target distance, for example with an
  angular distance. It returns a finite `(len(labels), len(references))` tensor.

**Tie handling**: Ties follow the `≥` in $S_{i,j}$. Samples at exactly the same target
distance from the anchor are in each other's denominators, so the loss asks for no order
among them. This includes a sample's second view, and two samples either side of the
anchor at equal distance. Comparison is exact, so targets that differ by rounding are not
tied. If every target is equal, each sample's denominator holds all others.

**Implementation notes**: A row's denominators are suffix sums over the row sorted by
target distance, computed once with `logcumsumexp`, so one forward pass costs
$O(N^2 \log N)$ time rather than the $O(N^3)$ of evaluating each denominator. Peak memory
is proportional to the distance matrix. float16 and bfloat16 embeddings accumulate in
float32. With a similarity distance, $s$ is its value. A batch with no other sample gives
a differentiable zero. `indices_tuple` raises `ValueError`, because every sample is both
a positive and a negative at different ranks.

`ref_emb` and `ref_labels` compare anchors with a separate set: every reference is
considered, and none is excluded as the anchor's own row.

**Paper settings**: This is equations 1 to 3 of the paper. Its $2N$ samples are two views
of $N$ inputs, as in the example, so pass both views and repeat the targets. The paper
selects $\tau$ from 0.1 to 5 for each dataset. The default of 2 is the authors' code's.
The authors then train a regressor on frozen embeddings.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.RankNContrastLoss()
first_view = torch.randn(16, 128, requires_grad=True)
second_view = torch.randn(16, 128, requires_grad=True)
targets = torch.rand(16) * 100

embeddings = torch.cat([first_view, second_view])
loss = loss_func(embeddings, targets.repeat(2))
loss.backward()
```

**Parameters**:

- **temperature**: The temperature $\tau$. It must be positive.
- **label_ndim**: 1 for `(N,)` targets, or 2 for `(N, T)` targets.
- **label_distance**: Optional replacement for the target distance.

**Default distance**:

- [`LpDistance(normalize_embeddings=False, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
