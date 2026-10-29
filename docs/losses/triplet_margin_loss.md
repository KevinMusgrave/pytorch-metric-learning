# TripletMarginLoss

[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
(Schroff, Kalenichenko, and Philbin, 2015). The triplet hinge builds on the
large-margin nearest-neighbor objective of
[Distance Metric Learning for Large Margin Nearest Neighbor Classification](https://jmlr.org/papers/v10/weinberger09a.html)
(Weinberger and Saul, 2009).

```python
losses.TripletMarginLoss(
    margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all", **options
)
```

**Equation**:

$$
L_{triplet} = \left[d_{ap} - d_{an} + m\right]_+
$$

Here $d_{ap}$ and $d_{an}$ are the anchor-positive and anchor-negative distances, $m$ is `margin`, and $[x]_+ = \max(0, x)$. With a similarity such as `CosineSimilarity`, the loss is $\left[s_{an} - s_{ap} + m\right]_+$.

**Implementation notes**: The distances are not squared by default; FaceNet
uses squared Euclidean distances, which `LpDistance(power=2)` provides. With
`smooth_loss=True`, the hinge is replaced by
$\log\left(1 + e^{d_{ap} - d_{an} + m}\right)$. With `swap=True`, $d_{an}$ is
replaced by the positive-negative distance when that is smaller (or, for a
similarity, when it is larger), following
[Balntas et al., 2016](http://www.bmva.org/bmvc/2016/papers/paper119/index.html).
With `ref_emb`, positives and negatives come from the reference set, and the
swap distance is computed between the reference rows. Triplets come from labels
or from the supplied `indices_tuple`; pairs are combined into every triplet
that shares an anchor.

**Paper settings**: FaceNet uses squared Euclidean distances between
L2-normalized embeddings and a margin of 0.2; the default `margin=0.05` and
unsquared distances differ. Pass
`margin=0.2, distance=distances.LpDistance(power=2)` for FaceNet's setting.

**Example**:

```python
import torch
from pytorch_metric_learning import losses, miners

loss_func = losses.TripletMarginLoss(margin=0.2)
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels, miner(embeddings, labels))
loss.backward()
```

**Parameters**:

- **margin**: The margin $m$.
- **swap**: Use the positive-negative distance instead of the anchor-negative
  distance when it violates the margin more.
- **smooth_loss**: Use the soft-plus form of the hinge.
- **triplets_per_anchor**: The number of triplets per anchor sampled within a
  batch, an integer or `"all"`. For example, a batch of 128 with
  `triplets_per_anchor=100` samples 12800 triplets. With `"all"`, every triplet
  in the batch is used. It applies only when triplets come from labels.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: One loss per triplet, indexed by
  `TripletIndices(anchors, positives, negatives)`.
