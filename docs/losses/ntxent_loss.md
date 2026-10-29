# NTXentLoss

Also known as InfoNCE, a generalization of [NPairsLoss](n_pairs_loss.md) used in
[CPC](https://arxiv.org/abs/1807.03748), [MoCo](https://arxiv.org/abs/1911.05722),
and [SimCLR](https://arxiv.org/abs/2002.05709).

```python
losses.NTXentLoss(temperature=0.07, **options)
```

**Equation**:

$$
\mathcal{L}_{q,k_+} = -\log \frac{\exp\left(q \cdot k_+ / \tau\right)}{\exp\left(q \cdot k_+ / \tau\right) + \sum_{i=1}^{K} \exp\left(q \cdot k_i / \tau\right)}
$$

Here $q$ is the anchor, $k_+$ one of its positives, $k_1, \ldots, k_K$ the negatives that share anchor $q$, and $q \cdot k$ the similarity, cosine by default.

**Implementation notes**: One loss is computed for each positive pair. Its
denominator holds that positive and every negative pair with the same anchor;
the anchor's other positives are not in it. For example, with
`labels = [0, 0, 1, 2]` there are two losses: positive pair `(0, 1)` with
negative pairs `(0, 2)` and `(0, 3)`, and positive pair `(1, 0)` with negative
pairs `(1, 2)` and `(1, 3)`. Labels 1 and 2 have no positive pairs, so the
negative pair `(2, 3)` is not used. With an `indices_tuple`, the positive pairs
are its `(pos_anchors, positives)` and the negatives of anchor `a` are the pairs
whose `neg_anchors` entry is `a`.

The loss is computed in log space as
$\log\left(e^{q \cdot k_+ / \tau} + e^{\mathrm{LSE}_q}\right) - q \cdot k_+ / \tau$,
where $\mathrm{LSE}_q$ is the log-sum-exp over the anchor's negatives, so no
positive-by-negative matrix is built. A positive pair whose anchor has no
negatives contributes 0. float16 and bfloat16 inputs accumulate in float32.
With a distance where smaller means closer, the logits are negated distances.

For two augmented views of the same batch, wrap the loss in
[SelfSupervisedLoss](self_supervised_loss.md) instead of building labels.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=0.5))
embeddings = torch.randn(32, 128, requires_grad=True)
ref_emb = torch.randn(32, 128, requires_grad=True)
loss = loss_func(embeddings, ref_emb)
loss.backward()
```

**Parameters**:

- **temperature**: The temperature $\tau$. MoCo uses 0.07; SimCLR uses 0.5.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per (anchor, positive) pair, indexed by `(anchors, positives)`.
