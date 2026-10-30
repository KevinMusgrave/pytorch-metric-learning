# AngularLoss

[Deep Metric Learning with Angular Loss](https://arxiv.org/abs/1708.01682)

```python {.signature}
losses.AngularLoss(alpha=40, **options)
```

**Equation**:

$$
\begin{aligned}
l_{ang}(\mathcal{B}) &= \frac{1}{N} \sum_{\mathbf{x}_a \in \mathcal{B}} \left\lbrace \log \left[ 1 + \sum_{\substack{\mathbf{x}_n \in \mathcal{B} \\ y_n \neq y_a, y_p}} \exp\left(f_{a,p,n}\right) \right] \right\rbrace \\
f_{a,p,n} &= 4 \tan^2 \alpha \, (\mathbf{x}_a + \mathbf{x}_p)^T \mathbf{x}_n - 2 \left(1 + \tan^2 \alpha\right) \mathbf{x}_a^T \mathbf{x}_p
\end{aligned}
$$

Here $N$ is the batch size. Terms that depend only on $\lVert \mathbf{x} \rVert$ are dropped from $f_{a,p,n}$, as in the N-pair loss.

**Implementation notes**: Anchors, positives, and negatives are L2-normalized,
matching the unit-length derivation of the equation. The loss is computed for
every positive pair `(a, p)`, from labels or from `indices_tuple`, rather than
one positive per anchor. The negatives of a pair are every reference with a
label different from the anchor's, so `labels` are required even with
`indices_tuple`; negative pairs in `indices_tuple` are not used, so a miner that
returns only positive pairs still gives a loss. A pair without negatives
contributes 0. With `ref_emb`, positives and negatives come from the reference
set.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.AngularLoss(alpha=45)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **alpha**: The angle $\alpha$ in degrees. The paper uses values between 36 and 55.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance).
  This is the only compatible distance; another distance class raises
  `TypeError` and other settings raise `ValueError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per (anchor, positive) pair, indexed by `(anchors, positives)`.
