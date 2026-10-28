# LiftedStructureLoss

The original lifted structure loss as presented in [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf){target=_blank}

```python
losses.LiftedStructureLoss(neg_margin=1, pos_margin=0, **kwargs):
```

**Equation**:

$$
\begin{aligned}
\tilde{J}_{i,j} &= \log\left(\sum_{(i,k) \in \mathcal{N}} \exp\lbrace \alpha - D_{i,k} \rbrace + \sum_{(j,l) \in \mathcal{N}} \exp\lbrace \alpha - D_{j,l} \rbrace\right) + D_{i,j} \\
\tilde{J} &= \frac{1}{2\lvert\mathcal{P}\rvert} \sum_{(i,j) \in \mathcal{P}} \max\left(0, \tilde{J}_{i,j}\right)^2
\end{aligned}
$$

Here $\mathcal{P}$ and $\mathcal{N}$ are the positive and negative pairs in the batch, and $D_{i,j}$ is the distance between embeddings $i$ and $j$.

**Implementation notes**: $\tilde{J}_{i,j}$ uses the negatives of both $i$ and $j$, so the anchors and positives must come from the same set. Passing a separate `ref_emb` raises `ValueError`; use [GeneralizedLiftedStructureLoss](generalized_lifted_structure_loss.md#generalizedliftedstructureloss) for separate query and reference sets.

**Parameters**:

- **pos_margin**: The margin in the expression `D_(i,j) - margin`. The paper uses `pos_margin = 0 `, which is why it does not appear in the above equation.
- **neg_margin**: This is `alpha` in the above equation. The paper uses 1.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
