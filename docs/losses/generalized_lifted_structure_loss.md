# GeneralizedLiftedStructureLoss

This was presented in [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf){target=_blank}. It is a modification of the original [LiftedStructureLoss](lifted_structure_loss.md#liftedstructureloss)

```python
losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0, **kwargs)
```

**Equation**:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{LG}} = \sum_{i=1}^{P} \sum_{a=1}^{K} \Bigg[ & \log \sum_{\substack{p=1 \\ p \neq a}}^{K} e^{D\left(f(x_a^i),\, f(x_p^i)\right)} \\
& + \log \sum_{\substack{j=1 \\ j \neq i}}^{P} \sum_{n=1}^{K} e^{m - D\left(f(x_a^i),\, f(x_n^j)\right)} \Bigg]_+
\end{aligned}
$$

Here the batch has $P$ classes with $K$ samples each, $x_a^i$ is sample $a$ of class $i$, $f$ is the embedding network, and $D$ is the distance. The outer sums run over all anchors, the first inner sum over their positives, and the second over their negatives.

**Implementation notes**: The loss is computed per anchor and averaged over the batch rather than summed.

**Parameters**:

- **pos_margin**: The margin in the expression `e^(D - margin)`. The paper uses `pos_margin = 0 `, which is why this margin does not appear in the above equation.
- **neg_margin**: This is `m` in the above equation. The paper used values between 0.1 and 1.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
