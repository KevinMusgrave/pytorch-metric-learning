# AngularLoss

[Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf){target=_blank}

```python
losses.AngularLoss(alpha=40, **kwargs)
```

**Equation**:

$$
\begin{aligned}
l_{ang}(\mathcal{B}) &= \frac{1}{N} \sum_{\mathbf{x}_a \in \mathcal{B}} \left\lbrace \log \left[ 1 + \sum_{\substack{\mathbf{x}_n \in \mathcal{B} \\ y_n \neq y_a, y_p}} \exp\left(f_{a,p,n}\right) \right] \right\rbrace \\
f_{a,p,n} &= 4 \tan^2 \alpha \, (\mathbf{x}_a + \mathbf{x}_p)^T \mathbf{x}_n - 2 \left(1 + \tan^2 \alpha\right) \mathbf{x}_a^T \mathbf{x}_p
\end{aligned}
$$

Here $N$ is the batch size. Terms that depend only on $\lVert \mathbf{x} \rVert$ are dropped from $f_{a,p,n}$, as in the N-pair loss.

**Implementation notes**: Anchors, positives, and negatives are L2-normalized, matching the unit-length derivation of the equation. The loss is averaged over all ordered positive pairs rather than one positive per anchor; pairs without negatives contribute 0.

**Parameters**:

- **alpha**: The angle specified in degrees. The paper uses values between 36 and 55.

**Default distance**:

- [`LpDistance(p=2, power=1, normalize_embeddings=True)`](../distances.md#lpdistance)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss for every `a1`, where `(a1,p)` represents every positive pair in the batch. Reduction type is `"pos_pair"`.
