# PNPLoss

[Rethinking the Optimization of Average Precision: Only Penalizing Negative Instances before Positive Ones is Enough](https://arxiv.org/pdf/2102.04640.pdf){target=_blank}

```python
losses.PNPLoss(b=2, alpha=1, anneal=0.01, variant="O", **kwargs)
```

**Equation**:

$$
\begin{aligned}
L_O &= \frac{1}{|S_P|} \sum_{i \in S_P} R(i, S_N) && (5) \\
L_{I_u} &= \frac{1}{|S_P|} \sum_{i \in S_P} \left(1 + R(i, S_N)\right) \log\left(1 + R(i, S_N)\right) && (7) \\
L_{I_b} &= \frac{1}{|S_P|} \sum_{i \in S_P} \frac{1}{b^2} \left(b R(i, S_N) - \log\left(1 + b R(i, S_N)\right)\right) && (9) \\
L_{D_s} &= \frac{1}{|S_P|} \sum_{i \in S_P} \log\left(1 + R(i, S_N)\right) && (12) \\
L_{D_q} &= 1 - \frac{1}{|S_P|} \sum_{i \in S_P} \frac{1}{\left(1 + R(i, S_N)\right)^{\alpha}} \quad (\alpha \geq 1) && (13)
\end{aligned}
$$

Here $R(i, S_N) = \sum_{j \in S_N} \sigma\left((s_j - s_i)/\tau\right)$, where $s_i$ is the cosine similarity between the query and element $i$, $\sigma$ is the sigmoid, and $\tau$ is `anneal`.

**Implementation notes**: The query is counted in its own $S_P$, as in the authors' implementation.

**Parameters**:

- **b**: The boundary of PNP-Ib (see equation 9 above). The paper uses 2.
- **alpha**: The power of PNP-Dq (see equation 13 above). The paper uses 8.
- **anneal**: The temperature of the sigmoid function. (The sigmoid function is used for `R` in the equations above.) The paper uses 0.01.
- **variant**: The name of the variant. The options are {"Ds", "Dq", "Iu", "Ib", "O"}. The paper uses "Dq".

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)
  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss averaged over elements that have at least 1 other positive in the batch. Reduction type is `"already_reduced"`, so the reducer does not change it.
