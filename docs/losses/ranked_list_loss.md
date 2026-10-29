# RankedListLoss

[Ranked List Loss for Deep Metric Learning](https://arxiv.org/abs/1903.03238)

```python
losses.RankedListLoss(margin, Tn, imbalance=0.5, alpha=None, Tp=0, **options)
```

**Equation**:

$$
L = \frac{1}{N} \sum_{i=1}^{N} \left[(1 - \lambda)\, L_P(\mathbf{x}_i) + \lambda\, L_N(\mathbf{x}_i)\right]
$$

$$
L_P(\mathbf{x}_i) = \frac{\sum_{j \in P^*_i} w_{ij} \left(d_{ij} - (\alpha - m)\right)}{\sum_{j \in P^*_i} w_{ij}},
\qquad
L_N(\mathbf{x}_i) = \frac{\sum_{k \in N^*_i} w_{ik} \left(\alpha - d_{ik}\right)}{\sum_{k \in N^*_i} w_{ik}}
$$

$$
w_{ij} = e^{T_p \left(d_{ij} - (\alpha - m)\right)}, \qquad w_{ik} = e^{T_n \left(\alpha - d_{ik}\right)}
$$

Here $d$ is the Euclidean distance between normalized embeddings, $P^*_i$ holds the positives of $\mathbf{x}_i$ with $d_{ij} > \alpha - m$, $N^*_i$ holds the negatives with $d_{ik} < \alpha$, $\lambda$ is `imbalance`, and $m$ is `margin`.

**Implementation notes**: Weight sums are offset by `1e-5` so anchors without violating pairs contribute 0.

**Parameters**:

The defaults correspond to the paper's RLL-Simpler variant, which it recommends for exploratory analysis: `Tp=0` weights positives uniformly, and `alpha=None` sets $\alpha = 1 + m/2$ for margin $m$.

- **margin** (float): margin between positive and negative set
- **imbalance** (float): tradeoff between positive and negative sets. As the name suggests this takes into account
  the imbalance between positive and negative samples in the dataset
- **alpha** (float): smallest distance between negative points
- **Tp & Tn** (float): temperatures for, respectively, positive and negative pairs weighting.
