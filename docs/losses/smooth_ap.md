# SmoothAPLoss

[Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163){target=_blank}

```python
losses.SmoothAPLoss(
    temperature=0.01,
    **kwargs
)
```

**Equation**:

$$
\mathcal{G}(x; \tau) = \frac{1}{1 + e^{-x / \tau}}
$$

$$
AP_q \approx \frac{1}{|\mathcal{S}_P|} \sum_{i \in \mathcal{S}_P} \frac{1 + \sum_{j \in \mathcal{S}_P, j \neq i} \mathcal{G}(D_{ij}; \tau)}{1 + \sum_{j \in \mathcal{S}_P, j \neq i} \mathcal{G}(D_{ij}; \tau) + \sum_{j \in \mathcal{S}_N} \mathcal{G}(D_{ij}; \tau)}
$$

$$
\mathcal{L}_{AP} = \frac{1}{m} \sum_{k=1}^{m} \left(1 - AP_k\right)
$$

Here $D_{ij} = s_j - s_i$, where $s_i$ is the cosine similarity between query $q$ and element $i$; $\mathcal{S}_P$ and $\mathcal{S}_N$ are the positive and negative sets of $q$; and $m$ is the batch size.

**Implementation notes**: The query is counted in its own $\mathcal{S}_P$, as in the authors' implementation.

**Parameters**:

- **temperature**: The desired temperature for scaling the sigmoid function. This is denoted by $\tau$ in the first and second equations.

**Other info**:

- Positive sets come from the labels, so classes may appear in any order and with different counts. Each element is a query against the rest of the batch. Memory grows with the cube of the batch size.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)
  - This is the only compatible distance.
