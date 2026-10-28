# CircleLoss

[Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/pdf/2002.10857.pdf){target=_blank}

```python
losses.CircleLoss(m=0.4, gamma=80, **kwargs)
```

**Equation**:

$$
\mathcal{L}_{circle} = \log \left[ 1 + \sum_{j=1}^{L} \exp\left(\gamma \alpha_n^j \left(s_n^j - \Delta_n\right)\right) \sum_{i=1}^{K} \exp\left(-\gamma \alpha_p^i \left(s_p^i - \Delta_p\right)\right) \right]
$$

For each anchor, $s_p^i$ and $s_n^j$ are its cosine similarities to its $K$ positives and $L$ negatives.

where

$$
\begin{aligned}
\alpha_p^i &= \left[O_p - s_p^i\right]_+ \\
\alpha_n^j &= \left[s_n^j - O_n\right]_+
\end{aligned}
$$

$$
\begin{aligned}
O_p &= 1 + m, & O_n &= -m, \\
\Delta_p &= 1 - m, & \Delta_n &= m
\end{aligned}
$$

**Implementation notes**: $\alpha_p$ and $\alpha_n$ are treated as constants for gradients. Anchors without a positive or a negative contribute 0; the default reducer averages the anchors whose loss is positive.

**Parameters**:

- **m**: The relaxation factor that controls the radius of the decision boundary. The paper uses 0.25 for face recognition, and 0.4 for fine-grained image retrieval (images of birds, cars, and online products).
- **gamma**: The scale factor that determines the largest scale of each similarity score. The paper uses 256 for face recognition, and 80 for fine-grained image retrieval.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
