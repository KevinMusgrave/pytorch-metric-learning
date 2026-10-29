# SupConLoss

Described in [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362){target=_blank}.

```python
losses.SupConLoss(temperature=0.1, **options)
```

**Equation**:

$$
\mathcal{L}^{sup}_{out} = \sum_{i \in I} \frac{-1}{\left|P(i)\right|} \sum_{p \in P(i)} \log \frac{\exp\left(\mathbf{z}_i \cdot \mathbf{z}_p / \tau\right)}{\sum_{a \in A(i)} \exp\left(\mathbf{z}_i \cdot \mathbf{z}_a / \tau\right)}
$$

Here $I$ indexes the batch, $A(i)$ is every index except $i$, and $P(i) \subseteq A(i)$ holds the indices that share $i$'s label.

**Implementation notes**: The default reducer averages over anchors with a positive loss instead of summing. A batch without both a positive and a negative pair yields 0.

**Parameters**:

- **temperature**: This is tau in the above equation. The paper uses 0.1.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [AvgNonZeroReducer](../reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. If an element has only negative pairs or no pairs, it's ignored thanks to `AvgNonZeroReducer`. Reduction type is `"element"`.
