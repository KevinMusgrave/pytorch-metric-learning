# MultiSimilarityLoss

[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}

```python
losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, **kwargs)
```

**Equation**:

$$
\begin{aligned}
\mathcal{L}_{MS} = \frac{1}{m} \sum_{i=1}^{m} \Bigg\lbrace & \frac{1}{\alpha} \log\left[1 + \sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik} - \lambda)}\right] \\
& + \frac{1}{\beta} \log\left[1 + \sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik} - \lambda)}\right] \Bigg\rbrace
\end{aligned}
$$

Here $m$ is the batch size, $S_{ik}$ is the similarity between embeddings $i$ and $k$, and $\mathcal{P}_i$ and $\mathcal{N}_i$ are the positive and negative pairs of anchor $i$.

**Implementation notes**: $\mathcal{P}_i$ and $\mathcal{N}_i$ are all positive and negative pairs of anchor $i$ unless a miner supplies them; the paper mines them with [MultiSimilarityMiner](../miners.md#multisimilarityminer). The default `base=0.5` differs from the paper's $\lambda = 1$; pass `base=1` for the paper's setting.

**Parameters**:

- **alpha**: The weight applied to positive pairs. The paper uses 2.
- **beta**: The weight applied to negative pairs. The paper uses 50.
- **base**: The offset applied to the exponent in the loss. This is lambda in the above equation. The paper uses 1.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.
