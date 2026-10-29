# MultiSimilarityLoss

[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)

```python {.signature}
losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, **options)
```

**Equation**:

$$
\begin{aligned}
\mathcal{L}_{MS} = \frac{1}{m} \sum_{i=1}^{m} \Bigg\lbrace & \frac{1}{\alpha} \log\left[1 + \sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik} - \lambda)}\right] \\
& + \frac{1}{\beta} \log\left[1 + \sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik} - \lambda)}\right] \Bigg\rbrace
\end{aligned}
$$

Here $m$ is the batch size, $S_{ik}$ is the similarity between embeddings $i$ and $k$, and $\mathcal{P}_i$ and $\mathcal{N}_i$ are the positive and negative pairs of anchor $i$.

**Implementation notes**: $\mathcal{P}_i$ and $\mathcal{N}_i$ are all positive
and negative pairs of anchor $i$ unless an `indices_tuple` supplies them; the
paper mines them with [MultiSimilarityMiner](../miners.md#multisimilarityminer).
An anchor without positives or without negatives contributes 0 for that part,
the value of $\log(1 + 0)$. An anchor without any pair has no loss term, so the
default reducer averages over anchors that have pairs. With a
distance where smaller means closer, $S_{ik} - \lambda$ becomes
$\lambda - D_{ik}$.

**Paper settings**: The paper uses $\lambda = 1$; the default `base=0.5`
differs. Pass `base=1` for the paper's setting. The paper divides the sum of
per-anchor losses by the batch size, counting anchors without mined pairs; pass
`reducer=reducers.SumReducer()` and divide the loss by the batch size for that
normalization.

**Example**:

```python
import torch
from pytorch_metric_learning import losses, miners

loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=1)
miner = miners.MultiSimilarityMiner(epsilon=0.1)
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels, miner(embeddings, labels))
loss.backward()
```

**Parameters**:

- **alpha**: The scale $\alpha$ of positive pairs. The paper uses 2.
- **beta**: The scale $\beta$ of negative pairs. The paper uses 50.
- **base**: The offset $\lambda$.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per anchor that has at least one pair, indexed by anchor
  position in `embeddings`.
